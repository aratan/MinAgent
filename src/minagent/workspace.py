"""Workspace boundaries and the file tools built on top of them.

The agent may only change files inside the directory it was started in, so every
path is resolved through :meth:`WorkspaceAccess.resolve_path` and re-checked
with ``lstat`` immediately before each syscall. Symbolic links, hard links,
special files, and paths that escape the root are rejected; writes go through a
temporary file and an atomic rename, then the persisted bytes are read back and
compared before the tool reports success.
"""

from __future__ import annotations

import base64
import os
import shutil
import stat as stat_module
import time
from typing import Any

from .errors import AgentError, as_agent_error, describe_system_error, is_missing
from .image import detect_image_mime_type
from .jsutil import byte_length, decode_utf8

MAX_READ_BYTES = 10 * 1024 * 1024
MAX_WRITE_BYTES = 10 * 1024 * 1024
MAX_READ_OUTPUT_BYTES = 48 * 1024
MAX_READ_LINES = 300

_MAX_AGENTS_BYTES = 64 * 1024
_MAX_INVENTORY_ENTRIES = 10_000
_MAX_INVENTORY_CHARS = 128 * 1024
_MAX_DIRECTORY_ENTRIES = 10_000
_MAX_DIRECTORY_OUTPUT_BYTES = 50 * 1024

EXCLUDED_DIRECTORIES = {
    ".git", ".hg", ".svn", "node_modules", ".next", ".cache", "dist", "build", "coverage",
}


def _escape_directory_label(value: str) -> str:
    """Escape control characters so a directory name cannot disturb the terminal."""
    return "".join(
        f"\\u{ord(character):04x}" if (ord(character) < 0x20 or 0x7F <= ord(character) < 0xA0) else character
        for character in value
    )


def _entry_kind(entry: os.DirEntry) -> str:
    """Classify a directory entry without following links."""
    try:
        mode = entry.stat(follow_symlinks=False).st_mode
    except OSError:
        return "SPECIAL, not readable"
    if stat_module.S_ISLNK(mode):
        return "LINK, not traversed"
    if stat_module.S_ISDIR(mode):
        return "DIR"
    if stat_module.S_ISREG(mode):
        return "FILE"
    return "SPECIAL, not readable"


def _is_symlink(mode: int) -> bool:
    return stat_module.S_ISLNK(mode)


def _is_dir(mode: int) -> bool:
    return stat_module.S_ISDIR(mode)


def _is_file(mode: int) -> bool:
    return stat_module.S_ISREG(mode)


def _mtime_ms(entry: os.stat_result) -> float:
    return entry.st_mtime_ns / 1_000_000


def _ctime_ms(entry: os.stat_result) -> float:
    return entry.st_ctime_ns / 1_000_000


def _same_file(left: os.stat_result, right: os.stat_result) -> bool:
    return left.st_dev == right.st_dev and left.st_ino == right.st_ino


def _same_version(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        _same_file(left, right)
        and left.st_size == right.st_size
        and _mtime_ms(left) == _mtime_ms(right)
        and _ctime_ms(left) == _ctime_ms(right)
    )


class WorkspaceAccess:
    """All workspace file access, funnelled through the root boundary checks."""

    def __init__(self, root_directory: str, workspace_name: str, list_limit: int = -1) -> None:
        self.root_directory = root_directory
        self.workspace_name = workspace_name
        self.list_limit = list_limit

    # ---------------------------------------------------------------- paths

    def is_within_root(self, candidate: str) -> bool:
        """True when ``candidate`` is the workspace root or below it."""
        try:
            relative = os.path.relpath(candidate, self.root_directory)
        except ValueError:
            return False
        return (
            relative == ""
            or (relative != ".." and not relative.startswith(f"..{os.sep}") and not os.path.isabs(relative))
        )

    def resolve_path(self, value: str, allow_outside: bool = False) -> str:
        """Resolve a model-supplied path, honouring the redundant workspace prefix.

        Started inside ``Test``, ``README.md`` means ``Test/README.md`` and
        ``Test/README.md`` resolves to the same file. A real subdirectory of
        that name wins, and ``./Test/...`` always targets the subdirectory.
        """
        if not isinstance(value, str) or not value or "\0" in value:
            raise AgentError("A non-empty file path is required.")
        absolute = os.path.isabs(value)
        candidate = os.path.normpath(value) if absolute else os.path.normpath(os.path.join(self.root_directory, value))
        if not allow_outside and not self.is_within_root(candidate):
            raise AgentError("Path is outside the current workspace.")

        explicitly_relative = value[:1] in (".",) and len(value) > 1 and value[1] in ("/", os.sep)
        if not absolute and not explicitly_relative:
            parts = [part for part in value.replace(os.sep, "/").split("/") if part and part != "."]
            case_sensitive = os.name != "nt"
            names_workspace = bool(parts) and (
                parts[0] == self.workspace_name
                if case_sensitive
                else parts[0].lower() == self.workspace_name.lower()
            )
            if names_workspace and ".." not in parts:
                # A real child with this name takes precedence over the prefix.
                child_exists = True
                try:
                    os.lstat(os.path.join(self.root_directory, parts[0]))
                except OSError as error:
                    normalized = as_agent_error(error)
                    if normalized.code in ("ENOENT", "ENOTDIR"):
                        child_exists = False
                    else:
                        raise
                if not child_exists:
                    candidate = os.path.normpath(os.path.join(self.root_directory, *parts[1:]))

        if not allow_outside and not self.is_within_root(candidate):
            raise AgentError("Path is outside the current workspace.")
        return candidate

    async def assert_path(self, candidate: str) -> None:
        """Reject the path unless every existing component is a real directory entry."""
        if not self.is_within_root(candidate):
            raise AgentError("Path is outside the current workspace.")
        relative = os.path.relpath(candidate, self.root_directory)
        if relative == "":
            return
        current = self.root_directory
        for part in relative.split(os.sep):
            if not part:
                continue
            current = os.path.join(current, part)
            try:
                entry = os.lstat(current)
            except OSError as error:
                if is_missing(error):
                    return
                raise
            if _is_symlink(entry.st_mode):
                raise AgentError("Symbolic links and junctions are blocked to keep file access inside the workspace.")

    def relative_name(self, target: str) -> str:
        """Workspace-relative display path, always with forward slashes."""
        return os.path.relpath(target, self.root_directory).replace(os.sep, "/")

    async def _regular_file(
        self, target: str, action: str, allow_outside: bool = False
    ) -> os.stat_result:
        """Validate that ``target`` is a regular, single-linked file."""
        if not allow_outside or self.is_within_root(target):
            await self.assert_path(target)
        if target == self.root_directory:
            raise AgentError(
                f"{action} requires a file path; {self.workspace_name} names the workspace directory."
            )
        entry = os.lstat(target)
        if not _is_file(entry.st_mode):
            raise AgentError(f"{action} only works on regular files.")
        if entry.st_nlink > 1:
            raise AgentError("Hard-linked files are blocked to keep access inside the workspace.")
        return entry

    # ---------------------------------------------------------------- reads

    async def _read_regular_buffer(
        self, target: str, action: str, allow_outside: bool = False
    ) -> tuple[os.stat_result, bytes]:
        """Read a file, checking its identity before, during, and after the read."""
        before = await self._regular_file(target, action, allow_outside)
        if before.st_size > MAX_READ_BYTES:
            raise AgentError(f"File is larger than the {MAX_READ_BYTES} byte {action} limit.")
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        try:
            handle = os.open(target, flags)
        except OSError as error:
            raise as_agent_error(error) from error
        try:
            entry = os.fstat(handle)
            if not _is_file(entry.st_mode) or entry.st_nlink > 1 or not _same_file(before, entry):
                raise AgentError("The file changed while it was being opened.")
            resolved = os.path.realpath(target)
            if not allow_outside and not self.is_within_root(resolved):
                raise AgentError("Path resolved outside the current workspace.")
            current = os.lstat(target)
            if not _is_file(current.st_mode) or current.st_nlink > 1 or not _same_file(entry, current):
                raise AgentError("The file changed while it was being opened.")
            with os.fdopen(handle, "rb", closefd=False) as stream:
                data = stream.read()
            if len(data) > MAX_READ_BYTES:
                raise AgentError(f"File grew beyond the {MAX_READ_BYTES} byte {action} limit while being read.")
            after = os.lstat(target)
            if not _is_file(after.st_mode) or not _same_version(current, after):
                raise AgentError("The file changed while it was being read.")
            return after, data
        finally:
            os.close(handle)

    @staticmethod
    def _decode_text(buffer: bytes, action: str) -> str:
        """Decode UTF-8 strictly, rejecting binary payloads."""
        if b"\x00" in buffer:
            raise AgentError(f"{action} cannot process binary files.")
        try:
            return decode_utf8(buffer)
        except UnicodeDecodeError:
            raise AgentError(f"{action} requires a UTF-8 text file.") from None

    async def _read_text(self, target: str, action: str) -> tuple[os.stat_result, bytes, str]:
        entry, buffer = await self._read_regular_buffer(target, action)
        return entry, buffer, self._decode_text(buffer, action)

    async def read_raw_file(self, value: str, allow_outside: bool = False) -> bytes:
        """Read a file's bytes for attachment without decoding it."""
        target = self.resolve_path(value, allow_outside=allow_outside)
        _, buffer = await self._read_regular_buffer(target, "attachment", allow_outside=allow_outside)
        return buffer

    # --------------------------------------------------------------- writes

    async def _write_atomically(
        self, target: str, content: str, previous_entry: os.stat_result | None
    ) -> None:
        """Write through a temporary file, then rename it over the target."""
        data = content.encode("utf-8")
        if len(data) > MAX_WRITE_BYTES:
            raise AgentError(f"File content exceeds the {MAX_WRITE_BYTES} byte write limit.")
        directory = os.path.dirname(target)
        temporary = os.path.join(
            directory,
            f".{os.path.basename(target)}.minagent-{os.getpid()}-{os.urandom(8).hex()}.tmp",
        )
        created = False
        try:
            await self.assert_path(directory)
            mode = (previous_entry.st_mode & 0o777) if previous_entry is not None else 0o666
            descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode)
            created = True
            try:
                if not self.is_within_root(os.path.realpath(temporary)):
                    raise AgentError("Temporary file resolved outside the current workspace.")
                with os.fdopen(descriptor, "wb", closefd=False) as stream:
                    stream.write(data)
                    stream.flush()
                    os.fsync(descriptor)
            finally:
                os.close(descriptor)

            await self.assert_path(target)
            try:
                current = os.lstat(target)
                if _is_symlink(current.st_mode):
                    raise AgentError("Symbolic links and junctions are blocked to keep file access inside the workspace.")
                if _is_dir(current.st_mode):
                    raise AgentError("The target path is a directory.")
                if not _is_file(current.st_mode):
                    raise AgentError("Only regular files can be overwritten.")
                if current.st_nlink > 1:
                    raise AgentError("Hard-linked files are blocked to keep access inside the workspace.")
                if previous_entry is None or not _same_version(previous_entry, current):
                    raise AgentError("The target file changed before it could be replaced.")
            except OSError as error:
                if not is_missing(error):
                    raise
                if previous_entry is not None:
                    raise AgentError("The target file disappeared before it could be replaced.") from None

            await self.assert_path(directory)
            if not self.is_within_root(os.path.realpath(directory)):
                raise AgentError("Target directory resolved outside the current workspace.")
            os.replace(temporary, target)
            created = False
        finally:
            if created:
                try:
                    if self.is_within_root(os.path.realpath(temporary)):
                        os.unlink(temporary)
                except OSError:
                    # The temporary file may already have been removed or moved.
                    pass

    async def _verify_written_text(self, target: str, expected: str) -> None:
        """Read the file back and confirm it holds exactly the requested text."""
        try:
            _, _, content = await self._read_text(target, "write verification")
            if content != expected:
                raise AgentError("Written file does not match the requested content.")
        except AgentError as error:
            error.may_have_changed = True
            raise

    # ----------------------------------------------------------------- tools

    async def list_directory(self, args: dict[str, Any] | None = None) -> dict[str, str]:
        """List one directory's immediate entries, including hidden ones."""
        args = args or {}
        input_path = args.get("path", ".")
        limit = args.get("limit", 500)
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1 or limit > _MAX_DIRECTORY_ENTRIES:
            raise AgentError(f"limit must be an integer from 1 to {_MAX_DIRECTORY_ENTRIES}.")

        target = self.resolve_path(input_path)
        display_path = self.relative_name(target) or "."
        safe_display_path = _escape_directory_label(display_path)
        await self.assert_path(target)

        try:
            directory_entry = os.lstat(target)
        except OSError as error:
            normalized = as_agent_error(error)
            if normalized.code in ("ENOENT", "ENOTDIR"):
                raise AgentError(f'Directory does not exist: "{safe_display_path}".') from None
            raise AgentError(
                f'Could not inspect directory "{safe_display_path}": {describe_system_error(error)}.'
            ) from None
        if _is_symlink(directory_entry.st_mode):
            raise AgentError("Symbolic links and junctions are blocked to keep file access inside the workspace.")
        if not _is_dir(directory_entry.st_mode):
            raise AgentError(f'list_directory requires a directory: "{safe_display_path}".')

        try:
            resolved_directory = os.path.realpath(target)
        except OSError as error:
            raise AgentError(
                f'Could not verify directory "{safe_display_path}": {describe_system_error(error)}.'
            ) from None
        if not self.is_within_root(resolved_directory):
            raise AgentError("Path resolved outside the current workspace.")

        try:
            with os.scandir(target) as scan:
                entries = sorted(scan, key=lambda entry: (entry.name.lower(), entry.name))
                total_entries = len(entries)
                candidates = entries[:limit]
                rendered: list[tuple[str, str, str]] = [
                    (
                        entry.name,
                        _entry_kind(entry),
                        "/" if _is_dir(entry.stat(follow_symlinks=False).st_mode) else "",
                    )
                    for entry in candidates
                ]
        except OSError as error:
            raise AgentError(
                f'Could not list directory "{safe_display_path}": {describe_system_error(error)}.'
            ) from None

        try:
            current_entry = os.lstat(target)
            current_resolved = os.path.realpath(target)
        except OSError as error:
            raise AgentError(
                f'Could not verify directory "{safe_display_path}" after listing: {describe_system_error(error)}.'
            ) from None
        if not _is_dir(current_entry.st_mode) or not _same_file(directory_entry, current_entry) or not self.is_within_root(
            current_resolved
        ):
            raise AgentError("The directory changed while it was being listed.")

        if total_entries == 0:
            return {
                "tool_text": f'Directory: "{safe_display_path}"\n(empty directory)',
                "display_text": f'Listed "{safe_display_path}" · empty directory',
            }

        output_lines = [f'Directory: "{safe_display_path}"']
        output_bytes = byte_length(output_lines[0])
        byte_limit_reached = False
        for name, kind, suffix in rendered:
            line = f"[{kind}] {_escape_directory_label(name)}{suffix}"
            line_bytes = byte_length(line) + 1
            if output_bytes + line_bytes > _MAX_DIRECTORY_OUTPUT_BYTES - 512:
                byte_limit_reached = True
                break
            output_lines.append(line)
            output_bytes += line_bytes

        shown_entries = len(output_lines) - 1
        omitted_entries = total_entries - shown_entries
        if omitted_entries > 0:
            if byte_limit_reached:
                output_lines.append(
                    f"[Output capped at {_MAX_DIRECTORY_OUTPUT_BYTES} bytes; {omitted_entries} of "
                    f"{total_entries} entries were omitted. List a subdirectory to narrow the results.]"
                )
            elif limit == _MAX_DIRECTORY_ENTRIES:
                output_lines.append(
                    f"[Showing {shown_entries} of {total_entries} entries; list a subdirectory to inspect "
                    f"the remainder beyond the {_MAX_DIRECTORY_ENTRIES}-entry limit.]"
                )
            else:
                output_lines.append(
                    f"[Showing {shown_entries} of {total_entries} entries. Call list_directory with a larger "
                    f"limit to see more.]"
                )
        return {
            "tool_text": "\n".join(output_lines),
            "display_text": f'Listed "{safe_display_path}" · {shown_entries}/{total_entries} entries'
            + (" · truncated" if omitted_entries > 0 else ""),
        }

    async def read_file(
        self, args: dict[str, Any], image_enabled: bool = False
    ) -> str | dict[str, Any]:
        """Read a text file, continuing within a long line via offset/column."""
        target = self.resolve_path(args.get("path"), allow_outside=True)
        _, buffer = await self._read_regular_buffer(target, "read_file", allow_outside=True)

        image_mime_type = detect_image_mime_type(buffer)
        if image_mime_type:
            if not image_enabled:
                raise AgentError("The configured model does not accept images.")
            return {
                "tool_text": f"Read image file [{image_mime_type}] {args.get('path')}",
                "image": {
                    "path": args.get("path"),
                    "mime_type": image_mime_type,
                    "data": base64.b64encode(buffer).decode("ascii"),
                },
            }
        path_text = str(args.get("path"))
        if path_text.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")):
            raise AgentError("Image format not recognized. Supported images are PNG, JPEG, GIF, and WebP.")

        content = self._decode_text(buffer, "read_file")
        lines = content.split("\n")
        offset = args.get("offset", 1)
        column = args.get("column", 1)
        limit = min(args.get("limit", MAX_READ_LINES), MAX_READ_LINES)
        for name, value in (("offset", offset), ("column", column), ("limit", limit)):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise AgentError(f"{name} must be an integer of at least 1.")
        if offset > len(lines):
            raise AgentError(f"offset is beyond the end of the file ({len(lines)} lines).")

        output = ""
        output_bytes = 0
        returned_lines = 0
        content_budget = MAX_READ_OUTPUT_BYTES - 160
        last_index = min(len(lines), offset - 1 + limit)
        for line_index in range(offset - 1, last_index):
            line = lines[line_index]
            start_column = column if line_index == offset - 1 else 1
            prefix = "" if returned_lines == 0 else "\n"
            fragment = ""
            fragment_bytes = 0
            current_column = 1
            for character in line:
                if current_column < start_column:
                    current_column += 1
                    continue
                character_bytes = byte_length(character)
                if output_bytes + byte_length(prefix) + fragment_bytes + character_bytes > content_budget:
                    output += f"{prefix}{fragment}"
                    return (
                        f"{output}\n\n[Read stopped at the output limit. Continue with "
                        f"offset={line_index + 1}, column={current_column}.]"
                    )
                fragment += character
                fragment_bytes += character_bytes
                current_column += 1
            if start_column > current_column:
                raise AgentError(f"column is beyond the end of line {line_index + 1}.")
            if output_bytes + byte_length(prefix) + fragment_bytes > content_budget:
                return (
                    f"{output}\n\n[Read stopped at the output limit. Continue with "
                    f"offset={line_index + 1}, column={start_column}.]"
                )
            output += f"{prefix}{fragment}"
            output_bytes += byte_length(prefix) + fragment_bytes
            returned_lines += 1

        next_offset = offset + returned_lines
        if next_offset <= len(lines):
            output += f"\n\n[{len(lines) - next_offset + 1} more lines. Continue with offset={next_offset}.]"
        return output

    async def edit_file(self, args: dict[str, Any]) -> str:
        """Replace one exact, unique text block in an existing file."""
        old_text = args.get("old_text")
        new_text = args.get("new_text")
        if not isinstance(old_text, str) or not old_text:
            raise AgentError("old_text must be a non-empty string.")
        if not isinstance(new_text, str):
            raise AgentError("new_text must be a string.")
        if byte_length(old_text) > MAX_WRITE_BYTES or byte_length(new_text) > MAX_WRITE_BYTES:
            raise AgentError(
                f"old_text and new_text must each fit within the {MAX_WRITE_BYTES} byte edit limit."
            )
        path_text = str(args.get("path"))
        target = self.resolve_path(path_text)
        entry, _, content = await self._read_text(target, "edit_file")
        first_index = content.find(old_text)
        if first_index < 0:
            raise AgentError(
                f"old_text was not found in {path_text}; no changes were made. Reread this path with "
                f"read_file, then rebuild the edit from its current contents."
            )
        if content.find(old_text, first_index + len(old_text)) >= 0:
            raise AgentError(
                f"old_text occurs more than once in {path_text}; no changes were made. Reread this path "
                f"with read_file and choose a unique exact text block."
            )
        changed = content[:first_index] + new_text + content[first_index + len(old_text):]
        await self._write_atomically(target, changed, entry)
        await self._verify_written_text(target, changed)
        return f"Updated {path_text}."

    async def write_file(self, args: dict[str, Any]) -> str:
        """Create or atomically replace a UTF-8 file, creating parent folders."""
        content = args.get("content")
        if not isinstance(content, str):
            raise AgentError("content must be a string.")
        if byte_length(content) > MAX_WRITE_BYTES:
            raise AgentError(f"File content exceeds the {MAX_WRITE_BYTES} byte write limit.")
        path_text = str(args.get("path"))
        if path_text.rstrip().endswith(("/", os.sep)):
            raise AgentError("write_file writes files. Use create_directory to create a folder.")
        target = self.resolve_path(path_text)
        await self.assert_path(target)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        await self.assert_path(target)

        previous_entry: os.stat_result | None = None
        try:
            previous_entry = os.lstat(target)
            if _is_symlink(previous_entry.st_mode):
                raise AgentError("Symbolic links and junctions are blocked to keep file access inside the workspace.")
            if _is_dir(previous_entry.st_mode):
                raise AgentError("The target path is a directory.")
            if not _is_file(previous_entry.st_mode):
                raise AgentError("Only regular files can be overwritten.")
            if previous_entry.st_nlink > 1:
                raise AgentError("Hard-linked files are blocked to keep access inside the workspace.")
        except OSError as error:
            if not is_missing(error):
                raise

        await self._write_atomically(target, content, previous_entry)
        await self._verify_written_text(target, content)
        return f"Wrote {path_text}."

    async def create_directory(self, args: dict[str, Any]) -> str:
        """Create a directory and any missing parents inside the workspace."""
        path_text = str(args.get("path"))
        target = self.resolve_path(path_text)
        await self.assert_path(target)
        try:
            entry: os.stat_result | None = os.lstat(target)
        except OSError as error:
            if not is_missing(error):
                raise
            entry = None
        if entry is not None:
            if _is_symlink(entry.st_mode):
                raise AgentError("Symbolic links and junctions are blocked to keep file access inside the workspace.")
            if _is_dir(entry.st_mode):
                return f"Directory already exists: {path_text}."
            raise AgentError(f"{path_text} already exists and is not a directory.")
        os.makedirs(target, exist_ok=True)
        created = os.lstat(target)
        if _is_symlink(created.st_mode) or not _is_dir(created.st_mode):
            raise AgentError("The target path is not a regular directory.")
        if not self.is_within_root(os.path.realpath(target)):
            raise AgentError("Directory resolved outside the current workspace.")
        return f"Created directory {path_text}."

    async def delete_file(self, args: dict[str, Any]) -> str:
        """Delete one regular file inside the workspace."""
        path_text = str(args.get("path"))
        target = self.resolve_path(path_text)
        await self.assert_path(target)
        entry = os.lstat(target)
        if _is_dir(entry.st_mode):
            raise AgentError("Directories cannot be deleted. delete_file removes files only.")
        if _is_symlink(entry.st_mode):
            raise AgentError("Symbolic links and junctions are blocked to keep file access inside the workspace.")
        if not _is_file(entry.st_mode):
            raise AgentError("Only regular files can be deleted.")
        if entry.st_nlink > 1:
            raise AgentError("Hard-linked files are blocked to keep access inside the workspace.")
        if not self.is_within_root(os.path.realpath(os.path.dirname(target))):
            raise AgentError("Parent directory resolved outside the current workspace.")
        current = os.lstat(target)
        if not _is_file(current.st_mode) or not _same_version(entry, current):
            raise AgentError("The file changed before it could be deleted.")
        os.unlink(target)
        return f"Deleted {path_text}."

    async def _validate_deletable_directory(self, directory_path: str) -> None:
        """Reject a directory tree containing links, special files, or hard links."""
        entry = os.lstat(directory_path)
        if _is_symlink(entry.st_mode):
            raise AgentError("Symbolic links and junctions are blocked inside deletable directories.")
        if _is_dir(entry.st_mode):
            with os.scandir(directory_path) as scan:
                children = sorted(scan, key=lambda item: item.name)
            for child in children:
                await self._validate_deletable_directory(os.path.join(directory_path, child.name))
            return
        if not _is_file(entry.st_mode):
            raise AgentError("Directories containing special files cannot be deleted.")
        if entry.st_nlink > 1:
            raise AgentError("Hard-linked files are blocked to keep access inside the workspace.")

    async def delete_directory(self, args: dict[str, Any]) -> str:
        """Recursively delete a validated subdirectory; never the workspace root."""
        path_text = str(args.get("path"))
        target = self.resolve_path(path_text)
        is_root = (
            target.lower() == self.root_directory.lower()
            if os.name == "nt"
            else target == self.root_directory
        )
        if is_root:
            raise AgentError("The workspace root cannot be deleted.")
        await self.assert_path(target)
        entry = os.lstat(target)
        if not _is_dir(entry.st_mode) or _is_symlink(entry.st_mode):
            raise AgentError("delete_directory only works on a regular subdirectory.")
        await self._validate_deletable_directory(target)
        if not self.is_within_root(os.path.realpath(target)):
            raise AgentError("Directory resolved outside the current workspace.")
        current = os.lstat(target)
        if not _is_dir(current.st_mode) or not _same_file(entry, current):
            raise AgentError("The directory changed before it could be deleted.")
        for attempt in range(3):
            try:
                shutil.rmtree(target)
                break
            except FileNotFoundError:
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(0.1)
        return f"Deleted directory {path_text} and its contents."

    # ------------------------------------------------------------- inventory

    async def refresh_inventory(
        self,
        include_snapshot: bool | None = None,
        list_limit_override: int | None = None,
    ) -> dict[str, Any]:
        """Walk the workspace for the prompt inventory and ``@`` autocomplete index.

        A disabled prompt inventory still builds the bounded local file index.
        """
        if include_snapshot is None:
            include_snapshot = self.list_limit != 0
        if list_limit_override is None:
            list_limit_override = self.list_limit

        traversal_limit = -1 if (list_limit_override == 0 and not include_snapshot) else list_limit_override
        lines = (
            [
                "## Workspace inventory (paths only; generated directories excluded)",
                f'Per-directory limit: {"unlimited" if list_limit_override == -1 else list_limit_override}',
            ]
            if include_snapshot
            else []
        )
        file_paths: list[str] = []
        omitted_entries = 0
        visited_entries = 0
        listed_chars = len("\n".join(lines))
        inventory_full = False

        def add_line(line: str) -> None:
            nonlocal listed_chars, inventory_full
            if not include_snapshot:
                return
            if listed_chars + len(line) + 1 > _MAX_INVENTORY_CHARS:
                inventory_full = True
                return
            lines.append(line)
            listed_chars += len(line) + 1

        async def append_directory(directory_path: str, indent: str) -> None:
            nonlocal omitted_entries, visited_entries, inventory_full
            if (include_snapshot and inventory_full) or visited_entries >= _MAX_INVENTORY_ENTRIES:
                return
            try:
                with os.scandir(directory_path) as scan:
                    entries = sorted(scan, key=lambda item: item.name)
            except OSError as error:
                add_line(f"{indent}[Could not list this directory: {describe_system_error(error)}]")
                return

            kept: list[os.DirEntry] = []
            for entry in entries:
                try:
                    mode = entry.stat(follow_symlinks=False).st_mode
                except OSError:
                    kept.append(entry)
                    continue
                if _is_dir(mode) and entry.name.lower() in EXCLUDED_DIRECTORIES:
                    continue
                kept.append(entry)

            visible = kept if traversal_limit == -1 else kept[:traversal_limit]
            omitted_entries += len(kept) - len(visible)
            for entry in visible:
                if (include_snapshot and inventory_full) or visited_entries >= _MAX_INVENTORY_ENTRIES:
                    inventory_full = True
                    break
                visited_entries += 1
                child_path = os.path.join(directory_path, entry.name)
                try:
                    mode = entry.stat(follow_symlinks=False).st_mode
                except OSError as error:
                    add_line(f"{indent}[Could not inspect this entry: {describe_system_error(error)}]")
                    continue
                if _is_symlink(mode):
                    add_line(f"{indent}[LINK, not traversed] {entry.name}")
                elif _is_dir(mode):
                    add_line(f"{indent}[DIR] {entry.name}/")
                    await append_directory(child_path, f"{indent}  ")
                elif _is_file(mode):
                    file_paths.append(self.relative_name(child_path))
                    add_line(f"{indent}[FILE] {entry.name}")
                else:
                    add_line(f"{indent}[SPECIAL, not readable] {entry.name}")

        if traversal_limit != 0:
            await append_directory(self.root_directory, "")
        if include_snapshot and omitted_entries > 0:
            lines.append(f"[{omitted_entries} entries omitted by WORKSPACE_LIST_LIMIT]")
        if include_snapshot and inventory_full:
            lines.append(
                f"[Inventory stopped at {_MAX_INVENTORY_ENTRIES} entries or {_MAX_INVENTORY_CHARS} characters.]"
            )

        guidance = ""
        agents_content = ""
        agents_exists = False
        try:
            target = os.path.join(self.root_directory, "AGENTS.md")
            await self.assert_path(target)
            entry = os.lstat(target)
            if not _is_file(entry.st_mode) or entry.st_nlink > 1:
                guidance = (
                    "## AGENTS.md project guidance\n"
                    "AGENTS.md is not a regular unlinked file and could not be loaded."
                )
            elif entry.st_size > _MAX_AGENTS_BYTES:
                guidance = (
                    f"## AGENTS.md project guidance\nAGENTS.md exceeds the {_MAX_AGENTS_BYTES} byte limit."
                )
            else:
                _, buffer = await self._read_regular_buffer(target, "AGENTS.md")
                if len(buffer) > _MAX_AGENTS_BYTES:
                    raise AgentError("AGENTS.md grew beyond its size limit.")
                agents_content = self._decode_text(buffer, "AGENTS.md")
                agents_exists = True
                guidance = (
                    "## AGENTS.md project guidance (reloaded before each model request)\n" + agents_content
                )
        except OSError as error:
            if not is_missing(error):
                guidance = (
                    "## AGENTS.md project guidance\n"
                    f"Could not load AGENTS.md: {describe_system_error(error)}"
                )
        except AgentError as error:
            guidance = (
                "## AGENTS.md project guidance\n"
                f"Could not load AGENTS.md: {error.code or error.message}"
            )

        return {
            "snapshot": "\n".join(lines) if include_snapshot else "",
            "files": sorted(file_paths, key=str.lower),
            "agents_context": guidance,
            "agents_content": agents_content,
            "agents_exists": agents_exists,
        }
