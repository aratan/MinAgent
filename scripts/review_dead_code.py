"""Report the code nothing calls, and fail on anything new.

Vulture is a good detector and a poor judge. Pointed at this repository it
finds three kinds of thing, and treating them alike would make it useless:

- **Genuinely dead**: a function nobody calls, a constant left behind by a
  design that changed. This is what the script exists to report.
- **Framework callbacks**: ``handle_starttag`` on an ``HTMLParser``,
  ``do_POST`` on an ``http.server`` handler, ``detach`` on a
  ``threading.Thread``. They are called by the framework, not by us, and no
  amount of grepping will prove otherwise.
- **Dynamic lookups**: ``_improvement_setting("min_runs", 9)`` reads
  ``config.improvement_min_runs`` by a name built at runtime, which no static
  tool can follow. Two of those settings are the evidence gate, and a report
  claiming otherwise would be worse than no report.

So the allowlist below is explicit and commented, and anything not on it is a
finding. Run it with ``uv run python scripts/review_dead_code.py``.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

#: Names the standard library calls for us. Matched against ``path:line: kind
#: 'name'``, so a same-named method of ours would need renaming to slip past.
ALLOWED = {
    # HTMLParser calls these for every tag it meets.
    "handle_starttag": "HTMLParser callback",
    "handle_endtag": "HTMLParser callback",
    "handle_data": "HTMLParser callback",
    "handle_entityref": "HTMLParser callback",
    "handle_charref": "HTMLParser callback",
    # http.server dispatches by method name.
    "do_GET": "http.server callback",
    "do_POST": "http.server callback",
    "do_DELETE": "http.server callback",
    "do_PUT": "http.server callback",
    # threading.Thread and subprocess hooks.
    "log_message": "http.server log hook",
    "daemon_threads": "ThreadingHTTPServer attribute",
    "detach": "threading.Thread method",
    "cpu": "threading.Thread.cpu override in a stub",
    # Written and read inside the class; vulture cannot see a same-class read
    # through the instance it just assigned.
    "_multiline": "LineEditor state read from its own methods",
    "job": "ComputeJob field read through the dataclass",
    # Set by the interpreter, read by the interpreter.
    "row_factory": "sqlite3.Connection attribute",
    # Read through getattr with a name built at runtime.
    "improvement_min_runs": "read via _improvement_count()",
    "improvement_max_regressions": "read via _improvement_count()",
    "improvement_holdout_fraction": "read via _improvement_setting()",
    "improvement_lesson_budget": "read via _improvement_count()",
}

#: Only these are worth reporting. Unused *locals* are style; unused functions,
#: methods, properties, classes and module constants are what rot.
KINDS = ("function", "method", "property", "class", "attribute")

LINE = re.compile(r"^(?P<path>[^:]+):(?P<line>\d+): unused (?P<kind>\w+) '(?P<name>[^']+)'")


def vulture_findings() -> list[str]:
    """Run vulture over the package *and* the tests.

    Both, deliberately: a function only the tests call is not dead code, it is
    a function with a smaller audience than its name suggests, and running
    without the tests would report all of them as unused.
    """
    completed = subprocess.run(
        [sys.executable, "-m", "vulture", "src/minagent", "tests", "--min-confidence", "60"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    # Vulture exits 0 when it finds nothing and 3 when it does. Guessing 1 here
    # would have turned a clean run into a silent pass: the crash message is not
    # a vulture line, the parser drops it, and the script reports "nothing dead"
    # over a run that never happened.
    if completed.returncode not in (0, 3):
        sys.stderr.write(completed.stderr)
        raise SystemExit(f"vulture could not run (exit {completed.returncode})")
    return [line for line in completed.stdout.splitlines() if line.strip()]


def report() -> tuple[list[str], list[str]]:
    """Return the findings worth acting on, and the allowlist hits."""
    findings: list[str] = []
    allowed: list[str] = []
    for line in vulture_findings():
        match = LINE.match(line)
        if match is None or match.group("kind") not in KINDS:
            continue  # an unused local, or a line vulture could not parse
        if match.group("name") in ALLOWED:
            allowed.append(line)
        else:
            findings.append(line)
    return findings, allowed


def main() -> int:
    findings, allowed = report()
    for line in allowed:
        name = LINE.match(line).group("name") if LINE.match(line) else ""
        print(f"allowed  {line}  ({ALLOWED.get(name, '')})")
    for line in findings:
        print(f"DEAD     {line}")
    if findings:
        print(
            f"\n{len(findings)} thing(s) nothing calls. Delete them, or - if the name "
            "belongs on the allowlist in scripts/review_dead_code.py - say why there."
        )
        return 1
    print(f"\nNothing dead. {len(allowed)} allowlisted callback(s) skipped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())