#!/usr/bin/env bash
# Captura la pantalla completa y la guarda en <proyecto>/capturas/.
#
# La raiz del proyecto se deriva de la ubicacion de este script
# (.agents/skills/captura-pantalla/scripts/ -> tres niveles arriba), no de una
# ruta absoluta fija: asi el skill sigue funcionando si el proyecto se mueve o
# se clona en otra carpeta.

set -euo pipefail

# --- Raiz del proyecto -------------------------------------------------------
# scripts/ -> captura-pantalla/ -> skills/ -> .agents/ -> <proyecto>
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd -- "${script_dir}/../../../.." && pwd)"

# Un symlink en capturas/ apuntando fuera del proyecto haria que la captura
# acabase en un sitio arbitrario, asi que se rechaza en vez de seguirlo.
output_dir="${project_root}/capturas"
if [ -L "${output_dir}" ]; then
  echo "ERROR: ${output_dir} es un symlink; se niega escribir fuera del proyecto." >&2
  exit 1
fi
mkdir -p -- "${output_dir}"

# --- Nombre de salida --------------------------------------------------------
# Acepta un nombre como primer argumento; si no, uno con timestamp.
if [ "$#" -ge 1 ] && [ -n "${1}" ]; then
  name="${1}"
else
  name="captura-$(date +%Y%m%d-%H%M%S).png"
fi

# Solo se admiten nombres de fichero, sin separadores de ruta ni "..".
case "${name}" in
  */* | *\\* | .* )
    echo "ERROR: nombre de captura invalido: ${name}" >&2
    exit 1
    ;;
esac

target="${output_dir}/${name}"
rm -f -- "${target}"

# --- Sesion grafica ----------------------------------------------------------
# run_terminal se hereda de una terminal que puede no tener el socket Wayland
# exportado, asi que se detectan los disponibles en vez de fijarlos.
if [ -n "${WAYLAND_DISPLAY:-}" ] && [ -S "${XDG_RUNTIME_DIR:-/tmp}/${WAYLAND_DISPLAY}" ]; then
  session_display="${WAYLAND_DISPLAY}"
elif [ -S "${XDG_RUNTIME_DIR:-/tmp}/wayland-0" ]; then
  session_display="wayland-0"
elif [ -n "${WAYLAND_DISPLAY:-}" ]; then
  session_display="${WAYLAND_DISPLAY}"
elif [ -n "${DISPLAY:-}" ]; then
  session_display="x11"
else
  echo "ERROR: no se detecta sesion grafica (ni WAYLAND_DISPLAY ni DISPLAY)." >&2
  exit 1
fi

echo "Sesion: ${session_display}"

# --- Captura -----------------------------------------------------------------
# Se registra el error de cada intento en vez de abortar, para poder encadenar
# el fallback en lugar de imprimir un error y un codigo de salida a la vez.
status=1

if [ "${session_display}" = "x11" ]; then
  echo "Sesion X11: se usa import de ImageMagick."
  if command -v import >/dev/null 2>&1; then
    if import -window root "${target}" 2>&1; then status=0; fi
  else
    echo "  import no esta instalado." >&2
  fi
else
  export QT_QPA_PLATFORM=wayland
  export WAYLAND_DISPLAY="${session_display}"

  if command -v spectacle >/dev/null 2>&1; then
    echo "Capturando con Spectacle (KDE, en segundo plano, pantalla completa)..."
    if spectacle -b -f -o "${target}" 2>&1; then status=0; fi
  fi

  if [ "${status}" -ne 0 ] && command -v grim >/dev/null 2>&1; then
    echo "Spectacle no ha funcionado; fallback a grim."
    if grim "${target}" 2>&1; then status=0; fi
  fi
fi

# --- Verificacion ------------------------------------------------------------
# Un codigo de salida 0 no basta: Spectacle puede terminar bien sin escribir, o
# escribir un contenedor con otra extension. Se comprueba el fichero en disco.
if [ "${status}" -eq 0 ] && [ -s "${target}" ]; then
  size="$(wc -c < "${target}" | tr -d ' ')"
  echo "OK: ${target} (${size} bytes)"
  exit 0
fi

echo "ERROR: no se ha generado la captura en ${target}" >&2
ls -la -- "${output_dir}" >&2 || true
exit 1
