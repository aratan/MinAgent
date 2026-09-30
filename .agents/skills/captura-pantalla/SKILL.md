---
name: captura-pantalla
description: Captura la pantalla completa en KDE Wayland con Spectacle en segundo plano (sin selector gráfico) y guarda el PNG en el directorio capturas/ del proyecto. Usar cuando pidan capturar la pantalla, hacer un screenshot o mirar lo que hay en pantalla.
allowed-tools:
  - bash
---

# Capturar la pantalla

Captura la pantalla completa y la guarda en `capturas/`, dentro del proyecto. La captura se hace
en segundo plano: no aparece la ventana de Spectacle ni ningún selector gráfico.

Usa la herramienta `run_terminal` (bash) para ejecutar el script de esta skill.

## Uso

Sin argumentos, la captura se llama con la hora:

```bash
.agents/skills/captura-pantalla/scripts/capturar.sh
```

Con un nombre propio (solo un nombre de fichero, sin carpetas):

```bash
.agents/skills/captura-pantalla/scripts/capturar.sh error-terminal.png
```

El script imprime la ruta final y el tamaño en bytes cuando termina bien:

```
OK: <raíz-del-proyecto>/capturas/captura-20260927-214534.png (344948 bytes)
```

Devuelve esa ruta al usuario. No hace falta volver a capturar para preguntas seguidas sobre la
misma imagen.

## Qué hace el script

1. **Deriva la raíz del proyecto** desde la ubicación del propio script
   (`.agents/skills/captura-pantalla/scripts/` → tres niveles arriba), así que la skill sigue
   funcionando si el proyecto se mueve o se clona en otra carpeta. No hay rutas absolutas fijas.
2. **Crea `capturas/`** si no existe, y **rechaza** que sea un symlink, para que la captura no
   acabe fuera del proyecto.
3. **Detecta la sesión gráfica**: usa `WAYLAND_DISPLAY` si su socket existe, si no prueba
   `wayland-0`, y si no cae a X11 (`DISPLAY` + `import`).
4. **Captura con Spectacle** en modo background y pantalla completa; si Spectacle no está o
   falla, **recurre a grim**.
5. **Verifica el resultado**: un código de salida 0 no basta, porque Spectacle puede terminar sin
   escribir, o escribir un contenedor con otra extensión. Comprueba que el fichero existe y no
   está vacío antes de darla por buena.

## Herramientas que usa

- `spectacle` (KDE) en Wayland: `-b` en segundo plano, `-f` pantalla completa, `-o` salida.
- `grim` como fallback en Wayland.
- `import` de ImageMagick si la sesión es X11.

## Comprobaciones y errores

El script sale con código 1 y un mensaje claro si:

- no hay sesión gráfica detectable;
- el nombre lleva `/`, `\` o empieza por `.`;
- `capturas/` es un symlink;
- la captura no llegó a escribirse (en ese caso lista el directorio para diagnosticar).

## Notas

- Las capturas van a `capturas/`, separadas de `salida/`, que es la carpeta que usa la
  herramienta de descargas. No mezcles las dos: `salida/` es solo para descargas.
- Spectacle guarda en formato PNG, pero según la versión puede escribir un contenedor con
  extensión distinta. El script comprueba el fichero real en disco, no el nombre.

## Ejemplos de uso

- "captura la pantalla"
- "haz una captura y dime qué hay en pantalla"
- "captura y revisa si hay algún error en la terminal"
