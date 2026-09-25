---
name: abrir-navegador
description: Abre el navegador web local con la página por defecto
---

# Abre el navegador local

1. Detecta el navegador disponible (chrome, chromium, firefox, safari, edge, msedge, o el navegador de sistema `system-browser`)
2. Abre la URL o la página de inicio del navegador
3. Usa la herramienta `run_terminal` para ejecutar el comando de apertura (ej. `xdg-open`, `open`, `start`)

## Parámetros

- **url**: (opcional) URI/URL que se debe abrir. Si no se proporciona, se abre la página de inicio del navegador.

## Ejemplo de uso

```
abrir-navegador
abrir-navegador url=https://ejemplo.com
```

## Detección del navegador

1. **Detección del navegador preferido:**
   - `chrome` / `chromium`: `google-chrome` o `google-chromium`
   - `firefox`: `firefox`
   - `safari`: `safari` (macOS)
   - `edge` / `msedge`: `edge` or `msedge`
   - Other: usar `system-browser`

2. **Apertura de la URL:**
   - Si se proporciona una URL, se abre con esa dirección.
   - Si no hay URL, se abre la página de inicio del navegador.

## Ejemplos:
- `abrir-navegador`: abre el navegador principal sin URL.
- `abrir-navegador url=https://ejemplo.com`: abre `https://ejemplo.com`.
- `abrir-navegador url=https://youtube.com/watch?v=ejemplo`: abre un video de YouTube.
