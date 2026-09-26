---
name: himalaya-cli
description: 'Gestionar correos con la CLI himalaya v2: cuentas, bandejas, listado, búsqueda con la DSL de consultas, lectura de mensajes y envío. Trigger: "leer mi correo", "buscar correos de", "ver spam", "envía un correo a", "listar mis bandejas".'
---

# himalaya-cli — Correo con himalaya v2

Asume `himalaya` v2 en el `PATH` y una cuenta ya configurada
(`himalaya account list`). Usa `run_terminal` para ejecutar los comandos.

Si el servidor MCP `himalaya` está configurado en `.minagent/mcp.json`, sus
herramientas (`listar_buzones`, `listar_mensajes`, `buscar_mensajes`,
`leer_mensaje`, `enviar_correo`) hacen lo mismo y devuelven JSON ya parseado.
Prefiere esas herramientas cuando estén disponibles.

## Comandos

```bash
# Cuentas definidas y cuál está activa
himalaya account list

# Bandejas disponibles (id, nombre, no leídos)
himalaya mailbox list

# Sobres de una bandeja, del más reciente al más antiguo
himalaya envelope list -m Inbox -s 10

# Buscar: cada condición va como argumento propio
himalaya envelope search -m Inbox from gmail.com
himalaya envelope search -s 20 subject factura and after 2026-01-01
himalaya envelope search -s 20 order by date desc

# Leer un mensaje por su id (el id sale de envelope list / envelope search)
himalaya message read 270419

# Marcar como leído al leerlo
himalaya message read --seen 270419

# Enviar: el cuerpo es un fichero, una cadena raw o entrada estándar
himalaya message send --save "Enviado" < correo.eml
```

Añade `--json` a cualquier comando de listado para obtener JSON en vez de tabla.

## DSL de búsqueda

Condiciones: `date <yyyy-mm-dd>`, `after <yyyy-mm-dd>`, `from <patrón>`,
`to <patrón>`, `subject <patrón>`, `body <patrón>`,
`flag <seen|answered|flagged|draft>`. Se combinan con `and`, `or`, `not` y
paréntesis. Ordenación: `order by <date|from|to|subject> [asc|desc]`.

## Migración desde v1

Estos comandos de v1 **no existen** en v2:

| v1 (eliminado) | v2 |
| --- | --- |
| `himalaya message list` | `himalaya envelope list` |
| `himalaya message get <id>` | `himalaya message read <id>` |
| `himalaya envelopes list` | `himalaya envelope list` |
| `himalaya search ...` | `himalaya envelope search ...` |
| `himalaya send ...` | `himalaya message send ...` |
| `-f/--folder` | `-m/--mailbox` |
| `--flags` | `envelope search flag <nombre>` |

himalaya v2 **no tiene** creación de filtros: el filtrado es por consulta
(`envelope search`) y se persiste con las reglas del backend IMAP.

## Notas

- Los ids de mensaje son opacos y **cambian entre cuentas y backends**: usa el
  id que devuelve el listado de esa misma bandeja, no uno anotado antes.
- Si un comando falla, muestra la salida real de himalaya antes de resumir; su
  JSON de error ya es accionable.
- `himalaya message send` **no tiene vuelta atrás**: confirma con el usuario el
  destinatario y el cuerpo antes de enviar.
