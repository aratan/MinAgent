---
name: read-mail
description: 'Leer correos con himalaya. Listado de bandejas, listado y lectura de mensajes del correo configurado. Trigger: "leer mi correo", "leer mis correos", "mailbox list", "mostrar correos", "ver spam", "leer correos de [remitente]".'
---

# read-mail — Leer correos con himalaya (v2)

**Trigger:** `leer mi correo`, `leer mis correos`, `mostrar correos`, `ver spam`.

Asume `himalaya` v2 o superior en el `PATH` y una cuenta ya configurada
(`himalaya account list`). Usa `run_terminal` para ejecutar los comandos.

## Pasos

1. Comprueba la cuenta activa: `himalaya account list`.
2. Lista las bandejas: `himalaya mailbox list`.
3. Lista los mensajes de una bandeja: `himalaya envelope list -m <bandeja> -s <n>`.
4. Lee un mensaje por su `ID`: `himalaya message read <id>`.
5. Presenta remitente, destinatario, fecha, asunto y un resumen del cuerpo. Señala los enlaces relevantes.

## Comandos

```bash
# Cuenta activa
himalaya account list

# Bandejas disponibles (Inbox, Trabajo, spam, ...)
himalaya mailbox list

# Últimos 10 mensajes de Inbox
himalaya envelope list -m Inbox -s 10

# Leer un mensaje concreto por su ID (el ID sale de envelope list)
himalaya message read 270419

# Buscar por remitente dentro de una bandeja
himalaya envelope search -m Inbox from:remite@ejemplo.com
```

## Notas

- En v1 las órdenes eran `himalaya mailbox list`, `himalaya message list` y
  `himalaya message get`. En v2 `message list` y `message get` ya no existen:
  usa `envelope list` para listar y `message read <id>` para leer.
- La bandeja se indica con `-m/--mailbox` (no con `-f`).
- Para limitar el tamaño, usa `-s/--page-size`.
- Si un comando falla, muestra la salida real de himalaya antes de resumir.
