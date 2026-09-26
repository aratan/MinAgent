#!/usr/bin/env node
/**
 * Himalaya MCP Server (himalaya v2).
 *
 * Expone la CLI `himalaya` como herramientas MCP sobre STDIN/STDOUT, que es el
 * transporte que MinAgent espera para un servidor stdio.
 *
 * Notas de diseño:
 *   - Nunca se invoca un shell. Se usa execFile/spawn con arrays de argumentos,
 *     así que un asunto o un remitente con comillas o `;` no puede inyectar
 *     comandos.
 *   - Todo el logging va a stderr, porque stdout transporta el JSON-RPC.
 *   -himalaya v2 no tiene `list mailboxes`, `read` ni `send` sueltos: el listado
 *     de mensajes es `envelope list`, la lectura es `message read <id>` y el
 *     envío es `message send`.
 */

import { spawn } from "node:child_process";
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

const COMMAND = "himalaya";
/** Un comando de red no debería tardar más que esto. */
const TIMEOUT_MS = 60_000;
/** Salida máxima que se devuelve al modelo; el resto se indica como recortado. */
const MAX_RESULT_CHARS = 24_000;

const server = new McpServer({
  name: "himalaya",
  version: "2.0.0",
});

/**
 * Ejecuta himalaya sin shell y devuelve stdout.
 *
 * @param {string[]} args Argumentos ya separados, sin quoting.
 * @param {{ input?: string, timeoutMs?: number }} options
 * @returns {Promise<{ stdout: string, stderr: string }>}
 */
function runHimalaya(args, { input, timeoutMs = TIMEOUT_MS } = {}) {
  return new Promise((resolve, reject) => {
    // `shell: false` es el valor por defecto, pero se deja explícito porque toda
    // la seguridad de este servidor depende de ello.
    const child = spawn(COMMAND, args, { shell: false, stdio: ["pipe", "pipe", "pipe"] });

    let stdout = "";
    let stderr = "";
    let settled = false;

    const timer = setTimeout(() => {
      child.kill("SIGKILL");
      reject(new Error(`\`${COMMAND} ${args.join(" ")}\` superó el límite de ${timeoutMs} ms.`));
    }, timeoutMs);
    timer.unref?.();

    child.stdout.on("data", (chunk) => {
      stdout += chunk.toString("utf8");
    });
    child.stderr.on("data", (chunk) => {
      stderr += chunk.toString("utf8");
    });
    child.on("error", (error) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      reject(error);
    });
    child.on("close", (code) => {
      if (settled) return;
      settled = true;
      clearTimeout(timer);
      if (code === 0) {
        resolve({ stdout, stderr });
        return;
      }
      // Se propaga el error real de himalaya: un mensaje sin configurar una
      // cuenta es accionable, un "falló la herramienta" genérico no lo es.
      const detail = (stderr || stdout).trim() || `código de salida ${code}`;
      reject(new Error(`\`${COMMAND} ${args.join(" ")}\` falló: ${detail}`));
    });

    if (input !== undefined) {
      child.stdin.end(input, "utf8");
    } else {
      child.stdin.end();
    }
  });
}

/** Recorta la salida para no reventar el contexto del modelo. */
function clip(text) {
  if (text.length <= MAX_RESULT_CHARS) return text;
  const head = Math.floor(MAX_RESULT_CHARS * 0.6);
  const tail = MAX_RESULT_CHARS - head;
  const omitted = text.length - MAX_RESULT_CHARS;
  return `${text.slice(0, head)}\n\n[...${omitted} caracteres omitidos...]\n\n${text.slice(-tail)}`;
}

/** Envoltura común: ejecuta, registra y convierte cualquier fallo en isError. */
async function himalayaTool(label, args, options) {
  console.error(`[MCP-HIMALAYA] ${label}: ${COMMAND} ${args.join(" ")}`);
  try {
    const { stdout, stderr } = await runHimalaya(args, options);
    const body = stdout.trim() || stderr.trim() || "(sin salida)";
    return { content: [{ type: "text", text: clip(body) }] };
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    return {
      isError: true,
      content: [{ type: "text", text: `No se pudo ejecutar ${label}: ${detail}` }],
    };
  }
}

const mailbox = z.string().min(1).optional().describe("Nombre o id de la bandeja; por defecto, la bandeja inbox de la cuenta activa");
const pageSize = z.number().int().positive().max(200).optional().describe("Máximo de mensajes a devolver (por defecto, el valor de configuración de himalaya)");

server.registerTool(
  "listar_cuentas",
  {
    title: "Listar cuentas",
    description: "Muestra las cuentas de himalaya definidas en la configuración y cuál está activa. Úsalo primero si una orden falla por falta de cuenta.",
    inputSchema: {
      json: z.boolean().optional().describe("Emitir JSON en lugar de tabla"),
    },
  },
  async ({ json }) => himalayaTool("listar_cuentas", ["account", "list", ...(json ? ["--json"] : [])])
);

server.registerTool(
  "listar_buzones",
  {
    title: "Listar bandejas",
    description: "Devuelve las bandejas de la cuenta activa con su id, nombre y, cuando el backend lo informa, totales y no leídos.",
    inputSchema: {
      json: z.boolean().optional().describe("Emitir JSON en lugar de tabla"),
    },
  },
  async ({ json }) => himalayaTool("listar_buzones", ["mailbox", "list", ...(json === false ? [] : ["--json"])])
);

server.registerTool(
  "listar_mensajes",
  {
    title: "Listar mensajes",
    description:
      "Lista los sobres (asunto, remitente, fecha, id) de una bandeja, del más reciente al más antiguo. El id que sale aquí es el que necesita leer_mensaje.",
    inputSchema: { mailbox, pageSize },
  },
  async ({ mailbox: name, pageSize: size }) =>
    himalayaTool("listar_mensajes", [
      "envelope",
      "list",
      "--json",
      ...(name ? ["-m", name] : []),
      ...(size ? ["-s", String(size)] : []),
    ])
);

server.registerTool(
  "buscar_mensajes",
  {
    title: "Buscar mensajes",
    description:
      'Busca sobres con la DSL de himalaya v2. Cada elemento de "terminos" es un token de la consulta: condiciones `date <fecha>`, `after <fecha>`, `from <patron>`, `to <patron>`, `subject <patron>`, `body <patron>`, `flag <seen|answered|flagged|draft>`, combinables con `and`/`or`/`not` y paréntesis, y ordenación `order by <campo> [asc|desc]`. Pasa cada condición como un elemento separado, sin comillas: ["from", "gmail.com", "subject", "factura"].',
    inputSchema: {
      terminos: z
        .array(z.string().min(1))
        .min(1)
        .describe("Tokens de la consulta de búsqueda, separados en elementos"),
      mailbox: mailbox,
      pageSize,
    },
  },
  async ({ terminos, mailbox: name, pageSize: size }) =>
    himalayaTool("buscar_mensajes", [
      "envelope",
      "search",
      "--json",
      ...(name ? ["-m", name] : []),
      ...(size ? ["-s", String(size)] : []),
      ...terminos,
    ])
);

server.registerTool(
  "leer_mensaje",
  {
    title: "Leer mensaje",
    description:      "Devuelve el mensaje completo parseado como JSON: cabeceras Date/From/To/Cc/Subject, flags y las partes MIME con su contenido.",
    inputSchema: {
      id: z.string().min(1).describe("Id del mensaje, tal cual aparece en listar_mensajes o buscar_mensajes"),
      mailbox,
      marcar_visto: z.boolean().optional().describe("Marcar el mensaje como leído (himalaya message read --seen)"),
    },
  },
  async ({ id, mailbox: name, marcar_visto }) =>
    himalayaTool("leer_mensaje", [
      "message",
      "read",
      "--json",
      ...(marcar_visto ? ["--seen"] : []),
      ...(name ? ["-m", name] : []),
      id,
    ])
);

server.registerTool(
  "enviar_correo",
  {
    title: "Enviar correo",
    description:
      "Envía un correo con la cuenta activa de himalaya. El cuerpo se entrega como mensaje RFC 5322 por entrada estándar, sin shell, así que el texto es literal. Manda solo lo que el usuario haya aprobado: enviar correo no es reversible.",
    inputSchema: {
      para: z.array(z.string().min(1)).min(1).describe("Direcciones de destinatario en To:"),
      asunto: z.string().describe("Asunto del correo"),
      cuerpo: z.string().describe("Cuerpo del correo en texto plano"),
      copia: z.array(z.string().min(1)).optional().describe("Direcciones adicionales en Cc:"),
      copia_oculta: z.array(z.string().min(1)).optional().describe("Direcciones adicionales en Bcc:"),
      remitente: z.string().optional().describe("Dirección de From:, si debe diferir de la cuenta activa"),
      guardar_en: z.string().optional().describe("Bandeja donde además se guarda una copia (himalaya message send --save)"),
    },
  },
  async ({ para, asunto, cuerpo, copia, copia_oculta, remitente, guardar_en }) => {
    const headers = [
      // Sin From:, himalaya usa la cuenta activa, que es lo habitual.
      ...(remitente ? [`From: ${remitente}`] : []),
      `To: ${para.join(", ")}`,
      ...(copia?.length ? [`Cc: ${copia.join(", ")}`] : []),
      ...(copia_oculta?.length ? [`Bcc: ${copia_oculta.join(", ")}`] : []),
      `Subject: ${asunto}`,
      "MIME-Version: 1.0",
      'Content-Type: text/plain; charset="utf-8"',
      "Content-Transfer-Encoding: 8bit",
    ];

    const raw = `${headers.join("\r\n")}\r\n\r\n${cuerpo}\r\n`;
    return himalayaTool(
      "enviar_correo",
      ["message", "send", ...(guardar_en ? ["--save", guardar_en] : [])],
      { input: raw }
    );
  }
);

async function run() {
  const transport = new StdioServerTransport();

  transport.onclose = () => {
    console.error("[MCP-HIMALAYA] Transporte cerrado por el cliente host.");
    process.exit(0);
  };

  transport.onerror = (error) => {
    console.error("[MCP-HIMALAYA] Error de transporte:", error);
  };

  await server.connect(transport);
  console.error("[MCP-HIMALAYA] Servidor iniciado sobre STDIN/STDOUT.");
}

run().catch((error) => {
  console.error("[MCP-HIMALAYA] Error fatal de inicialización:", error);
  process.exit(1);
});
