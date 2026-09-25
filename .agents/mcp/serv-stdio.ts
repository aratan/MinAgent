import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { readFile, readdir } from "node:fs/promises";
import { z } from "zod";

const server = new McpServer({
  name: "stdio-local-agent",
  version: "2.0.0",
});

/** One field out of the ``/proc/<pid>/status`` text. */
function statusField(status: string, key: string): string | null {
  const match = status.match(new RegExp(`^${key}:\\s*(.+)$`, "m"));
  return match ? match[1].trim() : null;
}

/** Read the real state of a process, so the model never gets invented numbers. */
async function inspectProcess(pid: number, metrica: string) {
  if (process.platform !== "linux") {
    throw new Error(`only implemented on Linux, running on ${process.platform}`);
  }
  const status = await readFile(`/proc/${pid}/status`, "utf8");
  const report: Record<string, unknown> = {
    target_pid: pid,
    modo: metrica,
    estado: statusField(status, "State"),
    hilos: Number(statusField(status, "Threads") ?? 0),
    timestamp_epoch: Date.now(),
  };
  if (metrica === "memoria" || metrica === "completo") {
    report.memoria_kb = {
      rss: Number(statusField(status, "VmRSS")?.replace(/\D/g, "") ?? 0),
      virtual: Number(statusField(status, "VmSize")?.replace(/\D/g, "") ?? 0),
      pico: Number(statusField(status, "VmPeak")?.replace(/\D/g, "") ?? 0),
    };
  }
  if (metrica === "fd" || metrica === "completo") {
    report.descriptores_abiertos = (await readdir(`/proc/${pid}/fd`)).length;
  }
  return report;
}

server.registerTool(
  "inspeccionar_proceso",
  {
    title: "Inspeccionar proceso",
    description: "Analiza el estado operacional de un proceso en la máquina local.",
    inputSchema: {
      pid: z.number().int().positive().describe("PID numérico del proceso target"),
      metrica: z.enum(["memoria", "fd", "completo"]).default("completo"),
    },
  },
  async ({ pid, metrica }) => {
    // Always log to stderr: stdout carries the JSON-RPC stream.
    console.error(`[MCP-STDIO] inspeccionar_proceso pid=${pid} metrica=${metrica}`);
    try {
      const report = await inspectProcess(pid, metrica ?? "completo");
      return { content: [{ type: "text", text: JSON.stringify(report, null, 2) }] };
    } catch (error) {
      const detail = error instanceof Error ? error.message : String(error);
      return {
        isError: true,
        content: [{ type: "text", text: `No se pudo inspeccionar el PID ${pid}: ${detail}` }],
      };
    }
  }
);

server.registerResource(
  "estado-sistema",
  "system://local/status",
  { title: "Estado del sistema", mimeType: "application/json", description: "Datos del host." },
  async (uri) => ({
    contents: [
      {
        uri: uri.href,
        mimeType: "application/json",
        text: JSON.stringify({
          hostname: process.env.HOSTNAME || "localhost",
          platform: process.platform,
          arch: process.arch,
          node_version: process.version,
        }),
      },
    ],
  })
);

async function run() {
  const transport = new StdioServerTransport();

  transport.onclose = () => {
    console.error("[MCP-STDIO] Transporte cerrado por el cliente host.");
    process.exit(0);
  };

  transport.onerror = (error) => {
    console.error("[MCP-STDIO] Error de transporte:", error);
  };

  await server.connect(transport);
  console.error("[MCP-STDIO] Servidor iniciado correctamente sobre STDIN/STDOUT.");
}

run().catch((error) => {
  console.error("[MCP-STDIO] Error fatal de inicialización:", error);
  process.exit(1);
});
