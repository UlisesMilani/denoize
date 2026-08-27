import { createReadStream } from "node:fs";
import { stat } from "node:fs/promises";
import http from "node:http";
import path from "node:path";
import { fileURLToPath } from "node:url";

const testDirectory = path.dirname(fileURLToPath(import.meta.url));
const sdkRoot = path.resolve(testDirectory, "../..");
const generatedWasmRoot = process.env.DENOIZE_WASM_BROWSER_PACKAGE_DIR
  ? path.resolve(process.env.DENOIZE_WASM_BROWSER_PACKAGE_DIR)
  : null;
const host = "127.0.0.1";
const port = 4173;
const mimeTypes = new Map([
  [".html", "text/html; charset=utf-8"],
  [".js", "text/javascript; charset=utf-8"],
  [".mjs", "text/javascript; charset=utf-8"],
  [".wasm", "application/wasm"],
]);

function headers(contentType = "text/plain; charset=utf-8") {
  return {
    "Cache-Control": "no-store",
    "Content-Type": contentType,
    "Cross-Origin-Embedder-Policy": "require-corp",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
  };
}

const server = http.createServer(async (request, response) => {
  try {
    if (request.method !== "GET" && request.method !== "HEAD") {
      response.writeHead(405, headers());
      response.end("method not allowed\n");
      return;
    }
    const url = new URL(request.url ?? "/", `http://${host}:${port}`);
    if (url.pathname === "/health") {
      response.writeHead(200, headers());
      response.end("ok\n");
      return;
    }
    const pathname =
      url.pathname === "/" ? "/web/test/browser-fixture.html" : url.pathname;
    const decoded = decodeURIComponent(pathname);
    const generatedPrefix = "/denoize-wasm/pkg/";
    const root = generatedWasmRoot !== null && decoded.startsWith(generatedPrefix)
      ? generatedWasmRoot
      : sdkRoot;
    const relative = root === generatedWasmRoot
      ? decoded.slice(generatedPrefix.length)
      : `.${decoded}`;
    const candidate = path.resolve(root, relative);
    if (candidate !== root && !candidate.startsWith(`${root}${path.sep}`)) {
      response.writeHead(403, headers());
      response.end("forbidden\n");
      return;
    }
    const metadata = await stat(candidate);
    if (!metadata.isFile()) {
      throw new Error("not a regular file");
    }
    response.writeHead(200, {
      ...headers(mimeTypes.get(path.extname(candidate)) ?? "application/octet-stream"),
      "Content-Length": metadata.size,
    });
    if (request.method === "HEAD") {
      response.end();
      return;
    }
    createReadStream(candidate).pipe(response);
  } catch {
    response.writeHead(404, headers());
    response.end("not found\n");
  }
});

server.listen(port, host);

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.once(signal, () => server.close(() => process.exit(0)));
}
