import fs from "fs/promises";
import path from "path";

import Fastify from "fastify";
import multipart from "@fastify/multipart";

const PYTHON_SERVICE_URL =
  process.env.PYTHON_SERVICE_URL || "http://python-service:8000";

// In production (Docker) __dirname is /app/dist, so ../frontend -> /app/frontend.
// Override with FRONTEND_DIR env var for local development.
const FRONTEND_DIR =
  process.env.FRONTEND_DIR || path.join(__dirname, "..", "frontend");

const PORT = parseInt(process.env.PORT || "3000", 10);

const fastify = Fastify({ logger: true });

fastify.register(multipart, {
  limits: {
    fileSize: 100 * 1024 * 1024, // 100 MB — accommodate long audio recordings
  },
});

// ── Health check ────────────────────────────────────────────────────────────

fastify.get("/health", async () => ({ status: "ok" }));

// ── Frontend ─────────────────────────────────────────────────────────────────

fastify.get("/test", async (_request, reply) => {
  const html = await fs.readFile(
    path.join(FRONTEND_DIR, "index.html"),
    "utf-8"
  );
  return reply.type("text/html").send(html);
});

// ── Proxy to Python analysis service ─────────────────────────────────────────

fastify.post("/api/analyze", async (request, reply) => {
  let text: string | undefined;
  let lang: string | undefined;
  let audioBuffer: Buffer | undefined;
  let audioFilename = "recording.wav";
  let audioMimetype = "audio/wav";

  // Iterate multipart parts and collect fields / file
  for await (const part of request.parts()) {
    if (part.type === "file" && part.fieldname === "audio") {
      audioBuffer = await part.toBuffer();
      audioFilename = part.filename || "recording.wav";
      audioMimetype = part.mimetype || "audio/wav";
    } else if (part.type === "field") {
      if (part.fieldname === "text") text = part.value as string;
      else if (part.fieldname === "lang") lang = part.value as string;
    }
  }

  if (!text) {
    return reply.status(400).send({ error: 'Missing "text" field' });
  }
  if (!audioBuffer) {
    return reply.status(400).send({ error: 'Missing "audio" file' });
  }

  // Rebuild FormData to forward to Python service
  const formData = new FormData();
  formData.append("text", text);
  if (lang) formData.append("lang", lang);
  formData.append(
    "audio",
    new Blob([audioBuffer], { type: audioMimetype }),
    audioFilename
  );

  let pythonResponse: Response;
  try {
    pythonResponse = await fetch(`${PYTHON_SERVICE_URL}/analyze`, {
      method: "POST",
      body: formData,
    });
  } catch (err) {
    fastify.log.error(err, "Failed to reach Python service");
    return reply.status(503).send({ error: "Analysis service unavailable" });
  }

  const result = await pythonResponse.json();
  return reply.status(pythonResponse.status).send(result);
});

// ── Start ────────────────────────────────────────────────────────────────────

fastify.listen({ port: PORT, host: "0.0.0.0" }, (err) => {
  if (err) {
    fastify.log.error(err);
    process.exit(1);
  }
});
