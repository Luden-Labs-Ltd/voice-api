// Load .env before reading any process.env values.
// In Docker / Railway the variables are already injected, so this is a no-op.
import "dotenv/config";

import fs from "fs/promises";
import { createReadStream } from "fs";
import path from "path";
import os from "os";
import { randomUUID } from "crypto";

import Fastify from "fastify";
import cors from "@fastify/cors";
import multipart from "@fastify/multipart";
import OpenAI from "openai";

// Railway injects PORT; NODE_PORT is the local-dev fallback from .env.
const PORT = parseInt(
  process.env.PORT || process.env.NODE_PORT || "3000",
  10
);

const PYTHON_SERVICE_URL =
  process.env.PYTHON_SERVICE_URL || "http://python-service:8000";

// In production (Docker) __dirname is /app/dist, so ../frontend -> /app/frontend.
const FRONTEND_DIR =
  process.env.FRONTEND_DIR || path.join(__dirname, "..", "frontend");

const LOG_LEVEL = process.env.LOG_LEVEL || "info";

const fastify = Fastify({ logger: { level: LOG_LEVEL } });

fastify.register(cors, { origin: true });

fastify.register(multipart, {
  limits: {
    fileSize: 100 * 1024 * 1024, // 100 MB
  },
});

// ── OpenAI transcription ──────────────────────────────────────────────────────

const OPENAI_API_KEY = process.env.OPENAI_API_KEY;
// Single client instance; null when key is absent (fallback to Python ASR).
const openaiClient = OPENAI_API_KEY
  ? new OpenAI({ apiKey: OPENAI_API_KEY })
  : null;

// Map our lang codes to ISO 639-1 codes accepted by OpenAI.
const LANG_ISO: Record<string, string> = { he: "he", en: "en", ru: "ru" };

const OPENAI_AUDIO_EXTS = new Set([
  ".mp3",
  ".mp4",
  ".mpeg",
  ".mpga",
  ".m4a",
  ".wav",
  ".webm",
]);

/** Расширение tempfile для OpenAI: mimetype надёжнее filename (фронт раньше слал webm как .wav). */
function extensionForOpenAIUpload(
  filename: string,
  mime?: string
): string {
  const m = (mime || "").toLowerCase();
  if (m.includes("webm")) return ".webm";
  if (m.includes("wav")) return ".wav";
  if (m.includes("mp3") || m.includes("mpeg")) return ".mp3";
  if (m.includes("mp4")) return ".mp4";
  if (m.includes("m4a")) return ".m4a";
  if (m.includes("mpga")) return ".mpga";

  const ext = path.extname(filename).toLowerCase();
  if (OPENAI_AUDIO_EXTS.has(ext)) return ext;

  return ".webm";
}

if (!openaiClient) {
  fastify.log.info("OPENAI_API_KEY not set — will use Python Whisper ASR for all requests");
}

// gpt-4o-transcribe returns plain text only (json format).
// Word timestamps are not available; Python derives timing from audio duration.
async function transcribeWithOpenAI(
  buffer: Buffer,
  filename: string,
  mime: string | undefined,
  lang: string | undefined
): Promise<string | null> {
  if (!openaiClient) return null;

  const ext = extensionForOpenAIUpload(filename, mime);
  const tempPath = path.join(os.tmpdir(), `rs-${randomUUID()}${ext}`);

  try {
    await fs.writeFile(tempPath, buffer);

    const iso = lang ? LANG_ISO[lang] : undefined;

    const transcription = await openaiClient.audio.transcriptions.create({
      file: createReadStream(tempPath),
      model: "gpt-4o-transcribe",
      response_format: "json",
      ...(iso ? { language: iso } : {}),
    } as any);

    const text: string = (transcription as any).text || "";

    fastify.log.info(
      { asr: "openai", lang, transcript: text },
      "OpenAI transcription succeeded"
    );

    return text;
  } finally {
    await fs.unlink(tempPath).catch(() => {});
  }
}

// ── Health check ──────────────────────────────────────────────────────────────

fastify.get("/health", async () => ({ status: "ok" }));

// ── Frontend ──────────────────────────────────────────────────────────────────

fastify.get("/test", async (_request, reply) => {
  const html = await fs.readFile(
    path.join(FRONTEND_DIR, "index.html"),
    "utf-8"
  );
  return reply.type("text/html").send(html);
});

// ── Main analysis route ───────────────────────────────────────────────────────
//
// New flow:
//   1. Collect audio + text + lang from frontend multipart request
//   2. Transcribe audio via OpenAI gpt-4o-transcribe (if key is configured)
//      Falls back to Python Whisper if OpenAI is unavailable or fails.
//   3. Forward audio + transcript to Python for scoring (alignment, prosody, etc.)
//
// The API key is never exposed to the frontend — only the Node service uses it.

fastify.post("/api/analyze", async (request, reply) => {
  let text: string | undefined;
  let lang: string | undefined;
  let audioBuffer: Buffer | undefined;
  let audioFilename = "recording.wav";
  let audioMimetype = "audio/wav";

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

  // ── Step 1: OpenAI transcription (primary) ──────────────────────────────────
  // On success: transcript text forwarded to Python, which skips local Whisper.
  // On failure / no key: Python runs local Whisper ASR instead.
  let openaiTranscript: string | null = null;
  try {
    openaiTranscript = await transcribeWithOpenAI(
      audioBuffer,
      audioFilename,
      audioMimetype,
      lang
    );
  } catch (err) {
    fastify.log.warn({ asr: "openai", err }, "OpenAI transcription failed — falling back to Python Whisper ASR");
  }

  if (openaiTranscript === null && openaiClient) {
    fastify.log.info({ asr: "whisper" }, "Falling back to Python Whisper ASR");
  } else if (openaiTranscript === null) {
    fastify.log.info({ asr: "whisper" }, "Using Python Whisper ASR (no OpenAI key configured)");
  }

  // ── Step 2: Forward to Python for scoring ───────────────────────────────────
  // Audio is always forwarded (needed for prosody analysis regardless of ASR source).
  // If OpenAI transcript is available, Python uses it directly and skips Whisper.
  // Word timestamps are not provided (json format); Python uses audio duration for timing.
  const formData = new FormData();
  formData.append("text", text);
  if (lang) formData.append("lang", lang);
  formData.append(
    "audio",
    new Blob([audioBuffer], { type: audioMimetype }),
    audioFilename
  );

  if (openaiTranscript !== null) {
    formData.append("transcript", openaiTranscript);
    // transcript_words omitted: no word timestamps from json format.
    // Python will derive timing from audio duration via _get_audio_duration().
  }

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

// ── Start ─────────────────────────────────────────────────────────────────────

fastify.listen({ port: PORT, host: "0.0.0.0" }, (err) => {
  if (err) {
    fastify.log.error(err);
    process.exit(1);
  }
});
