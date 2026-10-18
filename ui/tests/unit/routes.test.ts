// Next sunucu rotaları (Faz 3 tasarımı §4.4; plan Görev 28): geçici DATA_DIR ve gerçek bir sahte
// motor (http.createServer). Rotalar doğrudan Request nesneleriyle çağrılır.
import { createHash } from "node:crypto";
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { createServer, type Server } from "node:http";
import { type AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, beforeEach, describe, expect, it } from "vitest";

import { POST as detect } from "@/app/api/detect/route";
import { GET as frame } from "@/app/api/frame/[seq]/[frame]/[mod]/route";
import { GET as sequences } from "@/app/api/sequences/route";
import { POST as reset } from "@/app/api/session/reset/route";
import { listSequences } from "@/lib/data";

type Received = { url: string; parts: Record<string, Buffer> };

let dataDir: string;
let engine: Server;
let received: Received[] = [];
let engineReply: { status: number; body: unknown } = { status: 200, body: { detections: [] } };

// Yalnız başlığı geçerli bir PNG (IHDR genişlik ve yükseklik): boyut denetimi bunu okur.
function png(width: number, height: number, pad = 0): Buffer {
  const b = Buffer.alloc(33 + pad);
  Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]).copy(b, 0);
  b.writeUInt32BE(13, 8);
  b.write("IHDR", 12, "ascii");
  b.writeUInt32BE(width, 16);
  b.writeUInt32BE(height, 20);
  return b;
}

const sha = (b: Buffer) => createHash("sha256").update(b).digest("hex");

function writeSeq(name: string, split: string, frames: number[]) {
  const d = join(dataDir, "sequences", name);
  mkdirSync(join(d, "rgb"), { recursive: true });
  mkdirSync(join(d, "ir"), { recursive: true });
  const index = { name, split, frames, fps: 30, size: [640, 512], source: "test" };
  writeFileSync(join(d, "index.json"), JSON.stringify(index));
  for (const f of frames) {
    const id = String(f).padStart(6, "0");
    writeFileSync(join(d, "rgb", `${id}.jpg`), Buffer.from(`jpeg-${name}-${f}`));
    writeFileSync(join(d, "ir", `${id}.png`), png(640, 512, f));
  }
}

type FrameParams = { seq: string; frame: string; mod: string };
const params = (p: FrameParams) => ({ params: Promise.resolve(p) });
const blob = (b: Buffer, type: string) => new Blob([new Uint8Array(b)], { type });

function jsonRequest(url: string, body: unknown) {
  return new Request(`http://ui${url}`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
  });
}

beforeAll(async () => {
  dataDir = mkdtempSync(join(tmpdir(), "baykus-ui-"));
  writeSeq("1_08", "val", [1, 2, 3]);
  writeSeq("2-01", "test", [1]);
  process.env.DATA_DIR = dataDir;
  engine = createServer(async (req, res) => {
    const chunks: Buffer[] = [];
    for await (const c of req) chunks.push(c as Buffer);
    const body = Buffer.concat(chunks);
    const parts: Record<string, Buffer> = {};
    if ((req.headers["content-type"] ?? "").startsWith("multipart/form-data")) {
      const form = await new Response(body, {
        headers: { "content-type": req.headers["content-type"]! },
      }).formData();
      for (const [k, v] of form.entries()) {
        if (typeof v !== "string") parts[k] = Buffer.from(await v.arrayBuffer());
      }
    }
    received.push({ url: req.url ?? "", parts });
    res.writeHead(engineReply.status, { "content-type": "application/json" });
    res.end(JSON.stringify(engineReply.body));
  });
  await new Promise<void>((r) => engine.listen(0, "127.0.0.1", r));
  process.env.ENGINE_URL = `http://127.0.0.1:${(engine.address() as AddressInfo).port}`;
});

afterAll(() => engine.close());

beforeEach(() => {
  received = [];
  engineReply = { status: 200, body: { detections: [] } };
});

describe("data routes", () => {
  it("rejects path traversal", async () => {
    for (const p of [
      { seq: "..", frame: "1", mod: "rgb" },
      { seq: "../1_08", frame: "1", mod: "rgb" },
      { seq: "1_08", frame: "1", mod: "../../index.json" },
      { seq: "1_08", frame: "1.5", mod: "rgb" },
      { seq: "1_08", frame: "-1", mod: "ir" },
    ]) {
      const r = await frame(new Request("http://ui/api/frame"), params(p));
      expect(r.status, JSON.stringify(p)).toBe(400);
    }
  });

  it("404 for frame out of range", async () => {
    const r = await frame(new Request("http://ui/"), params({ seq: "1_08", frame: "99999", mod: "rgb" }));
    expect(r.status).toBe(404);
    const missing = await frame(new Request("http://ui/"), params({ seq: "yok", frame: "1", mod: "rgb" }));
    expect(missing.status).toBe(404);
  });

  it("serves original frame bytes", async () => {
    const r = await frame(new Request("http://ui/"), params({ seq: "1_08", frame: "2", mod: "rgb" }));
    expect(r.status).toBe(200);
    expect(r.headers.get("content-type")).toBe("image/jpeg");
    expect(Buffer.from(await r.arrayBuffer()).toString()).toBe("jpeg-1_08-2");
    const ir = await frame(new Request("http://ui/"), params({ seq: "1_08", frame: "2", mod: "ir" }));
    expect(ir.headers.get("content-type")).toBe("image/png");
  });

  it("hides test split sequences", async () => {
    expect((await listSequences()).map((s) => s.name)).toEqual(["1_08"]);
    const r = await sequences();
    expect((await r.json()).map((s: { name: string }) => s.name)).toEqual(["1_08"]);
    const f = await frame(new Request("http://ui/"), params({ seq: "2-01", frame: "1", mod: "rgb" }));
    expect(f.status).toBe(404);
  });
});

describe("detect route", () => {
  const meta = { seq: "1_08", frame: 2, conf: 0.3, track: true, gate: "auto", session: "s-1" };

  it("json body forwards original bytes", async () => {
    engineReply = { status: 200, body: { detections: [{ cls: "car" }], session_new: true } };
    const r = await detect(jsonRequest("/api/detect", meta));
    expect(r.status).toBe(200);
    expect((await r.json()).session_new).toBe(true);
    expect(received).toHaveLength(1);
    const { url, parts } = received[0];
    const dir = join(dataDir, "sequences", "1_08");
    expect(sha(parts.rgb)).toBe(sha(readFileSync(join(dir, "rgb", "000002.jpg"))));
    expect(sha(parts.ir)).toBe(sha(readFileSync(join(dir, "ir", "000002.png"))));
    const q = new URL(url, "http://e").searchParams;
    expect(q.get("conf")).toBe("0.3");
    expect(q.get("track")).toBe("true");
    expect(q.get("gate")).toBe("auto");
    expect(q.get("session")).toBe("s-1");
  });

  it("multipart fills missing channel from disk", async () => {
    const ir = png(640, 512, 77);
    const form = new FormData();
    form.set("meta", JSON.stringify({ ...meta, fault: { ir_noise: 20 } }));
    form.set("ir", blob(ir, "image/png"), "ir.png");
    const r = await detect(new Request("http://ui/api/detect", { method: "POST", body: form }));
    expect(r.status).toBe(200);
    const { parts } = received[0];
    expect(sha(parts.ir)).toBe(sha(ir));
    expect(parts.rgb.toString()).toBe("jpeg-1_08-2");
  });

  it("400 for wrong size or type", async () => {
    for (const [part, name] of [
      [blob(png(320, 256), "image/png"), "size"],
      [blob(Buffer.from("not a png"), "image/png"), "signature"],
      [blob(png(640, 512), "image/jpeg"), "type"],
    ] as const) {
      const form = new FormData();
      form.set("meta", JSON.stringify(meta));
      form.set("rgb", part, "rgb.png");
      const r = await detect(new Request("http://ui/api/detect", { method: "POST", body: form }));
      expect(r.status, name).toBe(400);
    }
    expect(received).toHaveLength(0);
  });

  it("400 for bad json body", async () => {
    expect((await detect(jsonRequest("/api/detect", { seq: "1_08" }))).status).toBe(400);
    expect((await detect(jsonRequest("/api/detect", { ...meta, seq: "../x" }))).status).toBe(400);
    const bad = new Request("http://ui/api/detect", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: "{",
    });
    expect((await detect(bad)).status).toBe(400);
  });

  it("413 over 10 MB", async () => {
    const form = new FormData();
    form.set("meta", JSON.stringify(meta));
    form.set("ir", blob(png(640, 512, 10 * 1024 * 1024), "image/png"), "ir.png");
    const r = await detect(new Request("http://ui/api/detect", { method: "POST", body: form }));
    expect(r.status).toBe(413);
    expect(received).toHaveLength(0);
  });

  it("passes engine errors through", async () => {
    engineReply = { status: 400, body: { error: "bilinmeyen gate modu" } };
    const r = await detect(jsonRequest("/api/detect", meta));
    expect(r.status).toBe(400);
    expect((await r.json()).error).toBe("bilinmeyen gate modu");
  });

  it("502 when engine down", async () => {
    const saved = process.env.ENGINE_URL;
    process.env.ENGINE_URL = "http://127.0.0.1:1";
    try {
      const r = await detect(jsonRequest("/api/detect", meta));
      expect(r.status).toBe(502);
      expect((await r.json()).error).toBe("motor yanıt vermiyor");
    } finally {
      process.env.ENGINE_URL = saved;
    }
  });
});

describe("session reset route", () => {
  it("session reset forwards", async () => {
    engineReply = { status: 200, body: { session: "s-9", reset: true } };
    const r = await reset(jsonRequest("/api/session/reset", { session: "s-9" }));
    expect(r.status).toBe(200);
    expect(received[0].url).toBe("/v1/session/reset?session=s-9");
  });

  it("400 without session", async () => {
    expect((await reset(jsonRequest("/api/session/reset", {}))).status).toBe(400);
    expect(received).toHaveLength(0);
  });
});
