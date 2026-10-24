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
import { GET as modelRoute } from "@/app/api/model/route";
import { GET as sequences } from "@/app/api/sequences/route";
import { POST as ack } from "@/app/api/session/alarms/ack/route";
import { GET as events } from "@/app/api/session/events/route";
import { POST as reset } from "@/app/api/session/reset/route";
import { GET as trackRoute } from "@/app/api/session/track/route";
import { GET as getZones, PUT as putZones } from "@/app/api/session/zones/route";
import { listSequences } from "@/lib/data";

type Received = { url: string; method: string; body: string; parts: Record<string, Buffer> };

let dataDir: string;
let engine: Server;
let received: Received[] = [];
// `text` verilirse gövde JSON yerine olduğu gibi (contentType ile) gönderilir.
let engineReply: { status: number; body?: unknown; contentType?: string; text?: string } = {
  status: 200,
  body: { detections: [] },
};

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
    received.push({
      url: req.url ?? "",
      method: req.method ?? "",
      body: (req.headers["content-type"] ?? "").startsWith("multipart/form-data")
        ? ""
        : body.toString("utf-8"),
      parts,
    });
    if (engineReply.text !== undefined) {
      res.writeHead(engineReply.status, {
        "content-type": engineReply.contentType ?? "text/plain; charset=utf-8",
      });
      res.end(engineReply.text);
      return;
    }
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

  it("detect forwards frame focus and tiles", async () => {
    const r = await detect(jsonRequest("/api/detect", { ...meta, focus: 7, tiles: 4 }));
    expect(r.status).toBe(200);
    const q = new URL(received[0].url, "http://e").searchParams;
    expect(q.get("frame")).toBe("2");
    expect(q.get("focus")).toBe("7");
    expect(q.get("tiles")).toBe("4");

    received = [];
    expect((await detect(jsonRequest("/api/detect", meta))).status).toBe(200);
    const plain = new URL(received[0].url, "http://e").searchParams;
    expect(plain.get("frame")).toBe("2");
    expect(plain.has("focus")).toBe(false);
    expect(plain.has("tiles")).toBe(false);

    received = [];
    expect((await detect(jsonRequest("/api/detect", { ...meta, tiles: 3 }))).status).toBe(400);
    expect(received).toHaveLength(0);
  });

  it("freeze reads the frozen frame bytes from disk", async () => {
    const dir = join(dataDir, "sequences", "1_08");
    const r = await detect(jsonRequest("/api/detect", { ...meta, freeze: { ir: 1 } }));
    expect(r.status).toBe(200);
    const { parts } = received[0];
    expect(sha(parts.ir)).toBe(sha(readFileSync(join(dir, "ir", "000001.png"))));
    expect(parts.rgb.toString()).toBe("jpeg-1_08-2");

    received = [];
    expect((await detect(jsonRequest("/api/detect", { ...meta, freeze: { rgb: 3 } }))).status).toBe(200);
    expect(received[0].parts.rgb.toString()).toBe("jpeg-1_08-3");
  });

  it("freeze rejects frame outside sequence (400)", async () => {
    expect((await detect(jsonRequest("/api/detect", { ...meta, freeze: { rgb: 99 } }))).status).toBe(400);
    expect((await detect(jsonRequest("/api/detect", { ...meta, freeze: { ir: "x" } }))).status).toBe(400);
    expect(received).toHaveLength(0);
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

  it("reset route rejects unknown scope", async () => {
    expect((await reset(jsonRequest("/api/session/reset", { session: "s-9", scope: "x" }))).status).toBe(400);
    expect(received).toHaveLength(0);

    expect((await reset(jsonRequest("/api/session/reset", { session: "s-9", scope: "tracks" }))).status).toBe(200);
    expect(received[0].url).toBe("/v1/session/reset?session=s-9&scope=tracks");

    received = [];
    expect((await reset(jsonRequest("/api/session/reset", { session: "s-9", scope: "all" }))).status).toBe(200);
    expect(received[0].url).toBe("/v1/session/reset?session=s-9&scope=all");
  });
});

describe("zones route", () => {
  const zones = [{ id: "z1", points: [[0, 0], [10, 0], [10, 10]] }];

  it("zones route forwards engine 400 message", async () => {
    engineReply = { status: 400, body: { error: "çokgen kendini kesiyor" } };
    const r = await putZones(jsonRequest("/api/session/zones?session=s-1", { zones }));
    expect(r.status).toBe(400);
    expect((await r.json()).error).toBe("çokgen kendini kesiyor");
    expect(received).toHaveLength(1);
    expect(received[0].method).toBe("PUT");
    expect(received[0].url).toBe("/v1/session/zones?session=s-1");
    expect(received[0].body).toBe(JSON.stringify({ zones }));
  });

  it("zones get forwards", async () => {
    engineReply = { status: 200, body: { zones } };
    const r = await getZones(new Request("http://ui/api/session/zones?session=s-1"));
    expect(r.status).toBe(200);
    expect((await r.json()).zones).toEqual(zones);
    expect(received[0].method).toBe("GET");
    expect(received[0].url).toBe("/v1/session/zones?session=s-1");
  });

  it("zones route needs session", async () => {
    expect((await getZones(new Request("http://ui/api/session/zones"))).status).toBe(400);
    expect((await putZones(jsonRequest("/api/session/zones", { zones }))).status).toBe(400);
    expect(received).toHaveLength(0);
  });

  it("zones body over 64 KB is 413", async () => {
    const big = { zones: [{ id: "z1", points: [[0, 0]], pad: "x".repeat(64 * 1024) }] };
    const r = await putZones(jsonRequest("/api/session/zones?session=s-1", big));
    expect(r.status).toBe(413);
    expect(received).toHaveLength(0);
  });
});

describe("events route", () => {
  it("events route passes csv content type", async () => {
    engineReply = { status: 200, text: "seq,n\n1,2\n", contentType: "text/csv; charset=utf-8" };
    const r = await events(new Request("http://ui/api/session/events?session=s-1&since=5&format=csv"));
    expect(r.status).toBe(200);
    expect(r.headers.get("content-type")).toBe("text/csv; charset=utf-8");
    expect(await r.text()).toBe("seq,n\n1,2\n");
    expect(received[0].url).toBe("/v1/session/events?session=s-1&since=5&format=csv");
  });

  it("events route validates since and format", async () => {
    for (const q of ["session=s-1&since=-1", "session=s-1&since=a", "session=s-1&format=xml"]) {
      const r = await events(new Request(`http://ui/api/session/events?${q}`));
      expect(r.status, q).toBe(400);
    }
    expect(received).toHaveLength(0);

    expect((await events(new Request("http://ui/api/session/events?session=s-1"))).status).toBe(200);
    expect(received[0].url).toBe("/v1/session/events?session=s-1&since=0&format=json");
  });
});

describe("alarm ack route", () => {
  it("ack forwards alarm id", async () => {
    const r = await ack(jsonRequest("/api/session/alarms/ack", { session: "s-1", alarm: "a3" }));
    expect(r.status).toBe(200);
    expect(received[0].method).toBe("POST");
    expect(received[0].url).toBe("/v1/session/alarms/ack?session=s-1&alarm=a3");
  });

  it("ack passes 404 through", async () => {
    engineReply = { status: 404, body: { error: "alarm yok: a9" } };
    const r = await ack(jsonRequest("/api/session/alarms/ack", { session: "s-1", alarm: "a9" }));
    expect(r.status).toBe(404);
    expect((await r.json()).error).toBe("alarm yok: a9");
  });

  it("ack needs session and alarm", async () => {
    expect((await ack(jsonRequest("/api/session/alarms/ack", { session: "s-1" }))).status).toBe(400);
    expect((await ack(jsonRequest("/api/session/alarms/ack", { alarm: "a3" }))).status).toBe(400);
    expect(received).toHaveLength(0);
  });
});

describe("track route", () => {
  it("track forwards id", async () => {
    const r = await trackRoute(new Request("http://ui/api/session/track?session=s-1&id=3"));
    expect(r.status).toBe(200);
    expect(received[0].method).toBe("GET");
    expect(received[0].url).toBe("/v1/session/track?session=s-1&id=3");
  });

  it("track rejects bad id", async () => {
    expect((await trackRoute(new Request("http://ui/api/session/track?session=s-1&id=x"))).status).toBe(400);
    expect((await trackRoute(new Request("http://ui/api/session/track?session=s-1"))).status).toBe(400);
    expect(received).toHaveLength(0);
  });
});

describe("model route", () => {
  it("model forwards", async () => {
    engineReply = { status: 200, body: { model: "yolo26n", classes: ["car"] } };
    const r = await modelRoute();
    expect(r.status).toBe(200);
    expect((await r.json()).model).toBe("yolo26n");
    expect(received[0].method).toBe("GET");
    expect(received[0].url).toBe("/v1/model");
  });
});
