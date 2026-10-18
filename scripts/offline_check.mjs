// Kapalı ağ kanıtı 1 (Faz 3 tasarımı §4.5; plan Görev 33): kapalı katmanda bir kare /api/detect
// üzerinden uçtan uca işlenir ve motorun /v1/model sha256'ları models.json ile karşılaştırılır.
// Kullanım (ui imajında): docker compose exec ui node /app/scripts/offline_check.mjs
// Ortam: UI_URL (varsayılan http://127.0.0.1:3000), ENGINE_URL (http://engine:8080),
//        MANIFEST (/app/models.json), WAIT_S (60). Çıkış 0 = geçti; JSON stdout.
import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

const SHA_FILES = {
  model: "e3c.onnx",
  thresholds: "thresholds.json",
  calibration: "calibration.json",
  gate_model: "gate_model.json",
};

export function compareShas(model, manifest) {
  const mismatched = Object.entries(SHA_FILES)
    .filter(([key, file]) => model?.sha256?.[key] !== manifest.files?.[file]?.sha256)
    .map(([, file]) => file);
  return { ok: mismatched.length === 0, mismatched };
}

async function waitFor(url, seconds) {
  const until = Date.now() + seconds * 1000;
  for (;;) {
    try {
      const res = await fetch(url);
      if (res.ok) return res;
    } catch {
      // henüz açılmadı
    }
    if (Date.now() > until) throw new Error(`${url} ${seconds} sn içinde yanıt vermedi`);
    await new Promise((r) => setTimeout(r, 1000));
  }
}

async function main() {
  const ui = process.env.UI_URL ?? "http://127.0.0.1:3000";
  const engine = process.env.ENGINE_URL ?? "http://engine:8080";
  const manifest = JSON.parse(readFileSync(process.env.MANIFEST ?? "/app/models.json", "utf-8"));
  const wait = Number(process.env.WAIT_S ?? 60);
  const out = { ok: false };
  try {
    await waitFor(`${engine}/health`, wait);
    const seqs = await (await waitFor(`${ui}/api/sequences`, wait)).json();
    if (!Array.isArray(seqs) || seqs.length === 0) throw new Error("dizi yok");
    const seq = seqs[0].name;
    const frame = seqs[0].frames[0];
    const res = await fetch(`${ui}/api/detect`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        seq,
        frame,
        conf: 0.25,
        track: true,
        gate: "auto",
        session: "offline-check",
      }),
    });
    const body = await res.json();
    Object.assign(out, {
      sequence: seq,
      frame,
      detect_status: res.status,
      detections: Array.isArray(body.detections) ? body.detections.length : null,
    });
    const model = await (await fetch(`${engine}/v1/model`)).json();
    const shas = compareShas(model, manifest);
    Object.assign(out, { model_sha_ok: shas.ok, mismatched: shas.mismatched });
    out.ok = res.status === 200 && Array.isArray(body.detections) && shas.ok;
  } catch (e) {
    out.error = e instanceof Error ? e.message : String(e);
  }
  process.stdout.write(`${JSON.stringify(out)}\n`);
  process.exit(out.ok ? 0 : 1);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) await main();
