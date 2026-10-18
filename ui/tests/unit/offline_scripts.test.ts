// Kapalı ağ kanıt betikleri (Faz 3 tasarımı §4.5; plan Görev 33): scripts/{egress_check,
// scan_next,offline_check}.mjs. Ağ sonucu yerel soketlerle sınanır (dış adres yok).
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { createServer, type Server } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { afterAll, beforeAll, describe, expect, it } from "vitest";

import { probeTcp, verdict, EXPECTED } from "../../../scripts/egress_check.mjs";
import { loadAllow, scan } from "../../../scripts/scan_next.mjs";
import { compareShas } from "../../../scripts/offline_check.mjs";
import { assemble, readEvidence } from "../../../scripts/offline_report.mjs";

describe("egress_check", () => {
  let server: Server;
  let port = 0;
  beforeAll(async () => {
    server = createServer((s) => s.end());
    await new Promise<void>((r) => server.listen(0, "127.0.0.1", r));
    port = (server.address() as { port: number }).port;
  });
  afterAll(() => server.close());

  it("expected network errors are the design list", () => {
    expect([...EXPECTED].sort()).toEqual(
      ["EAI_AGAIN", "ECONNREFUSED", "EHOSTUNREACH", "ENETUNREACH", "ENOTFOUND", "ETIMEDOUT"].sort(),
    );
  });

  it("a connection that succeeds fails the check", async () => {
    const tcp = await probeTcp("127.0.0.1", port, 3000);
    expect(tcp).toMatchObject({ connected: true, expected: false });
    const dns = { target: "example.com", result: "EAI_AGAIN", connected: false, expected: true };
    expect(verdict(tcp, dns).blocked).toBe(false);
  });

  it("refused connection counts as blocked", async () => {
    const closed = createServer();
    await new Promise<void>((r) => closed.listen(0, "127.0.0.1", r));
    const p = (closed.address() as { port: number }).port;
    await new Promise<void>((r) => closed.close(() => r()));
    const tcp = await probeTcp("127.0.0.1", p, 3000);
    expect(tcp).toMatchObject({ connected: false, expected: true, result: "ECONNREFUSED" });
    const dns = { target: "example.com", result: "ENOTFOUND", connected: false, expected: true };
    expect(verdict(tcp, dns)).toMatchObject({ blocked: true });
  });

  it("an unexpected error cannot prove the block", () => {
    const tcp = { target: "1.1.1.1:443", result: "EACCES", connected: false, expected: false };
    const dns = { target: "example.com", result: "EAI_AGAIN", connected: false, expected: true };
    expect(verdict(tcp, dns)).toMatchObject({
      blocked: false,
      message: "ağ engeli kanıtlanamadı",
    });
  });
});

describe("scan_next", () => {
  it("reports urls outside the allow list", () => {
    const dir = mkdtempSync(join(tmpdir(), "scan-"));
    mkdirSync(join(dir, "static", "chunks"), { recursive: true });
    mkdirSync(join(dir, "cache"), { recursive: true });
    writeFileSync(
      join(dir, "static", "chunks", "a.js"),
      'x="http://www.w3.org/2000/svg";y="https://react.dev/errors/418";z="https://evil.example/t"',
    );
    writeFileSync(join(dir, "cache", "b.js"), 'q="https://ignored.example/"');
    const allowFile = join(dir, "allow.txt");
    writeFileSync(allowFile, "# yorum\nhttp://www.w3.org/\nhttps://react.dev/errors/\n");
    const allow = loadAllow(allowFile);
    expect(allow).toEqual(["http://www.w3.org/", "https://react.dev/errors/"]);
    const r = scan(dir, allow);
    expect(r.violations).toEqual([{ file: "static/chunks/a.js", url: "https://evil.example/t" }]);
  });

  it("'=' entries match exactly, not as a prefix", () => {
    const dir = mkdtempSync(join(tmpdir(), "scan-"));
    writeFileSync(join(dir, "a.js"), 'u="https://a";v="https://ab.example/x";w="https://тест"');
    const r = scan(dir, ["=https://a", "=https://тест"]);
    expect(r.violations).toEqual([{ file: "a.js", url: "https://ab.example/x" }]);
  });
});

describe("offline_check", () => {
  const manifest = {
    files: {
      "e3c.onnx": { sha256: "a".repeat(64) },
      "thresholds.json": { sha256: "b".repeat(64) },
      "calibration.json": { sha256: "c".repeat(64) },
      "gate_model.json": { sha256: "d".repeat(64) },
    },
  };
  const model = {
    sha256: {
      model: "a".repeat(64),
      thresholds: "b".repeat(64),
      calibration: "c".repeat(64),
      gate_model: "d".repeat(64),
    },
  };

  it("model shas match the manifest", () => {
    expect(compareShas(model, manifest)).toEqual({ ok: true, mismatched: [] });
  });

  it("a different sha is reported", () => {
    const bad = { sha256: { ...model.sha256, calibration: "0".repeat(64) } };
    expect(compareShas(bad, manifest)).toEqual({ ok: false, mismatched: ["calibration.json"] });
  });
});

describe("offline_report", () => {
  const blockedProbe = { target: "x", result: "ENETUNREACH", connected: false, expected: true };
  const egress = { tcp: blockedProbe, dns: blockedProbe, blocked: true, message: "dış ağ kapalı" };
  const parts = {
    offline_check: { ok: true, detect_status: 200, model_sha_ok: true },
    egress_ui: egress,
    egress_engine: egress,
    browser: { requests: 81, detects: 33, fault_detects: 3, resets: 2, external: [] },
    scan: { violations: [], files: 207 },
  };

  it("assembles the design shape and passes when all four proofs pass", () => {
    const r = assemble(parts);
    expect(r).toMatchObject({
      offline_check: parts.offline_check,
      egress: { ui: egress, engine: egress },
      browser_requests_external: 0,
      scan: { violations: [] },
      passed: true,
    });
  });

  it("any failing proof fails the report", () => {
    expect(assemble({ ...parts, egress_ui: { ...egress, blocked: false } }).passed).toBe(false);
    const ext = { ...parts.browser, external: ["https://x.example/"] };
    expect(assemble({ ...parts, browser: ext })).toMatchObject({
      browser_requests_external: 1,
      passed: false,
    });
    expect(assemble({ ...parts, scan: { violations: [{ file: "a", url: "b" }] } }).passed).toBe(
      false,
    );
    expect(assemble({ ...parts, offline_check: { ok: false } }).passed).toBe(false);
  });
});

describe("offline_report evidence", () => {
  it("a missing evidence file fails with its name (no stale pass)", () => {
    const dir = mkdtempSync(join(tmpdir(), "ev-"));
    for (const n of ["offline_check", "egress_ui", "egress_engine", "scan"]) {
      writeFileSync(join(dir, `${n}.json`), "{}");
    }
    expect(() => readEvidence(dir)).toThrow("browser.json");
  });
});
