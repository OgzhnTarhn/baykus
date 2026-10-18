// Kapalı ağ kanıtları 1-4'ü tek dosyada birleştirir (Faz 3 tasarımı §4.5, Kabul 6 otomatik kısmı):
// results/system/offline.json = {offline_check, egress: {ui, engine}, browser_requests_external,
// scan: {violations}, browser, passed}.
// Kullanım: node scripts/offline_report.mjs <kanıt klasörü> <çıktı>
//   klasörde: offline_check.json, egress_ui.json, egress_engine.json, browser.json, scan.json
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

export const EVIDENCE = ["offline_check", "egress_ui", "egress_engine", "browser", "scan"];

// Eksik dosya hatadır (adım düşmüş demektir); offline_proof.sh eski dosyaları baştan siler, bayat bir
// "geçti" okunmaz.
export function readEvidence(dir) {
  const out = {};
  for (const name of EVIDENCE) {
    const file = join(dir, `${name}.json`);
    if (!existsSync(file)) throw new Error(`kanıt dosyası yok: ${name}.json (o adım düştü)`);
    out[name] = JSON.parse(readFileSync(file, "utf-8"));
  }
  return out;
}

export function assemble({ offline_check, egress_ui, egress_engine, browser, scan }) {
  const external = browser.external.length;
  const passed =
    offline_check.ok === true &&
    egress_ui.blocked === true &&
    egress_engine.blocked === true &&
    external === 0 &&
    scan.violations.length === 0;
  return {
    offline_check,
    egress: { ui: egress_ui, engine: egress_engine },
    browser_requests_external: external,
    browser,
    scan: { violations: scan.violations, files: scan.files },
    passed,
  };
}

function main() {
  const [dir, out] = process.argv.slice(2);
  if (!dir || !out) {
    console.error("kullanım: node offline_report.mjs <kanıt klasörü> <çıktı>");
    process.exit(2);
  }
  let r;
  try {
    r = assemble(readEvidence(dir));
  } catch (e) {
    console.error(`offline.json: DÜŞTÜ — ${e instanceof Error ? e.message : e}`);
    process.exit(1);
  }
  writeFileSync(out, `${JSON.stringify(r, null, 2)}\n`, "utf-8");
  console.log(`offline.json: ${r.passed ? "geçti" : "DÜŞTÜ"}`);
  process.exit(r.passed ? 0 : 1);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) main();
