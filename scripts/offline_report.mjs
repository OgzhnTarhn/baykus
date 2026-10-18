// Kapalı ağ kanıtları 1-4'ü tek dosyada birleştirir (Faz 3 tasarımı §4.5, Kabul 6 otomatik kısmı):
// results/system/offline.json = {offline_check, egress: {ui, engine}, browser_requests_external,
// scan: {violations}, browser, passed}.
// Kullanım: node scripts/offline_report.mjs <kanıt klasörü> <çıktı>
//   klasörde: offline_check.json, egress_ui.json, egress_engine.json, browser.json, scan.json
import { readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { pathToFileURL } from "node:url";

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
  const read = (name) => JSON.parse(readFileSync(join(dir, `${name}.json`), "utf-8"));
  const r = assemble({
    offline_check: read("offline_check"),
    egress_ui: read("egress_ui"),
    egress_engine: read("egress_engine"),
    browser: read("browser"),
    scan: read("scan"),
  });
  writeFileSync(out, `${JSON.stringify(r, null, 2)}\n`, "utf-8");
  console.log(`offline.json: ${r.passed ? "geçti" : "DÜŞTÜ"}`);
  process.exit(r.passed ? 0 : 1);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) main();
