// Kapalı ağ kanıtı 4 (ikincil; Faz 3 tasarımı §4.5): derlenmiş Next çıktısında http(s):// adresleri;
// izin listesinde olmayan varsa çıkış 1. Satır önek eşleşmesidir; `=` ile başlayan satır birebir
// eşleşme (ör. kod içi örnek "https://a"). Dinamik kurulan adresleri yakalamaz.
// `cache/` (derleme önbelleği) ve `standalone/` (node_modules kopyası) taranmaz: taranan, tarayıcıya
// ve sunucuya giden derlenmiş uygulama (static/, server/ ve kök dosyalar).
// Kullanım: node scripts/scan_next.mjs ui/.next scripts/scan_allow.txt [--out scan.json]
import { readdirSync, readFileSync, writeFileSync } from "node:fs";
import { join, relative, sep } from "node:path";
import { pathToFileURL } from "node:url";

const SKIP_DIRS = new Set(["cache", "standalone"]);
const TEXT_EXT = /\.(js|mjs|cjs|html|json|css|txt|rsc|body|meta|map)$/;
const URL_RE = /https?:\/\/[^\s"'`<>()\\{}|^,;]+/g;

export function loadAllow(file) {
  return readFileSync(file, "utf-8")
    .split(/\r?\n/)
    .map((l) => l.trim())
    .filter((l) => l !== "" && !l.startsWith("#"));
}

function* walk(dir, root) {
  for (const e of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, e.name);
    if (e.isDirectory()) {
      if (dir === root && SKIP_DIRS.has(e.name)) continue;
      yield* walk(p, root);
    } else if (TEXT_EXT.test(e.name)) {
      yield p;
    }
  }
}

export function scan(root, allow) {
  const violations = [];
  const seen = new Set();
  let files = 0;
  for (const p of walk(root, root)) {
    files += 1;
    const file = relative(root, p).split(sep).join("/");
    for (const m of readFileSync(p, "utf-8").matchAll(URL_RE)) {
      const url = m[0];
      if (allow.some((a) => (a.startsWith("=") ? url === a.slice(1) : url.startsWith(a)))) continue;
      const key = `${file} ${url}`;
      if (seen.has(key)) continue;
      seen.add(key);
      violations.push({ file, url });
    }
  }
  return { files, violations };
}

function main() {
  const [dir, allowFile, ...rest] = process.argv.slice(2);
  if (!dir || !allowFile) {
    console.error("kullanım: node scan_next.mjs <.next klasörü> <izin listesi> [--out dosya]");
    process.exit(2);
  }
  const r = scan(dir, loadAllow(allowFile));
  const json = JSON.stringify({ violations: r.violations, files: r.files }, null, 2);
  const out = rest[0] === "--out" ? rest[1] : null;
  if (out) writeFileSync(out, `${json}\n`, "utf-8");
  process.stdout.write(`${json}\n`);
  process.exit(r.violations.length === 0 ? 0 : 1);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) main();
