// Kapalı ağ kanıtı 2 (Faz 3 tasarımı §4.5; plan Görev 33): konteynerden dış adrese çıkış denemesi.
// Geçer yalnız beklenen ağ hatalarında; bağlantı kurulursa düşer; başka hata (izin, komut vb.) da
// düşer ("ağ engeli kanıtlanamadı"). Motor imajında aynı mantık: baykus_server --egress-check.
// Kullanım (ui imajında): node /app/scripts/egress_check.mjs   → JSON stdout; çıkış 0 = engelli.
import { lookup } from "node:dns/promises";
import { connect } from "node:net";
import { pathToFileURL } from "node:url";

export const EXPECTED = new Set([
  "ENETUNREACH",
  "EHOSTUNREACH",
  "ECONNREFUSED",
  "ETIMEDOUT",
  "EAI_AGAIN",
  "ENOTFOUND",
]);

export const TCP_TARGET = { host: "1.1.1.1", port: 443 };
export const DNS_TARGET = "example.com";
export const TIMEOUT_MS = 3000;

const probe = (target, result, connected) => ({
  target,
  result,
  connected,
  expected: !connected && EXPECTED.has(result),
});

export function probeTcp(host, port, timeoutMs = TIMEOUT_MS) {
  const target = `${host}:${port}`;
  return new Promise((resolve) => {
    const socket = connect({ host, port });
    const done = (r) => {
      clearTimeout(timer);
      socket.destroy();
      resolve(r);
    };
    const timer = setTimeout(() => done(probe(target, "ETIMEDOUT", false)), timeoutMs);
    socket.once("connect", () => done(probe(target, "connected", true)));
    socket.once("error", (e) => done(probe(target, e.code ?? String(e), false)));
  });
}

export async function probeDns(host, timeoutMs = TIMEOUT_MS) {
  let timer;
  const timeout = new Promise((resolve) => {
    timer = setTimeout(() => resolve(probe(host, "ETIMEDOUT", false)), timeoutMs);
  });
  const attempt = lookup(host).then(
    (a) => probe(host, `resolved ${a.address}`, true),
    (e) => probe(host, e.code ?? String(e), false),
  );
  const r = await Promise.race([attempt, timeout]);
  clearTimeout(timer);
  return r;
}

export function verdict(tcp, dns) {
  const connected = tcp.connected || dns.connected;
  const blocked = !connected && tcp.expected && dns.expected;
  const message = blocked
    ? "dış ağ kapalı"
    : connected
      ? "dış adrese ulaşıldı"
      : "ağ engeli kanıtlanamadı";
  return { tcp, dns, blocked, message };
}

async function main() {
  const tcp = await probeTcp(TCP_TARGET.host, TCP_TARGET.port);
  const dns = await probeDns(DNS_TARGET);
  const v = verdict(tcp, dns);
  process.stdout.write(`${JSON.stringify(v)}\n`);
  process.exit(v.blocked ? 0 : 1);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) await main();
