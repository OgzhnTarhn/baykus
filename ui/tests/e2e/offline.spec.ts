// Kapalı ağ kanıtı 3 (Faz 3 tasarımı §4.5; plan Görev 33): normal compose (ağ açık), başsız tarayıcı
// klip dizisini iki tur oynatır (15 + 15 = 30 kare; tur başında oturum sıfırlanır) ve arızayı bir kez
// açar. Tarayıcının yaptığı bütün istekler kaydedilir; 127.0.0.1:3000 dışına tek istek girişimi
// (başarısız olsa bile) testi düşürür. Kayıt EVIDENCE_DIR/browser.json'a yazılır.
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { expect, test } from "@playwright/test";

const ORIGIN = new URL(process.env.UI_URL ?? "http://127.0.0.1:3000").origin;

function isLocal(url: string): boolean {
  if (url.startsWith("data:")) return true;
  if (url.startsWith("blob:")) return url.startsWith(`blob:${ORIGIN}/`);
  try {
    return new URL(url).origin === ORIGIN;
  } catch {
    return false;
  }
}

test("browser talks only to the local ui while playing with a fault", async ({ page }) => {
  const requests: string[] = [];
  const external: string[] = [];
  let detects = 0;
  let faultDetects = 0;
  let resets = 0;
  page.on("request", (r) => {
    requests.push(r.url());
    if (!isLocal(r.url())) external.push(r.url());
  });
  page.on("response", (r) => {
    const path = new URL(r.url()).pathname;
    if (path === "/api/session/reset" && r.status() === 200) resets += 1;
    if (path !== "/api/detect" || r.status() !== 200) return;
    detects += 1;
    if ((r.request().headers()["content-type"] ?? "").startsWith("multipart/form-data")) {
      faultDetects += 1;
    }
  });

  await page.goto("/");
  const play = page.getByRole("button", { name: "Oynat" });
  await expect(play).toBeEnabled();
  await page.getByLabel("Hız üst sınırı").selectOption("30");
  await play.click();
  await expect.poll(() => detects, { timeout: 120_000 }).toBeGreaterThanOrEqual(8);
  await page.getByLabel("IR kaydırma").selectOption("20"); // arıza bir kez açılır
  await expect.poll(() => faultDetects, { timeout: 60_000 }).toBeGreaterThanOrEqual(1);
  await page.getByRole("button", { name: "Arızaları kapat" }).click();
  await expect.poll(() => detects, { timeout: 180_000 }).toBeGreaterThanOrEqual(30);
  await page.getByRole("button", { name: "Durdur" }).click();

  const record = { requests: requests.length, detects, fault_detects: faultDetects, resets, external };
  const dir = process.env.EVIDENCE_DIR;
  if (dir) {
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, "browser.json"), `${JSON.stringify(record, null, 2)}\n`, "utf-8");
  }
  expect(resets).toBeGreaterThanOrEqual(1); // klip 15 kare: 30 karede tur başı sıfırlaması
  expect(external).toEqual([]);
});
