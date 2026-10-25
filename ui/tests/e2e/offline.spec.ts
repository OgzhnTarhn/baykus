// Kapalı ağ kanıtı 3 (Faz 3 tasarımı §4.5; plan Görev 33): normal compose (ağ açık), başsız tarayıcı
// klip dizisini iki tur oynatır (15 + 15 = 30 kare; tur başında oturum sıfırlanır) ve arızayı bir kez
// açar. Faz 4 (§5.8, §5.13): bölge çizilir, sekmeler ve düzen değiştirilir, üç dışa aktarma indirilir.
// Tarayıcının yaptığı bütün istekler kaydedilir; 127.0.0.1:3000 dışına tek istek girişimi
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
  // Arıza kontrolleri "Sağlık" sekmesinde (Faz 4 §5.8).
  await page.getByRole("tab", { name: "Sağlık" }).click();
  await page.getByLabel("IR kaydırma").selectOption("20"); // arıza bir kez açılır
  await expect.poll(() => faultDetects, { timeout: 60_000 }).toBeGreaterThanOrEqual(1);
  await page.getByRole("button", { name: "Arızaları kapat" }).click();
  await expect.poll(() => detects, { timeout: 180_000 }).toBeGreaterThanOrEqual(30);

  // Bölge çizimi (Faz 4 §5.8): "Bölgeler" sekmesinde araç seçilir, haritaya dört nokta konur; son
  // nokta çift tıklamadır ve çokgeni bitirir. Noktalar tuvalin %25–%75 aralığındadır.
  await page.getByRole("tab", { name: "Bölgeler" }).click();
  await page.getByRole("button", { name: "Alan", exact: true }).click();
  const map = page.getByRole("img", { name: /^Göreli harita/ });
  const box = await map.boundingBox();
  expect(box).not.toBeNull();
  const at = (fx: number, fy: number): [number, number] => [
    box!.x + box!.width * fx,
    box!.y + box!.height * fy,
  ];
  const points: [number, number][] = [at(0.3, 0.3), at(0.7, 0.3), at(0.7, 0.7), at(0.3, 0.7)];
  for (const [x, y] of points.slice(0, 3)) await page.mouse.click(x, y);
  await page.mouse.dblclick(points[3][0], points[3][1]);
  const zoneRows = page.getByRole("listbox", { name: "Bölgeler" }).getByRole("option");
  await expect(zoneRows).toHaveCount(1);
  await expect(zoneRows.first()).toContainText("Alan");
  await expect(page.getByLabel("Bölge adı")).toHaveValue("Alan 1");

  // Sekmeler ve düzen (Faz 4 §5.8): paneller açılır, harita tek başına ve yan yana denenir.
  await page.getByRole("tab", { name: "Alarmlar" }).click();
  await page.getByRole("tab", { name: "Olaylar" }).click();
  await page.getByRole("button", { name: "Harita", exact: true }).click();
  await page.getByRole("button", { name: "İkisi", exact: true }).click();

  await page.getByRole("button", { name: "Durdur" }).click();

  // Dışa aktarma (Faz 4 §5.13): dosyalar tarayıcıda üretilir, adları baykus_<dizi>_ ile başlar.
  let downloads = 0;
  const exportItem = async (label: string, ext: string): Promise<void> => {
    await page.getByRole("button", { name: "Dışa aktar" }).click();
    const item = page.getByRole("button", { name: label, exact: true });
    await expect(item).toBeEnabled();
    const [dl] = await Promise.all([page.waitForEvent("download"), item.click()]);
    const name = dl.suggestedFilename();
    expect(name.startsWith("baykus_clip_")).toBe(true);
    expect(name.endsWith(`.${ext}`)).toBe(true);
    downloads += 1;
  };
  await exportItem("Olaylar (JSON)", "json");
  await exportItem("Olaylar (CSV)", "csv");
  await exportItem("Harita (PNG: tamamı)", "png");

  const record = {
    requests: requests.length,
    detects,
    fault_detects: faultDetects,
    resets,
    downloads,
    zones_drawn: true,
    external,
  };
  const dir = process.env.EVIDENCE_DIR;
  if (dir) {
    mkdirSync(dir, { recursive: true });
    writeFileSync(join(dir, "browser.json"), `${JSON.stringify(record, null, 2)}\n`, "utf-8");
  }
  expect(downloads).toBe(3);
  expect(resets).toBeGreaterThanOrEqual(1); // klip 15 kare: 30 karede tur başı sıfırlaması
  expect(external).toEqual([]);
});
