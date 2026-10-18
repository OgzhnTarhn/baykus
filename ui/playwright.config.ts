// Kapalı ağ kanıtı 3 (Faz 3 tasarımı §4.5): tarayıcı akışı. Yalnız CI'da (imaja girmez); compose
// zaten 127.0.0.1:3000'da çalışıyor olmalı. Yerelde sistem Chrome'u: PW_CHANNEL=chrome.
import { defineConfig } from "@playwright/test";

export default defineConfig({
  testDir: "tests/e2e",
  timeout: 240_000,
  retries: 0,
  reporter: [["list"]],
  use: {
    baseURL: process.env.UI_URL ?? "http://127.0.0.1:3000",
    channel: process.env.PW_CHANNEL || undefined,
    headless: true,
  },
});
