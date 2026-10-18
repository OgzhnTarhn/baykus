# Baykuş operatör ekranı (`ui/`)

Next.js (App Router) + TypeScript + Tailwind. Tasarım: `docs/design/faz3.md` §4.4; plan:
`docs/plans/F3.md` Görev 28-33.

- Veri: `DATA_DIR` (varsayılan `/data`; `sequences/<dizi>/{rgb,ir,index.json}`), motor: `ENGINE_URL`
  (varsayılan `http://engine:8080`). `NEXT_TELEMETRY_DISABLED=1`.
- Komutlar: `npm ci`, `npm test` (Vitest), `npm run lint`, `npm run build`.
- Sürümler `package-lock.json` ile kilitli; `.npmrc` `legacy-peer-deps=true` (npm 10.9 eş çözücü hatası).
- Fontlar repoda (`public/fonts/`, Geist, SIL OFL 1.1); dış kaynak (CDN) yok.
