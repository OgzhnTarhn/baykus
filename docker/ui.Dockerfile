# Operatör ekranı imajı (Faz 3 tasarımı §4.4-4.5; plan Görev 33). Next standalone çıktısı; telemetri
# derlemede ve çalışmada kapalı; fontlar repoda. Kapalı ağ kanıt betikleri (offline_check,
# egress_check) ve models.json imajda: `docker compose exec ui node /app/scripts/...`.
# Derleme: docker build -f docker/ui.Dockerfile -t baykus-ui .
ARG NODE=node:22-bookworm-slim@sha256:c3de60bf2f9dd0ac6370e6117950ff62d6e339527e7472301c9c78a017978392

FROM ${NODE} AS deps
WORKDIR /app
COPY ui/package.json ui/package-lock.json ui/.npmrc ./
RUN npm ci --no-audit --no-fund

FROM deps AS build
ENV NEXT_TELEMETRY_DISABLED=1
COPY ui/ ./
RUN npm run build

FROM ${NODE} AS runtime
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    HOSTNAME=0.0.0.0 \
    PORT=3000 \
    DATA_DIR=/data \
    ENGINE_URL=http://engine:8080
WORKDIR /app
COPY --from=build /app/.next/standalone ./
COPY --from=build /app/.next/static ./.next/static
COPY --from=build /app/public ./public
COPY scripts/offline_check.mjs scripts/egress_check.mjs ./scripts/
COPY configs/engine/models.json ./models.json
USER node
EXPOSE 3000
CMD ["node", "server.js"]
