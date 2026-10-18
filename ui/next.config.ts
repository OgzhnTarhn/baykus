import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Docker imajı (Görev 33) yalnız .next/standalone + static ile çalışır
  output: "standalone",
  cacheComponents: true,
  partialPrefetching: true,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
