import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8787",
        changeOrigin: true,
        configure(proxy) {
          proxy.on("proxyReq", (request) => request.removeHeader("origin"));
        },
      },
    },
  },
  build: {
    chunkSizeWarningLimit: 650,
    rollupOptions: {
      input: {
        index: "index.html",
        performance: "performance.html",
        results: "results.html",
      },
      output: {
        manualChunks: {
          three: ["three"],
          editor: ["codemirror", "@codemirror/lang-python", "@codemirror/lint"],
        },
      },
    },
  },
});
