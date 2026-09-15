import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Proxy /images to backend so product images load in dev
      "/images": "http://localhost:8000",
    },
  },
});
