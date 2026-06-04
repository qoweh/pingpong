import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  assetsInclude: ["**/*.wasm", "**/*.mjb", "**/*.stl", "**/*.obj", "**/*.zip"],
  server: {
    fs: {
      strict: true
    }
  }
});
