import { defineConfig, devices } from "@playwright/test";

/**
 * Launches both the FastAPI backend and the Next.js dev server against the
 * deterministic offline profile (no LLM_PROVIDER set = no credential
 * required), so this suite is reproducible in CI or on a clean checkout with
 * no external service reachable.
 *
 * Run from frontend/: `npm run test:e2e`.
 */
export default defineConfig({
  testDir: "./e2e",
  fullyParallel: false, // the backend's approval store and mock ERP are shared, in-memory state
  retries: process.env.CI ? 1 : 0,
  reporter: [
    ["list"],
    ["json", { outputFile: "evidence/playwright-results.json" }],
    ["html", { outputFolder: "evidence/playwright-report", open: "never" }],
  ],
  use: {
    baseURL: "http://127.0.0.1:3000",
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [
    { name: "desktop-chromium", use: { ...devices["Desktop Chrome"] } },
    { name: "mobile-safari", use: { ...devices["iPhone 13"] } },
  ],
  webServer: [
    {
      command:
        "cd .. && uv run uvicorn app.api.main:app --port 8000",
      url: "http://127.0.0.1:8000/health",
      reuseExistingServer: !process.env.CI,
      timeout: 30_000,
    },
    {
      command: "npm run dev",
      url: "http://127.0.0.1:3000",
      reuseExistingServer: !process.env.CI,
      timeout: 30_000,
    },
  ],
});
