import { Page, expect } from "@playwright/test";

/** Signs in via the demo RoleSelector (top-right of the header). The token
 *  lives only in the page's JS memory (lib/api.ts), matching how a real user
 *  session behaves — a reload always returns to signed-out. */
export async function signIn(page: Page, role: "Developer" | "Project Manager" | "Auditor") {
  await page.getByRole("button", { name: /sign in|developer|project manager|auditor/i }).click();
  await page.getByRole("option", { name: role }).click();
  await expect(page.getByRole("button", { name: role })).toBeVisible();
}

/** Sends a message through the composer using the Enter key (not a click on
 *  Send), exercising the keyboard-only submission path. */
export async function sendMessage(page: Page, text: string) {
  const input = page.getByLabel("Message the assistant");
  await input.fill(text);
  await input.press("Enter");
}

/** Waits for the most recent assistant bubble to stop streaming. */
export async function waitForAssistantReply(page: Page) {
  const log = page.getByRole("log", { name: "Conversation" });
  await expect(log.locator('[aria-hidden="true"].motion-safe\\:animate-pulse')).toHaveCount(0, {
    timeout: 15_000,
  });
}
