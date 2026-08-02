import { test, expect } from "@playwright/test";
import { sendMessage, signIn, waitForAssistantReply } from "./helpers";

test.describe("document ACL filtering", () => {
  test("a restricted document is never cited for a developer, but is for an auditor", async ({ page }) => {
    await page.goto("/");
    await signIn(page, "Developer");
    await sendMessage(page, "What is the licensing negotiation target?");
    await waitForAssistantReply(page);

    const devReply = page.getByRole("log").locator(".max-w-\\[90\\%\\]").last();
    await expect(devReply.getByLabel("Sources")).toHaveCount(0);

    await page.reload();
    await signIn(page, "Auditor");
    await sendMessage(page, "What is the licensing negotiation target?");
    await waitForAssistantReply(page);

    const auditorReply = page.getByRole("log").locator(".max-w-\\[90\\%\\]").last();
    await expect(auditorReply.getByLabel("Sources")).toBeVisible();
  });
});

test.describe("auth required", () => {
  test("sending a message while signed out surfaces a clear sign-in prompt, not a silent failure", async ({ page }) => {
    await page.goto("/");
    await sendMessage(page, "What's the status of PRJ-001?");
    await waitForAssistantReply(page);

    const reply = page.getByRole("log").locator(".max-w-\\[90\\%\\]").last();
    await expect(reply).toContainText(/sign in/i);
  });

  test("a session never persists a credential to storage", async ({ page }) => {
    await page.goto("/");
    await signIn(page, "Project Manager");
    const storage = await page.evaluate(() => ({
      local: JSON.stringify(localStorage),
      session: JSON.stringify(sessionStorage),
    }));
    expect(storage.local).not.toContain("pm-token");
    expect(storage.session).not.toContain("pm-token");
    expect(await page.context().cookies()).toHaveLength(0);
  });
});
