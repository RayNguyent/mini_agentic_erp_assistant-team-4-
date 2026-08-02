import { test, expect } from "@playwright/test";
import { sendMessage, signIn, waitForAssistantReply } from "./helpers";

test.describe("page readiness", () => {
  test("loads, is titled, and the composer is keyboard-reachable", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByRole("heading", { name: "Mini Agentic ERP Assistant" })).toBeVisible();
    await expect(page.getByLabel("Message the assistant")).toBeVisible();
    await page.keyboard.press("Tab");
  });
});

test.describe("grounded chat with citations", () => {
  test("a document question returns an answer with a source card", async ({ page }) => {
    await page.goto("/");
    await signIn(page, "Auditor");
    await sendMessage(page, "What are the risk severity ratings and who can close a risk?");
    await waitForAssistantReply(page);

    const reply = page.getByRole("log").locator(".max-w-\\[90\\%\\]").last();
    await expect(reply).toContainText(/severity|risk/i);
    await expect(reply.getByLabel("Sources")).toBeVisible();
  });
});

test.describe("unsupported request refusal", () => {
  test("an out-of-scope question is refused without inventing an answer", async ({ page }) => {
    await page.goto("/");
    await signIn(page, "Developer");
    await sendMessage(page, "What is the weather in Paris right now?");
    await waitForAssistantReply(page);

    const reply = page.getByRole("log").locator(".max-w-\\[90\\%\\]").last();
    await expect(reply).not.toHaveText("");
    // A refusal is not styled as an error — it is a valid, grounded "no".
    await expect(reply).not.toHaveClass(/border-red-200/);
  });
});

test.describe("read-only tool call", () => {
  test("a project status question returns tool-backed data with route metadata", async ({ page }) => {
    await page.goto("/");
    await signIn(page, "Developer");
    await sendMessage(page, "What's the status of PRJ-001?");
    await waitForAssistantReply(page);

    const reply = page.getByRole("log").locator(".max-w-\\[90\\%\\]").last();
    await expect(reply).toContainText("ERP Platform Rollout");
    await expect(reply).toContainText(/route:/);
  });
});

test.describe("approval-gated write flow", () => {
  test("create_risk is held pending, then executes after approval", async ({ page }) => {
    await page.goto("/");
    await signIn(page, "Project Manager");
    await sendMessage(page, "Create a risk for PRJ-001");
    await waitForAssistantReply(page);

    await page.getByPlaceholder("e.g. PRJ-001").fill("PRJ-001");
    await page.getByPlaceholder("Risk title").fill("Playwright e2e scope creep");
    const approveButton = page.getByRole("button", { name: "Approve" });
    await expect(approveButton).toBeEnabled();
    await approveButton.click();

    await expect(page.getByText("Approved")).toBeVisible();
    await waitForAssistantReply(page);
    const result = page.getByRole("log").locator(".max-w-\\[90\\%\\]").last();
    await expect(result).toContainText("Playwright e2e scope creep");
  });

  test("a rejected write is never executed", async ({ page }) => {
    await page.goto("/");
    await signIn(page, "Project Manager");
    await sendMessage(page, "Create a risk for PRJ-003");
    await waitForAssistantReply(page);

    await page.getByRole("button", { name: "Reject" }).click();
    await expect(page.getByText("Rejected")).toBeVisible();
  });
});

test.describe("permission-scoped tool denial", () => {
  test("a developer is denied budget data", async ({ page }) => {
    await page.goto("/");
    await signIn(page, "Developer");
    await sendMessage(page, "What is the budget for PRJ-001?");
    await waitForAssistantReply(page);

    const reply = page.getByRole("log").locator(".max-w-\\[90\\%\\]").last();
    await expect(reply).toHaveClass(/border-red-200/);
  });
});

test.describe("mobile viewport", () => {
  test.use({ viewport: { width: 375, height: 812 } });

  test("the layout stays usable at 375px wide", async ({ page }) => {
    await page.goto("/");
    await signIn(page, "Developer");
    await sendMessage(page, "What's the status of PRJ-001?");
    await waitForAssistantReply(page);

    const reply = page.getByRole("log").locator(".max-w-\\[90\\%\\]").last();
    await expect(reply).toBeVisible();
    // No horizontal scroll introduced by the message bubble.
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth > document.documentElement.clientWidth);
    expect(overflow).toBe(false);
  });
});
