import { expect, test } from "@playwright/test";

test("Worker/WASM initializes across engines and Chromium drains an observed quantum", async ({
  browserName,
  page,
}) => {
  const pageErrors = [];
  page.on("pageerror", (error) => pageErrors.push(String(error)));
  const render = browserName === "chromium" ? "1" : "0";
  await page.goto(`/web/test/browser-fixture.html?render=${render}`);
  await expect(page.locator("#start")).toBeVisible();
  expect(await page.evaluate(() => globalThis.crossOriginIsolated)).toBe(true);
  await page.locator("#start").click();
  await expect
    .poll(() => page.evaluate(() => globalThis.denoizeBrowserResult.state))
    .toMatch(/^(failed|stopped)$/);
  const result = await page.evaluate(() => globalThis.denoizeBrowserResult);
  expect(result.state, result.error).toBe("stopped");
  expect(result.errorCode).toBe(0);
  expect(result.generation).toBe(1);
  if (browserName === "chromium") {
    expect(result.mode).toBe("render-finish");
    expect(result.renderQuantum).toBeGreaterThan(0);
  } else {
    expect(result.mode).toBe("initialize-cancel");
    expect(result.renderQuantum).toBeGreaterThanOrEqual(0);
  }
  expect(pageErrors).toEqual([]);
});
