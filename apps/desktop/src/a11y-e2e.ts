import { invoke } from "@tauri-apps/api/core";
import {
  isStructuredDesktopError,
  locale,
  localizedError,
  setLocale,
  type StructuredDesktopError,
} from "./i18n";

type AccessibilityE2eReport = {
  schema: "denoize-desktop-a11y-e2e-v1";
  schemaVersion: 1;
  assertions: string[];
  failures: string[];
};

const nextFrame = () => new Promise<void>((resolve) => requestAnimationFrame(() => resolve()));
const delay = (milliseconds: number) => new Promise<void>((resolve) => window.setTimeout(resolve, milliseconds));

function accessibleName(element: HTMLElement): string {
  const direct = element.getAttribute("aria-label")?.trim();
  if (direct) return direct;
  const labelledBy = element.getAttribute("aria-labelledby")?.trim();
  if (labelledBy) {
    const value = labelledBy
      .split(/\s+/)
      .map((id) => document.getElementById(id)?.textContent?.trim() ?? "")
      .filter(Boolean)
      .join(" ");
    if (value) return value;
  }
  const labels = "labels" in element
    ? [...((element as HTMLInputElement).labels ?? [])]
    : [];
  const label = labels.map((item) => item.textContent?.trim() ?? "").filter(Boolean).join(" ");
  if (label) return label;
  if (element instanceof HTMLButtonElement || element instanceof HTMLAnchorElement) {
    return element.textContent?.trim() ?? "";
  }
  return "";
}

export async function runAccessibilityE2e(): Promise<void> {
  const report: AccessibilityE2eReport = {
    schema: "denoize-desktop-a11y-e2e-v1",
    schemaVersion: 1,
    assertions: [],
    failures: [],
  };
  const check = (name: string, passed: boolean, detail: string) => {
    report.assertions.push(name);
    if (!passed) report.failures.push(`${name}: ${detail}`.slice(0, 512));
  };

  report.assertions.push("runtime.started");
  try {

  const ids = [...document.querySelectorAll<HTMLElement>("[id]")].map((element) => element.id);
  check("runtime.unique-ids", new Set(ids).size === ids.length, "duplicate DOM ids exist");

  const controls = [...document.querySelectorAll<HTMLElement>(
    "button, input:not([type=hidden]), select, textarea, a[href], audio[controls], [role=slider]",
  )];
  const unnamed = controls.filter((control) => !accessibleName(control)).map((control) => control.id || control.tagName);
  check("runtime.control-names", unnamed.length === 0, `unnamed controls: ${unnamed.join(", ")}`);

  const tabs = [...document.querySelectorAll<HTMLButtonElement>('[role="tab"][data-page]')];
  const tabPairsValid = tabs.length === 10 && tabs.every((tab) => {
    const panel = document.getElementById(tab.getAttribute("aria-controls") ?? "");
    return panel?.getAttribute("role") === "tabpanel"
      && panel.getAttribute("aria-labelledby") === tab.id;
  });
  check("runtime.tab-panel-pairs", tabPairsValid, "navigation tabs and panels do not form ten valid pairs");
  check(
    "runtime.single-tab-stop",
    tabs.filter((tab) => tab.getAttribute("aria-selected") === "true" && tab.tabIndex === 0).length === 1,
    "navigation does not have exactly one selected tab stop",
  );

  const first = tabs[0]!;
  first.focus();
  first.dispatchEvent(new KeyboardEvent("keydown", { key: "ArrowDown", bubbles: true }));
  await nextFrame();
  const second = tabs[1]!;
  check(
    "runtime.keyboard-tab-navigation",
    document.activeElement === second
      && second.getAttribute("aria-selected") === "true"
      && document.getElementById("page-batch")?.getAttribute("aria-hidden") === "false"
      && document.getElementById("page-process")?.getAttribute("aria-hidden") === "true",
    "ArrowDown did not move focus, selection, and panel visibility together",
  );
  second.dispatchEvent(new KeyboardEvent("keydown", { key: "End", bubbles: true }));
  await nextFrame();
  check(
    "runtime.keyboard-end-navigation",
    document.activeElement === tabs.at(-1) && tabs.at(-1)?.getAttribute("aria-selected") === "true",
    "End did not select the final navigation tab",
  );
  tabs.at(-1)?.dispatchEvent(new KeyboardEvent("keydown", { key: "Home", bubbles: true }));
  await nextFrame();
  check(
    "runtime.keyboard-home-navigation",
    document.activeElement === first && first.getAttribute("aria-selected") === "true",
    "Home did not restore the first navigation tab",
  );
  const hiddenPanelsDisplayed = [...document.querySelectorAll<HTMLElement>('.page[aria-hidden="true"]')]
    .filter((panel) => getComputedStyle(panel).display !== "none");
  check("runtime.hidden-panels", hiddenPanelsDisplayed.length === 0, "ARIA-hidden pages remain visually displayed");

  const skip = document.querySelector<HTMLAnchorElement>(".skip-link")!;
  skip.focus();
  await delay(180);
  check(
    "runtime.skip-link-focus",
    document.activeElement === skip && skip.getBoundingClientRect().top >= 0,
    "the skip link is not visible when focused",
  );

  const toast = document.getElementById("toast")!;
  check(
    "runtime.live-region",
    toast.getAttribute("role") === "status"
      && toast.getAttribute("aria-live") === "polite"
      && toast.getAttribute("aria-atomic") === "true",
    "the notification live region is incomplete",
  );

  setLocale("en");
  await nextFrame();
  const englishTitle = document.getElementById("page-title")?.textContent ?? "";
  check(
    "runtime.english-locale",
    locale() === "en" && document.documentElement.lang === "en" && !/[ぁ-んァ-ヶ一-龠]/.test(englishTitle),
    "switching to English did not update the document and visible title",
  );

  let commandError: StructuredDesktopError | null = null;
  try {
    await invoke("cancel_job", { jobId: Number.MAX_SAFE_INTEGER });
  } catch (error) {
    if (isStructuredDesktopError(error)) commandError = error;
  }
  check(
    "runtime.structured-command-error",
    commandError?.code === "job.not-found"
      && commandError.technicalDetail.length > 0
      && !/[ぁ-んァ-ヶ一-龠]/.test(localizedError(commandError).split(" (Technical detail:")[0]),
    "Rust command errors did not cross the WebView boundary as a localizable structured object",
  );

  setLocale("ja");
  await nextFrame();
  check(
    "runtime.japanese-locale",
    locale() === "ja"
      && document.documentElement.lang === "ja"
      && /[ぁ-んァ-ヶ一-龠]/.test(document.getElementById("page-title")?.textContent ?? ""),
    "switching back to Japanese did not update the document and visible title",
  );
  if (commandError) {
    check(
      "runtime.localized-command-error",
      localizedError(commandError).startsWith("実行中の処理が見つかりません"),
      "the same structured error was not localized after the locale changed",
    );
  }
  } catch (error) {
    report.assertions.push("runtime.exception");
    report.failures.push(`runtime.exception: ${error instanceof Error ? error.message : String(error)}`.slice(0, 512));
  }

  await invoke("finish_accessibility_e2e", { report });
}
