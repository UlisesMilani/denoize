import fs from "node:fs";

const mainSource = fs.readFileSync(new URL("../src/main.ts", import.meta.url), "utf8");
const i18nSource = fs.readFileSync(new URL("../src/i18n.ts", import.meta.url), "utf8");
const styles = fs.readFileSync(new URL("../src/styles.css", import.meta.url), "utf8");
const marker = 'document.querySelector<HTMLDivElement>("#app")!.innerHTML = `';
const start = mainSource.indexOf(marker);
const end = start < 0 ? -1 : mainSource.indexOf("`;", start + marker.length);
const failures = [];

function requireSource(pattern, message) {
  if (!pattern.test(mainSource)) failures.push(message);
}

function requireStyle(pattern, message) {
  if (!pattern.test(styles)) failures.push(message);
}

if (start < 0 || end < 0) {
  failures.push("could not locate the application template");
} else {
  const markup = mainSource.slice(start + marker.length, end);
  const ids = new Set([...markup.matchAll(/\sid="([^"]+)"/g)].map((match) => match[1]));
  const duplicateIds = [...ids].filter((id) => (markup.match(new RegExp(`\\sid="${id.replace(/[.*+?^${}()|[\\]\\]/g, "\\$&")}"`, "g")) ?? []).length > 1);
  if (duplicateIds.length) failures.push(`duplicate static ids: ${duplicateIds.join(", ")}`);

  for (const match of markup.matchAll(/\saria-(?:controls|labelledby)="([^"]+)"/g)) {
    for (const id of match[1].split(/\s+/)) {
      if (!ids.has(id)) failures.push(`ARIA reference has no static target: ${id}`);
    }
  }

  const tabs = [...markup.matchAll(/<button\b[^>]*\brole="tab"[^>]*>/g)].map((match) => match[0]);
  if (tabs.length !== 10) failures.push(`expected 10 navigation tabs, found ${tabs.length}`);
  for (const tab of tabs) {
    const id = /\sid="([^"]+)"/.exec(tab)?.[1];
    const controls = /\saria-controls="([^"]+)"/.exec(tab)?.[1];
    if (!id || !controls) failures.push(`navigation tab is missing id/aria-controls: ${tab}`);
    else if (!new RegExp(`<section\\b[^>]*id="${controls}"[^>]*role="tabpanel"[^>]*aria-labelledby="${id}"`).test(markup)) {
      failures.push(`navigation tab ${id} is not paired with tabpanel ${controls}`);
    }
  }

  if (/\stabindex="[1-9][0-9]*"/.test(markup)) failures.push("positive tabindex is forbidden");
}

requireSource(/<a class="skip-link" href="#main-content">/, "missing skip link");
requireSource(/<main id="main-content" tabindex="-1">/, "main content is not a programmatic focus target");
requireSource(/<nav role="tablist"[^>]*aria-orientation="vertical"/, "main navigation is not a vertical ARIA tablist");
requireSource(/id="locale-select"[^>]*aria-label="表示言語"/, "language selector has no accessible name");
requireSource(/id="toast" role="status" aria-live="polite" aria-atomic="true"/, "toast live region is incomplete");
requireSource(/class="progress-track" role="progressbar"[^>]*aria-valuenow="0"/, "job progressbar semantics are incomplete");
requireSource(/class="level-meter" role="meter"[^>]*aria-valuenow="0"/g, "live meters are missing meter semantics");
requireSource(/id="waveform"[^>]*role="slider"[^>]*aria-disabled="true"/, "preview seek control lacks disabled slider semantics");
requireSource(/id="ipc-result"[^>]*aria-live="polite"/, "IPC response preview is not an accessible live region");
requireSource(/role="tab" aria-controls="preview-audition-panel"/, "dynamic preview candidates are not ARIA tabs");
requireSource(/\["ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight", "Home", "End"\]/, "navigation tabs lack keyboard traversal");
requireSource(/\["ArrowLeft", "ArrowRight", "Home", "End"\]/, "preview controls lack keyboard traversal");
if (!/document\.documentElement\.lang = activeLocale/.test(i18nSource)) {
  failures.push("locale switching does not update the document language");
}

requireStyle(/\.toggle input\{position:absolute;[^}]*opacity:0;/, "custom checkboxes are not visually hidden while remaining focusable");
if (/\.toggle input\{display:none/.test(styles)) failures.push("custom checkboxes are removed from the accessibility tree");
requireStyle(/\.toggle input:focus-visible\+span\{outline:/, "custom checkboxes have no visible keyboard focus");
requireStyle(/@media\(prefers-reduced-motion:reduce\)/, "reduced-motion styling is missing");
requireStyle(/@media\(forced-colors:active\)/, "forced-colors styling is missing");

if (failures.length) {
  console.error(failures.join("\n"));
  process.exit(1);
}
console.log("desktop accessibility structure check passed");
