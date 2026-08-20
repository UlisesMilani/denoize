import fs from "node:fs";
import ts from "typescript";

const japanesePattern = /[ぁ-んァ-ヶ一-龠]/;
const mainPath = new URL("../src/main.ts", import.meta.url);
const catalogPath = new URL("../src/i18n.ts", import.meta.url);
const rustPath = new URL("../src-tauri/src/lib.rs", import.meta.url);
const mainSource = fs.readFileSync(mainPath, "utf8");
const catalogSource = fs.readFileSync(catalogPath, "utf8");
const rustSource = fs.readFileSync(rustPath, "utf8");

function parse(path, source) {
  return ts.createSourceFile(path, source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS);
}

const catalogFile = parse("i18n.ts", catalogSource);
const translations = new Map();
const errorTranslations = new Map();
function collectCatalog(node) {
  if (ts.isVariableDeclaration(node)
    && ts.isIdentifier(node.name)
    && node.name.text === "english"
    && node.initializer
    && ts.isObjectLiteralExpression(node.initializer)) {
    for (const property of node.initializer.properties) {
      if (!ts.isPropertyAssignment(property)
        || (!ts.isStringLiteral(property.name) && !ts.isNoSubstitutionTemplateLiteral(property.name))
        || !ts.isStringLiteralLike(property.initializer)) continue;
      translations.set(property.name.text, property.initializer.text);
    }
  }
  if (ts.isVariableDeclaration(node)
    && ts.isIdentifier(node.name)
    && node.name.text === "errorMessages"
    && node.initializer
    && ts.isObjectLiteralExpression(node.initializer)) {
    for (const property of node.initializer.properties) {
      if (!ts.isPropertyAssignment(property)
        || (!ts.isStringLiteral(property.name) && !ts.isNoSubstitutionTemplateLiteral(property.name))
        || !ts.isObjectLiteralExpression(property.initializer)) continue;
      const values = {};
      for (const field of property.initializer.properties) {
        if (!ts.isPropertyAssignment(field)
          || (!ts.isIdentifier(field.name) && !ts.isStringLiteralLike(field.name))
          || !ts.isStringLiteralLike(field.initializer)) continue;
        values[field.name.text] = field.initializer.text;
      }
      errorTranslations.set(property.name.text, values);
    }
  }
  ts.forEachChild(node, collectCatalog);
}
collectCatalog(catalogFile);

const failures = [];
for (const [japanese, english] of translations) {
  if (!japanesePattern.test(japanese)) failures.push(`catalog key is not Japanese: ${JSON.stringify(japanese)}`);
  if (japanesePattern.test(english)) failures.push(`English translation still contains Japanese: ${JSON.stringify(japanese)}`);
}

const classifierStart = rustSource.indexOf("fn classify(technical_detail: String)");
const classifierEnd = rustSource.indexOf("impl From<String> for DesktopError", classifierStart);
if (classifierStart < 0 || classifierEnd < 0) failures.push("could not locate the Rust structured-error classifier");
const classifierSource = classifierStart < 0 || classifierEnd < 0
  ? ""
  : rustSource.slice(classifierStart, classifierEnd);
const rustErrorCodes = new Set(
  [...classifierSource.matchAll(/"([a-z]+\.[a-z-]+)"/g)].map((match) => match[1]),
);
for (const code of rustErrorCodes) {
  const message = errorTranslations.get(code);
  if (!message) {
    failures.push(`structured error translation missing: ${code}`);
    continue;
  }
  if (!japanesePattern.test(message.ja ?? "")) failures.push(`structured error Japanese message missing: ${code}`);
  if (!message.en || japanesePattern.test(message.en)) failures.push(`structured error English message invalid: ${code}`);
}
for (const code of errorTranslations.keys()) {
  if (!rustErrorCodes.has(code)) failures.push(`unused structured error translation: ${code}`);
}
if (mainSource.includes("tr(payload.error")) failures.push("structured errors must use localizedError(), not tr()");
if (!mainSource.includes("localizedError(error)")) failures.push("main.ts does not route command errors through localizedError()");

const templateMarker = 'document.querySelector<HTMLDivElement>("#app")!.innerHTML = `';
const templateStart = mainSource.indexOf(templateMarker);
const templateBodyStart = templateStart + templateMarker.length;
const templateEnd = mainSource.indexOf("`;", templateBodyStart);
if (templateStart < 0 || templateEnd < 0) failures.push("could not locate the application template");

if (templateStart >= 0 && templateEnd >= 0) {
  const markup = mainSource.slice(templateBodyStart, templateEnd);
  const staticPhrases = new Set();
  for (const match of markup.matchAll(/>([^<>]+)</g)) {
    const value = match[1].replace(/\s+/g, " ").trim();
    if (japanesePattern.test(value) && value !== "日本語") staticPhrases.add(value);
  }
  for (const match of markup.matchAll(/(?:aria-label|placeholder|title)="([^"]+)"/g)) {
    if (japanesePattern.test(match[1])) staticPhrases.add(match[1]);
  }
  for (const value of [...staticPhrases].sort((a, b) => a.localeCompare(b, "ja"))) {
    if (!translations.has(value)) failures.push(`static UI translation missing: ${JSON.stringify(value)}`);
  }
}

const mainFile = parse("main.ts", mainSource);
function insideTrCall(node) {
  for (let current = node.parent; current; current = current.parent) {
    if (ts.isCallExpression(current)
      && ts.isIdentifier(current.expression)
      && current.expression.text === "tr") return true;
    if (ts.isStatement(current)) return false;
  }
  return false;
}

function checkDynamicLiterals(node) {
  const literal = ts.isStringLiteralLike(node)
    || ts.isTemplateHead(node)
    || ts.isTemplateMiddle(node)
    || ts.isTemplateTail(node);
  if (literal && japanesePattern.test(node.text)) {
    const start = node.getStart(mainFile);
    const inStaticTemplate = start <= templateEnd && node.getEnd() >= templateBodyStart;
    if (!inStaticTemplate && !insideTrCall(node)) {
      const line = mainFile.getLineAndCharacterOfPosition(start).line + 1;
      failures.push(`dynamic Japanese literal is not wrapped in tr() at main.ts:${line}: ${JSON.stringify(node.text)}`);
    }
  }
  ts.forEachChild(node, checkDynamicLiterals);
}
checkDynamicLiterals(mainFile);

function checkTranslationCalls(node) {
  if (ts.isCallExpression(node)
    && ts.isIdentifier(node.expression)
    && node.expression.text === "tr"
    && node.arguments.length === 1
    && ts.isStringLiteralLike(node.arguments[0])
    && japanesePattern.test(node.arguments[0].text)
    && !translations.has(node.arguments[0].text)) {
    const line = mainFile.getLineAndCharacterOfPosition(node.getStart(mainFile)).line + 1;
    failures.push(`tr() call has no explicit or catalog translation at main.ts:${line}: ${JSON.stringify(node.arguments[0].text)}`);
  }
  ts.forEachChild(node, checkTranslationCalls);
}
checkTranslationCalls(mainFile);

if (failures.length) {
  console.error(failures.join("\n"));
  process.exit(1);
}
console.log(`desktop i18n check passed (${translations.size} catalog entries, ${errorTranslations.size} structured errors)`);
