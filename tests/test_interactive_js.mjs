/* Tests for the JavaScript half of the repo.
 *
 *     node tests/test_interactive_js.mjs
 *
 * The interactive pages compute their own numbers, so that math needs the same
 * scrutiny as the Python. `LA` is extracted out of the shared runtime (it is
 * pure — no DOM), and the hand-built head matrices from the attention page are
 * re-derived here and checked for the behaviour the page claims they have.
 */

import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join, resolve } from "node:path";
import assert from "node:assert/strict";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");

// --- load LA out of the browser runtime ------------------------------------

const src = readFileSync(join(ROOT, "interactive/shared/learnai.js"), "utf8");
const start = src.indexOf("const LA = {");
const end = src.indexOf("const Viz = {");
assert.ok(start > 0 && end > start, "could not locate LA in learnai.js");
const LA = new Function(src.slice(start, end).replace("const LA = {", "return {").replace(/};\s*$/, "};"))();
// LA's methods call each other through the `LA` identifier, which in the browser
// resolves to the module-level const. Give them the same global here.
globalThis.LA = LA;

function pageSource(path) {
  return readFileSync(join(ROOT, "interactive", path, "index.html"), "utf8");
}

function sourceBetween(source, startMarker, endMarker) {
  const startAt = source.indexOf(startMarker);
  const endAt = source.indexOf(endMarker, startAt);
  assert.ok(startAt >= 0 && endAt > startAt,
    `could not find source markers ${JSON.stringify(startMarker)} → ${JSON.stringify(endMarker)}`);
  return source.slice(startAt, endAt);
}

function pageExports(path, startMarker, endMarker, names) {
  const chunk = sourceBetween(pageSource(path), startMarker, endMarker);
  return new Function("LA", `${chunk}\nreturn { ${names.join(", ")} };`)(LA);
}

// --- test harness ----------------------------------------------------------

let passed = 0;
const failures = [];
function test(name, fn) {
  try {
    fn();
    passed++;
    console.log(`  ok   ${name}`);
  } catch (e) {
    failures.push([name, e.message]);
    console.log(`  FAIL ${name}\n       ${e.message}`);
  }
}
const close = (a, b, tol = 1e-9) => Math.abs(a - b) < tol;

// --- LA --------------------------------------------------------------------

console.log("\nLA");

test("softmax sums to 1 and is order-preserving", () => {
  const p = LA.softmax([3, 1, 0, -2]);
  assert.ok(close(p.reduce((a, b) => a + b, 0), 1));
  assert.ok(p[0] > p[1] && p[1] > p[2] && p[2] > p[3]);
});

test("softmax is stable on huge inputs", () => {
  const p = LA.softmax([1000, 999, 998]);
  assert.ok(p.every(Number.isFinite), "overflowed to NaN/Inf");
  assert.ok(close(p.reduce((a, b) => a + b, 0), 1, 1e-12));
});

test("softmax is shift-invariant", () => {
  const a = LA.softmax([1, 2, 3]);
  const b = LA.softmax([101, 102, 103]);
  a.forEach((v, i) => assert.ok(close(v, b[i], 1e-12)));
});

test("temperature: low sharpens, high flattens", () => {
  const logits = [3, 1, 0, -1];
  const hot = LA.entropy(LA.softmax(logits, 4));
  const cold = LA.entropy(LA.softmax(logits, 0.2));
  assert.ok(cold < hot, `expected cold(${cold}) < hot(${hot})`);
  assert.ok(hot < Math.log(4) + 1e-9, "entropy cannot exceed log(n)");
});

test("matmul matches a hand-computed product", () => {
  const C = LA.matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]]);
  assert.deepEqual(C, [[19, 22], [43, 50]]);
});

test("transpose round-trips", () => {
  const A = [[1, 2, 3], [4, 5, 6]];
  assert.deepEqual(LA.transpose(LA.transpose(A)), A);
});

test("attention rows are probability distributions", () => {
  const Q = LA.seededMatrix(6, 8, 1), K = LA.seededMatrix(6, 8, 2), V = LA.seededMatrix(6, 4, 3);
  const { weights, out } = LA.attention(Q, K, V);
  weights.forEach((r) => {
    assert.ok(close(r.reduce((a, b) => a + b, 0), 1, 1e-12));
    assert.ok(r.every((v) => v >= 0 && v <= 1));
  });
  assert.equal(out.length, 6);
  assert.equal(out[0].length, 4);
});

test("causal mask zeroes the strict upper triangle", () => {
  const Q = LA.seededMatrix(7, 8, 11), K = LA.seededMatrix(7, 8, 12);
  const { weights } = LA.attention(Q, K, Q, { causal: true });
  weights.forEach((r, i) => r.forEach((v, j) => {
    if (j > i) assert.equal(v, 0, `weight[${i}][${j}] = ${v}, expected 0`);
  }));
});

test("attention output is a convex combination of value rows", () => {
  // Every output coordinate must lie within the range of that column of V.
  const Q = LA.seededMatrix(5, 6, 21), K = LA.seededMatrix(5, 6, 22), V = LA.seededMatrix(5, 3, 23);
  const { out } = LA.attention(Q, K, V);
  for (let j = 0; j < 3; j++) {
    const col = V.map((r) => r[j]);
    const lo = Math.min(...col), hi = Math.max(...col);
    out.forEach((r) => assert.ok(r[j] >= lo - 1e-9 && r[j] <= hi + 1e-9));
  }
});

test("entropy: 0 for a spike, log(n) for uniform", () => {
  assert.ok(close(LA.entropy([1, 0, 0, 0]), 0, 1e-12));
  assert.ok(close(LA.entropy([0.25, 0.25, 0.25, 0.25]), Math.log(4), 1e-12));
});

test("unscaled attention saturates as d grows; scaled does not", () => {
  const T = 12;
  const measure = (d, scale) => {
    const Q = LA.seededMatrix(T, d, 991 + d), K = LA.seededMatrix(T, d, 7717 + d);
    const { weights } = LA.attention(Q, K, Q, { scale });
    return weights.reduce((a, r) => a + LA.entropy(r), 0) / T;
  };
  const unscaledSmall = measure(4, 1), unscaledLarge = measure(512, 1);
  const scaledSmall = measure(4, 1 / 2), scaledLarge = measure(512, 1 / Math.sqrt(512));
  assert.ok(unscaledLarge < unscaledSmall * 0.5,
    `unscaled entropy should collapse: ${unscaledSmall} -> ${unscaledLarge}`);
  assert.ok(Math.abs(scaledLarge - scaledSmall) < 0.5,
    `scaled entropy should stay flat: ${scaledSmall} -> ${scaledLarge}`);
});

test("seeded matrices are reproducible", () => {
  assert.deepEqual(LA.seededMatrix(3, 4, 7), LA.seededMatrix(3, 4, 7));
  assert.notDeepEqual(LA.seededMatrix(3, 4, 7), LA.seededMatrix(3, 4, 8));
});

// --- the attention page's hand-built heads ----------------------------------

console.log("\nattention page — hand-built heads");

const attentionPage = pageExports(
  "attention",
  "const D_SEM =",
  "const SEM_NAMES =",
  ["D_SEM", "D_POS", "D", "positional", "prevTokenHead", "contentHead"],
);
const { D_SEM, D_POS, D, positional, prevTokenHead, contentHead } = attentionPage;

test("previous-token head peaks at j = i-1 for every query", () => {
  const T = 9;
  const E = Array.from({ length: T }, (_, p) => [...new Array(D_SEM).fill(0), ...positional(p)]);
  const { Wq, Wk } = prevTokenHead();
  const { weights } = LA.attention(LA.matmul(E, Wq), LA.matmul(E, Wk), E,
    { scale: 1 / Math.sqrt(D_POS) });
  weights.forEach((row, i) => {
    const argmax = row.indexOf(Math.max(...row));
    const want = Math.max(i - 1, 0);
    assert.equal(argmax, want, `row ${i} attends to ${argmax}, expected ${want}`);
  });
});

test("content head groups same-class words and ignores position", () => {
  // Two "animate" tokens and two "action" tokens, interleaved so that any
  // position-based head would give a different answer than a content one.
  const sem = { animate: [1, 0, 0, 0, 0, 0], action: [0, 1, 0, 0, 0, 0] };
  const classes = ["animate", "action", "animate", "action"];
  const E = classes.map((c, p) => [...sem[c], ...positional(p)]);
  const { Wq, Wk } = contentHead();
  const { weights } = LA.attention(LA.matmul(E, Wq), LA.matmul(E, Wk), E,
    { scale: 1 / Math.sqrt(D_SEM) });
  // Token 0 (animate) must give more weight to token 2 (animate) than to 1 or 3.
  assert.ok(weights[0][2] > weights[0][1], "animate should prefer animate over action");
  assert.ok(weights[0][2] > weights[0][3], "animate should prefer animate over action");
  // ...and treat the two same-class tokens as interchangeable (position-blind).
  assert.ok(Math.abs(weights[0][0] - weights[0][2]) < 1e-9,
    "a content head must not distinguish two identical meanings by position");
});

// --- sampling page ---------------------------------------------------------

console.log("\nsampling page");

const { applySampling } = pageExports(
  "sampling",
  "function applySampling",
  "function label",
  ["applySampling"],
);

test("top-p keeps the smallest prefix that crosses the requested mass", () => {
  const probs = [0.60, 0.25, 0.10, 0.05];
  const rows = applySampling(
    ["a", "b", "c", "d"],
    probs.map(Math.log),
    { temp: 1, topK: 0, topP: 0.80 },
  );
  assert.deepEqual(rows.map((r) => r.kept), [true, true, false, false]);
  assert.ok(close(rows.reduce((sum, r) => sum + r.prob, 0), 1, 1e-12));
});

test("top-p is computed after top-k, matching GPT.generate", () => {
  const rows = applySampling(
    ["a", "b", "c", "d"],
    [5, 4, 3, 2],
    { temp: 1, topK: 2, topP: 0.70 },
  );
  // After top-k, softmax([5, 4]) gives the first token 73.1%, so p=.70
  // keeps only that token. Measuring p on all four logits would wrongly keep 2.
  assert.deepEqual(rows.map((r) => r.kept), [true, false, false, false]);
  assert.equal(rows[0].prob, 1);
});

test("temperature changes sharpness but never token ranking", () => {
  const logits = [3, 2, 1, 0];
  const cold = applySampling(["a", "b", "c", "d"], logits,
    { temp: 0.25, topK: 0, topP: 1 });
  const hot = applySampling(["a", "b", "c", "d"], logits,
    { temp: 2, topK: 0, topP: 1 });
  assert.ok(cold[0].prob > hot[0].prob);
  assert.deepEqual(
    cold.map((r) => r.prob).map((_, i) => i).sort((a, b) => cold[b].prob - cold[a].prob),
    hot.map((r) => r.prob).map((_, i) => i).sort((a, b) => hot[b].prob - hot[a].prob),
  );
});

// --- positional page -------------------------------------------------------

console.log("\npositional page");

const { scoreMatrix } = pageExports(
  "positional",
  "const BASE =",
  "// ===========================================================================\n// Copy for each scheme",
  ["scoreMatrix"],
);

function diagonalSpread(M, distance) {
  const values = [];
  for (let i = distance; i < M.length; i++) values.push(M[i][i - distance]);
  return Math.max(...values) - Math.min(...values);
}

test("no encoding contributes exactly zero positional score", () => {
  const M = scoreMatrix("none", 12, 16);
  assert.ok(M.flat().every((v) => v === 0));
});

test("RoPE and ALiBi scores are exactly relative", () => {
  for (const scheme of ["rope", "alibi"]) {
    const M = scoreMatrix(scheme, 16, 32);
    for (let d = 0; d < 8; d++) {
      assert.ok(diagonalSpread(M, d) < 1e-10,
        `${scheme} diagonal ${d} varied by ${diagonalSpread(M, d)}`);
    }
  }
});

test("additive sinusoidal scores retain absolute cross terms", () => {
  const M = scoreMatrix("sinusoidal", 16, 32);
  assert.ok(diagonalSpread(M, 3) > 1e-3,
    "full content + sinusoid score should not be presented as exactly relative");
});

test("learned positional scores are not translation-invariant", () => {
  const M = scoreMatrix("learned", 16, 32);
  assert.ok(diagonalSpread(M, 3) > 0.1);
});

// --- tokenizer page --------------------------------------------------------

console.log("\ntokenizer page");

const tokenizerPage = pageExports(
  "tokenizer",
  "const PAT =",
  "// ===========================================================================\n// State",
  ["trainBPE", "tokenize"],
);

test("BPE training and encoding replay the learned merge order", () => {
  const merges = tokenizerPage.trainBPE("low low lower lowest", 20);
  const ranks = new Map(merges.map((m) => [m.a + "\0" + m.b, m.rank]));
  const chars = tokenizerPage.tokenize("lowest", ranks, 0);
  const merged = tokenizerPage.tokenize("lowest", ranks, merges.length);
  assert.equal(chars.join(""), "lowest");
  assert.equal(merged.join(""), "lowest");
  assert.ok(merged.length < chars.length);
  assert.deepEqual(merges.map((m) => m.rank), merges.map((_, i) => i));
});

// --- offline shell + navigation -------------------------------------------

console.log("\noffline pages");

const pagePaths = [
  "index.html",
  "attention/index.html",
  "tokenizer/index.html",
  "positional/index.html",
  "sampling/index.html",
];

test("every page has a mobile viewport and all detail pages link home", () => {
  pagePaths.forEach((path) => {
    const html = readFileSync(join(ROOT, "interactive", path), "utf8");
    assert.match(html, /<meta name="viewport"/, `${path} has no viewport meta`);
    if (path !== "index.html") {
      assert.match(html, /class="home-link" href="\.\.\/index\.html"/,
        `${path} has no consistent home link`);
    }
  });
});

test("all script and stylesheet dependencies are local and present", () => {
  pagePaths.forEach((path) => {
    const fullPath = join(ROOT, "interactive", path);
    const html = readFileSync(fullPath, "utf8");
    const refs = [
      ...html.matchAll(/<script[^>]+src="([^"]+)"/g),
      ...html.matchAll(/<link[^>]+href="([^"]+)"/g),
    ].map((m) => m[1]);
    refs.forEach((ref) => {
      assert.ok(!/^(?:https?:)?\/\//.test(ref), `${path} loads network resource ${ref}`);
      assert.ok(existsSync(resolve(dirname(fullPath), ref)), `${path} is missing ${ref}`);
    });
    assert.ok(!/\bfetch\s*\(/.test(html), `${path} uses fetch(), which breaks file://`);
  });
});

test("narrow-screen CSS stacks controls instead of widening the page", () => {
  const css = readFileSync(join(ROOT, "interactive/shared/learnai.css"), "utf8");
  const home = readFileSync(join(ROOT, "interactive/index.html"), "utf8");
  assert.match(css, /@media \(max-width: 520px\)[\s\S]*\.controls > \.control/);
  assert.match(home, /minmax\(min\(100%, 310px\), 1fr\)/);
});

// --- summary ---------------------------------------------------------------

console.log(`\n${passed} passed, ${failures.length} failed`);
if (failures.length) {
  console.log("\nfailures:");
  failures.forEach(([n, m]) => console.log(`  ${n}: ${m}`));
  process.exit(1);
}
