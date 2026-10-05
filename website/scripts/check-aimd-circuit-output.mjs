import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { readFileSync } from 'node:fs';

const page = readFileSync(new URL('../src/pages/examples/aimd.astro', import.meta.url), 'utf8');
assert.ok(page.includes("import circuitSummarySource from '../../lib/aimd-circuit-summary.py?raw'"), 'The page imports the displayed circuit summary code');
assert.ok(page.includes("import circuitOutputText from '../../lib/aimd-circuit-output.txt?raw'"), 'The page imports the matching output');
assert.ok(page.includes('${circuitSummarySource.trim()}'), 'Code cell 2 contains the executable summary');

const model = readFileSync(new URL('../src/lib/aimd-model-source.py', import.meta.url), 'utf8');
const summary = readFileSync(new URL('../src/lib/aimd-circuit-summary.py', import.meta.url), 'utf8');
const expected = readFileSync(new URL('../src/lib/aimd-circuit-output.txt', import.meta.url), 'utf8');
const helperStart = model.indexOf('def _parameter(');
const buildStart = model.indexOf('def build_circuit()');
const saveStart = model.indexOf('\ndef _save_png');
assert.ok(helperStart >= 0 && buildStart > helperStart && saveStart > buildStart);

// Execute the same two visible cells rather than importing an easier, separate path.
const notebook = `import math
from uuid import NAMESPACE_URL, uuid5
from pivotq import QuantumCircuit, Parameter
ADAPT_OPERATORS = ("IYZ", "YII", "YZI", "IIX", "YII")

${model.slice(helperStart, buildStart).trim()}

${model.slice(buildStart, saveStart).trim()}
circuit = build_circuit()
${summary.trim()}
`;
const python = process.env.AIMD_QISKIT_PYTHON || 'python3';
const actual = execFileSync(python, ['-c', notebook], { encoding: 'utf8' });
assert.equal(actual, expected, 'The displayed output must match the displayed PivotQ cells');
assert.match(actual, /量子比特: 3/);
assert.match(actual, /未赋值参数: 14/);
assert.match(actual, /基础门总数: 70/);
console.log('PASS: AIMD visible PivotQ cells reproduce the displayed circuit summary.');
