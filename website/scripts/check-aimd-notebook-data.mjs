import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

const script = fileURLToPath(new URL('../src/lib/aimd-trajectory-analysis.py', import.meta.url));
const data = fileURLToPath(new URL('../public/aimd/imported/', import.meta.url));
const result = spawnSync(process.env.PYTHON || 'python3', [script, data], { encoding: 'utf8' });

assert.equal(result.status, 0, result.stderr);
assert.deepEqual(result.stdout.trim().split(/\r?\n/), [
  'step=0 time_fs=0.0',
  '  oh1_A=0.9572 oh2_A=0.9572 hoh_deg=104.52',
  '  total_eV=0.5912',
  'step=1000 time_fs=100.0',
  '  oh1_A=1.2024 oh2_A=1.1422 hoh_deg=95.27',
  '  total_eV=0.4855',
  'delta_total_eV=-0.1056',
]);
const shownOutput = readFileSync(new URL('../src/lib/aimd-trajectory-output.txt', import.meta.url), 'utf8').trim();
assert.equal(result.stdout.trim(), shownOutput, 'The notebook output matches the runnable script');
console.log('AIMD notebook CSV analysis matches the imported first and last frames.');
