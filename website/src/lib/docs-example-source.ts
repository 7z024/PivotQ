import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';

type ExampleFile = 'hybrid_program.py' | 'system_workflow.py' | 'performance_prediction.py' | 'custom_backend.py';

/** Prefer the executable SDK sources in a repository checkout, with snapshots for standalone site archives. */
export function docsExampleSource(filename: ExampleFile): string {
  const source = resolve(process.cwd(), '../packages/framework/examples', filename);
  if (existsSync(source)) return readFileSync(source, 'utf8');
  return readFileSync(resolve(process.cwd(), 'src/lib/docs-examples', filename), 'utf8');
}
