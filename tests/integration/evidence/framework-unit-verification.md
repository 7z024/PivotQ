# Framework unit verification — transcribed tool output

Evidence type: `transcribed_tool_output`.

This document transcribes the commands and summaries observed in the Codex tool
outputs on 2026-10-02. The two runs below were not redirected to files; this is
not an original pytest log. The baseline temporary directory was automatically
deleted when the verification script exited. No test was rerun to create this
document. Subsequent raw logs supersede the current-worktree snapshot below.

## Isolated HEAD baseline

Repository: `/opt/data/private/zhn/launch-event/qhai-2026`.

Baseline HEAD: `34b8edd38831ac3b33964144e1352ea616b8b1f3`.
The repository reflog confirms HEAD did not change between these tests and this
record. The baseline uses committed framework source and tests, with the same
unified Python environment used for the worktree tests.

Command executed from the repository root:

```python
import io, os, pathlib, subprocess, tarfile, tempfile
repo = pathlib.Path('/opt/data/private/zhn/launch-event/qhai-2026')
python = repo / '.venv/bin/python'
archive = subprocess.check_output(
    ['git', 'archive', '--format=tar', 'HEAD', 'packages/qhai-framework'], cwd=repo,
)
with tempfile.TemporaryDirectory(prefix='qhai-framework-baseline-') as staging:
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        bundle.extractall(staging, filter='data')
    package = pathlib.Path(staging) / 'packages/qhai-framework'
    env = dict(os.environ, PYTHONPATH=str(package), PYTHONDONTWRITEBYTECODE='1')
    subprocess.run(
        [str(python), '-c', 'import ray_quantum; print("Baseline module:", ray_quantum.__file__)'],
        cwd=package, env=env, check=True,
    )
    result = subprocess.run(
        [str(python), '-m', 'pytest', 'tests/unit', '-q', '--tb=no'], cwd=package, env=env,
    )
    print('Baseline exit code:', result.returncode)
```

Observed output:

```text
Baseline module: /tmp/qhai-framework-baseline-uhk13obf/packages/qhai-framework/ray_quantum/__init__.py
29 failed, 266 passed, 3 skipped, 2 warnings, 140 subtests passed in 4.26s
Baseline exit code: 1
```

## Earlier worktree snapshot

Working directory:
`/opt/data/private/zhn/launch-event/qhai-2026/packages/qhai-framework`.

```bash
/opt/data/private/zhn/launch-event/qhai-2026/.venv/bin/python -m pytest tests/unit -q --tb=no
```

Observed output:

```text
29 failed, 301 passed, 3 skipped, 2 warnings, 140 subtests passed in 4.65s
```

This snapshot precedes the final additions to `test_qpu_simulation.py`; it must
not be described as covering every case in the final 20-case QPU test file.
The later full-suite raw log is the source for the final worktree count.

## Identical failure selectors in both recorded runs

Both runs reported precisely the following 29 failures. They belong to the
historical `qasm_client` SDK tests while the committed production adapter uses
the HTTP client interface. The test fixture injects `qasm_client`, and its old
non-generator `_exchange` assumptions also differ from the current adapter.
The test environment additionally lacks `httpx`; the runs did not establish
working hardware transport.

```text
tests/unit/test_qasm_client_adapter.py::test_upload_wait_preserves_files_and_native_context
tests/unit/test_qasm_client_adapter.py::test_result_decoder_is_explicitly_pending_after_one_exchange
tests/unit/test_qasm_client_adapter.py::test_explicit_config_takes_precedence_over_environment
tests/unit/test_qasm_client_adapter.py::test_missing_config_never_constructs_client[None-QPU_DEVICE_URL]
tests/unit/test_qasm_client_adapter.py::test_missing_config_never_constructs_client[None-QPU_DEVICE_API_KEY]
tests/unit/test_qasm_client_adapter.py::test_missing_config_never_constructs_client[-QPU_DEVICE_URL]
tests/unit/test_qasm_client_adapter.py::test_missing_config_never_constructs_client[-QPU_DEVICE_API_KEY]
tests/unit/test_qasm_client_adapter.py::test_missing_config_never_constructs_client[   -QPU_DEVICE_URL]
tests/unit/test_qasm_client_adapter.py::test_missing_config_never_constructs_client[   -QPU_DEVICE_API_KEY]
tests/unit/test_qasm_client_adapter.py::test_missing_sdk_reports_provider_dependency
tests/unit/test_qasm_client_adapter.py::test_sdk_failures_cleanup_without_retry_or_payload_leak[enter]
tests/unit/test_qasm_client_adapter.py::test_sdk_failures_cleanup_without_retry_or_payload_leak[submit]
tests/unit/test_qasm_client_adapter.py::test_sdk_failures_cleanup_without_retry_or_payload_leak[wait]
tests/unit/test_qasm_client_adapter.py::test_sdk_failures_cleanup_without_retry_or_payload_leak[exit]
tests/unit/test_qasm_client_adapter.py::test_missing_task_id_is_uncertain_and_never_resubmitted[None]
tests/unit/test_qasm_client_adapter.py::test_missing_task_id_is_uncertain_and_never_resubmitted[job1]
tests/unit/test_qasm_client_adapter.py::test_missing_task_id_is_uncertain_and_never_resubmitted[job2]
tests/unit/test_qasm_client_adapter.py::test_missing_task_id_is_uncertain_and_never_resubmitted[job3]
tests/unit/test_qasm_client_adapter.py::test_missing_task_id_is_uncertain_and_never_resubmitted[job4]
tests/unit/test_qasm_client_adapter.py::test_unconfirmed_success_never_reaches_decoder[None]
tests/unit/test_qasm_client_adapter.py::test_unconfirmed_success_never_reaches_decoder[result1]
tests/unit/test_qasm_client_adapter.py::test_unconfirmed_success_never_reaches_decoder[result2]
tests/unit/test_qasm_client_adapter.py::test_unconfirmed_success_never_reaches_decoder[result3]
tests/unit/test_qasm_client_adapter.py::test_suppressed_sdk_exception_still_cannot_become_success
tests/unit/test_qasm_client_adapter.py::test_upload_names_cannot_escape_directory[../escape.qasm]
tests/unit/test_qasm_client_adapter.py::test_upload_names_cannot_escape_directory[..\\escape.qasm]
tests/unit/test_qasm_client_adapter.py::test_upload_names_cannot_escape_directory[D:escape.qasm]
tests/unit/test_qasm_client_adapter.py::test_upload_names_cannot_escape_directory[]
tests/unit/test_qasm_client_adapter.py::test_duplicate_upload_names_fail_before_client
```

The three skips in both runs are the already archived `record_cleanup`
application contract tests in `tests/unit/jobs/test_driver.py`. The two warnings
are existing protobuf extension deprecations for Python 3.14 compatibility.
