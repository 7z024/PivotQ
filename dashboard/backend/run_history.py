"""Persist task state atomically; interrupted tasks are never resubmitted."""
import json
from pathlib import Path
import tempfile
import threading
from .models import Run, WorkflowPlan, StagePlan

_WRITE_LOCK = threading.RLock()
_TERMINAL = {'SUCCEEDED', 'FAILED', 'CANCELLED'}


def read_history(path):
    path = Path(path)
    if not path.exists():
        return {}
    result = {}
    for saved in json.loads(path.read_text(encoding='utf-8-sig')):
        item = dict(saved)
        plan = dict(item['plan'])
        plan['stages'] = tuple(StagePlan(**(dict(s) | {'depends_on': tuple(s['depends_on'])})) for s in plan['stages'])
        item['plan'] = WorkflowPlan(**plan)
        run = Run(**item)
        if run.status not in _TERMINAL:
            from .local_execution import recover_interrupted_worker
            reconciliation = recover_interrupted_worker(run)
            old_status = run.status
            run.status = 'FAILED'
            run.result = (run.result or {}) | {'error': '服务重启，任务已中断；请重新运行', 'interrupted': True}
            run.events.append({'type': 'interrupted', 'previous_status': old_status,
                               'reconciliation': reconciliation})
        result[run.id] = run
    return result


def write_history(path, runs):
    path = Path(path)
    with _WRITE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [run.as_dict() for run in list(runs.values())]
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                             prefix=path.name + '.', suffix='.tmp', delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(rows, stream, ensure_ascii=False, indent=2)
                stream.flush()
                import os
                os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            if temporary:
                temporary.unlink(missing_ok=True)
