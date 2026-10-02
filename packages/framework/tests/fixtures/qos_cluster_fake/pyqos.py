"""Minimal in-process pyqos fake used by the deployable QOS test package."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from copy import deepcopy
from threading import Lock
from typing import Any


_LOCK = Lock()
_DATASETS: dict[str, dict[str, object]] = {}


class DataTree:
    """Return simulated P01 datasets registered by the sibling helper module."""

    def download(self, dataset_ids: Iterable[str]) -> dict[str, dict[str, object]]:
        ids = tuple(dataset_ids)
        if not ids:
            raise ValueError("fake DataTree.download requires at least one dataset ID")
        if any(not isinstance(dataset_id, str) or not dataset_id for dataset_id in ids):
            raise TypeError("fake DataTree dataset IDs must be non-empty strings")

        downloaded: dict[str, dict[str, object]] = {}
        with _LOCK:
            for dataset_id in ids:
                try:
                    dataset = _DATASETS.pop(dataset_id)
                except KeyError:
                    raise KeyError(
                        f"fake DataTree has no dataset {dataset_id!r}"
                    ) from None
                downloaded[dataset_id] = deepcopy(dataset)
        return downloaded


def _store_dataset(dataset_id: str, dataset: Mapping[str, Any]) -> None:
    if not isinstance(dataset_id, str) or not dataset_id:
        raise TypeError("fake dataset_id must be a non-empty string")
    if not isinstance(dataset, Mapping):
        raise TypeError("fake dataset must be a mapping")
    with _LOCK:
        _DATASETS[dataset_id] = deepcopy(dict(dataset))


def _reset() -> None:
    """Clear process-local fake datasets between tests."""

    with _LOCK:
        _DATASETS.clear()


__all__ = ["DataTree"]
