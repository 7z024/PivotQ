"""统一定义量子特征组件的公共 API。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from .contracts import QuantumFeatureRequest, QuantumFeatureResponse


class QuantumFeatureAPI(ABC):
    """定义量子后端或性能模拟器必须实现的稳定边界。"""

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """返回后端能力但不执行量子工作负载。"""

    @abstractmethod
    def extract_features(self, request: QuantumFeatureRequest) -> QuantumFeatureResponse:
        """构造并执行请求，按原始样本顺序返回特征。"""
