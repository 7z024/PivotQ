"""统一定义经典能量模型的公共 API。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
from typing import Self

from .contracts import (
    ClassicalFitRequest,
    ClassicalFitResponse,
    ClassicalPredictRequest,
    ClassicalPredictResponse,
)


class ClassicalPotentialAPI(ABC):
    """定义可训练经典能量模型必须实现的稳定边界。"""

    @abstractmethod
    def describe(self) -> dict[str, Any]:
        """返回模型能力和当前训练状态。"""

    @abstractmethod
    def fit(self, request: ClassicalFitRequest) -> ClassicalFitResponse:
        """用量子特征和参考标签拟合能量模型。"""

    @abstractmethod
    def predict(self, request: ClassicalPredictRequest) -> ClassicalPredictResponse:
        """为每行输入特征预测一个以 eV 为单位的能量。"""

    @abstractmethod
    def save_checkpoint(self, path: str | Path) -> Path:
        """保存足以恢复模型结构、参数和预处理状态的检查点。"""

    @classmethod
    @abstractmethod
    def load_checkpoint(cls, path: str | Path, **kwargs: Any) -> Self:
        """从检查点恢复一个可直接预测的经典后端实例。"""
