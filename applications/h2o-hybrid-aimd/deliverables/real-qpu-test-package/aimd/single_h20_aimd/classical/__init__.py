"""导出独立项目使用的最终经典回归实现。"""

from .torch_mlp import TorchMLPRegressor
from .remote_actor import RemoteClassicalPredictActor

__all__ = ["RemoteClassicalPredictActor", "TorchMLPRegressor"]
