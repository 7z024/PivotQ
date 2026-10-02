"""横场 Ising 基态制备线路的 Qiskit -> QCIS 一体化工具。

本文件集中提供以下功能：

1. 从 ``qiskit_prep_circ.py`` 安全读取四组门序列，不执行源文件；
2. 根据 ``CHIP_TOPOLOGY`` 自动查找相邻 qubit 的 QOS3 coupler；
3. 生成 QOS3 波形查看器使用的单窗口 RZZ 测时探针；
4. 从波形数据或波形序列数据提取两个 ``B`` 之间的真实时长；
5. 按有限脉宽 CPMG 公式给空闲 qubit 插入动态解耦；
6. 生成四个无注释、可由 QOS3 识别的 ``.qcis`` 文件；
7. 把 transpile 后的 QuantumCircuit 或单线路 QPY 转成同步分层的 QCIS；
8. 通过 ``RunCircuits`` 执行 QCIS，并支持逐 shot 或 shots 平均数据。

推荐使用流程
------------

重要：正式实验应优先使用 QOS3 对无 DD 测时探针的实际编译结果来确定
DD 窗口。QOS3 的结果已经包含标定门波形、硬件 padding/buffer 和组件时间线
对齐，不能用逻辑门数量或标称门时长代替。``dd_total_duration_ns`` 只是在
暂时无法取得 QOS3 导出数据时的手工回退，不是正式实验的首选输入。

第一步：生成无 DD 测时探针并在 QOS3 波形查看器中编译::

    probe_path = save_qos3_dd_timing_probe([99, 106, 100, 107])

探针只有一个 Q106-Q100 RZZ 子层。查看器导出的波形序列时间默认按 ns
解释；完整波形数组长度始终按 DAC samples 解释。

第二步：把查看器导出的 Python 对象直接交给转换函数。若有完整波形数据::

    paths = convert_qiskit_file_to_qcis(
        source_file="qiskit_prep_circ.py",
        qubit_ids=[99, 106, 100, 107],
        add_dynamic_decoupling=True,
        qos3_waveform_data=waveform_data,
    )

若导出的是波形序列数据，其记录应包含 start/tStart/start_sample 之一，
以及 length/duration/num_samples 或 end/stop 之一::

    paths = convert_qiskit_file_to_qcis(
        source_file="qiskit_prep_circ.py",
        qubit_ids=[99, 106, 100, 107],
        add_dynamic_decoupling=True,
        qos3_waveform_sequence_data=waveform_sequence_data,
        qos3_sequence_time_unit="ns",
    )

如果暂时只能从查看器手工读取总时长，仍可直接输入 ns::

    paths = convert_qiskit_file_to_qcis(
        source_file="qiskit_prep_circ.py",
        qubit_ids=[99, 106, 100, 107],
        add_dynamic_decoupling=True,
        dd_total_duration_ns=200.0,
    )

手工时长和 QOS3 导出对象不能同时提供。两类导出对象同时提供时，其结果
必须在一个 DAC sample 内一致，否则停止生成，避免错误 DD 时序。

第三步：在安装了 pyqos 的 QOS3 环境执行生成文件::

    runner = run_qcis_files_with_pyqos(
        paths,
        [99, 106, 100, 107],
        readout_mode="0_other",
        data_type="EVENT_STATE",
        sampling_interval=sampling_interval,
        num_shots=num_shots,
    )

``readout_mode="01"`` 使用 ``B -> M``，用于 0/1 判别；``"0_other"`` 按
QCIS 手册使用 ``B -> X12 -> M``，用于 0/非零态判别。``EVENT_STATE`` 和
``EVENT_IQ`` 保留逐 shot 数据；``P01`` 和 ``IQ`` 返回所有 shots 的平均结果。
``readout_mode`` 控制实际测量线路，``data_type`` 控制返回数据形式，二者彼此
独立。真机 ``RunCircuits`` 工作流必须保持 ``add_measurements=False``（默认值，
CLI 中不要使用 ``--measure``），由运行函数在提交前统一加入唯一的读取层。

已有 transpile 线路的转换
--------------------------

如果上层已经得到 ``qiskit.transpile()`` 返回的 ``QuantumCircuit``，可自行选择
符合实验和标定要求的门集，再调用独立入口。例如::

    transpiled_qc = transpile(
        qc,
        basis_gates=["rz", "sx", "x", "cz"],
        coupling_map=confirmed_qos3_coupling_map,
        optimization_level=3,
    )
    path = convert_transpiled_qiskit_to_qcis(
        transpiled_qc,
        [99, 106, 100, 107],
        "qcis_preparation_circuits/transpiled_preparation_N4.qcis",
    )

也可以把 ``source`` 换成一个 ``.qpy`` 路径，但该文件必须恰好保存一条线路。
``qubit_ids[i]`` 始终对应 ``transpiled_qc.qubits[i]``；转换器不会解释 Qiskit
``layout``，因此实验人员必须显式给出正确的 QOS3 注册表编号。转换器直接支持
``rx/ry/rz``、QOS3 半转门、CZ、delay、局部或全局 barrier，并把常见 Qiskit
单比特门等价降低为这些指令。未知双比特门、reset 和经典控制仍应在外部
transpile 阶段处理；转换器不会猜测未确认的硬件语义。

这个新入口默认生成无 ``M`` 的制备文件，读取通常仍由
``run_qcis_files_with_pyqos`` 按 ``readout_mode`` 追加；独立使用时也可显式设置
``add_measurements=True`` 保留或追加终端测量。通用 transpiled layer 的真实
QOS3 波形窗口尚未建立时长合同，因此此入口不插入 DD，不能把旧 RZZ 子层的
DD 时长直接套到单个 CZ layer 上。

注意：当前没有 QOS3 实际导出样例，解析器支持常见的嵌套字典、普通对象和
数值数组结构。若实际字段不同，应提供最小导出样例后扩展字段适配，不能猜测。
"""

from __future__ import annotations

import argparse
import ast
import math
import os
import tempfile
from collections.abc import Mapping as MappingABC
from collections.abc import Sequence as SequenceABC
from numbers import Number, Real
from pathlib import Path
from typing import Any, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple, Union


GateSequence = Tuple[Tuple[str, float], ...]
Edge = Tuple[int, int]
QubitId = Union[int, str]

SOURCE_PATH = Path(__file__).with_name("qiskit_prep_circ.py")
DEFAULT_QUBITS = ("Q099", "Q106", "Q100", "Q107")#注意：比特顺序代表 Ising 链顺序

# 这里逐行抄录用户提供的 QOS3 注册表拓扑图。图中没有频率数值只表示该次
# 绘图缺少对应标定值，不能据此删除已经显示的硬件对象，例如 Q001、Q002。
# 公开论文的 105/182 编号与这份 QOS3 内部注册表编号不同，不能混用。
ZUCHONGZHI3_QOS3_QUBIT_ROWS: Tuple[Tuple[int, ...], ...] = (
    (1, 2, 3, 4, 5, 6, 7),
    (8, 9, 10, 11, 12, 13, 14),
    (15, 16, 17, 18, 19, 20, 21, 22),
    (23, 24, 25, 26, 27, 28, 29),
    (30, 31, 32, 33, 34, 35, 36),
    (37, 38, 39, 40, 41, 42, 43),
    (44, 45, 46, 47, 48, 49, 50, 51),
    (52, 53, 54, 55, 56, 57, 58),
    (59, 60, 61, 62, 63, 64, 65),
    (66, 67, 68, 69, 70, 71, 72),
    (73, 74, 75, 76, 77, 78, 79, 80),
    (81, 82, 83, 84, 85, 86, 87),
    (88, 89, 90, 91, 92, 93, 94),
    (95, 96, 97, 98, 99, 100, 101),
    (102, 103, 104, 105, 106, 107),
)

# 每个数值是该图片行首个 qubit 的横坐标；同一行相邻 qubit 的坐标间隔为 2。
# 最后一行从坐标 2 开始，对应图片中 Q102 下方没有更靠左的硬件对象。
ZUCHONGZHI3_QOS3_ROW_OFFSETS: Tuple[int, ...] = (
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    2,
)


def build_chip_topology(
    qubit_rows: Sequence[Sequence[int]],
    row_offsets: Sequence[int],
) -> Dict[Edge, str]:
    """根据图片中的交错行坐标生成 QOS3 无向最近邻拓扑。

    一行中第 ``column`` 个比特的横坐标为
    ``row_offsets[row] + 2 * column``。只有位于相邻两行且横坐标相差 1
    的两个比特才构成图片中的最近邻。字典键始终按比特数字编号升序
    保存，值采用 QOS3 注册表的 ``G{小编号}{大编号}`` 命名方式。

    Args:
        qubit_rows: 按图片可视行排列的正整数比特编号；各行可以不等宽，
            但所有编号必须唯一。
        row_offsets: 每行第一个比特的非负整数横坐标，数量必须与行数相同。

    Returns:
        从无方向物理边 ``(小编号, 大编号)`` 到 QOS3 coupler 名称的映射。

    Raises:
        TypeError: 任一比特编号或行偏移不是整数，或误传了布尔值。
        ValueError: 布局为空、包含空行、偏移数量不匹配、编号非正或重复，
            或行偏移为负数。
    """
    rows = tuple(tuple(row) for row in qubit_rows)
    offsets = tuple(row_offsets)
    if not rows or any(not row for row in rows):
        raise ValueError("Chip layout must contain non-empty qubit rows")
    if len(offsets) != len(rows):
        raise ValueError("Chip layout requires exactly one offset per row")
    for offset in offsets:
        if isinstance(offset, bool) or not isinstance(offset, int):
            raise TypeError("Chip-layout row offsets must be integers")
        if offset < 0:
            raise ValueError("Chip-layout row offsets cannot be negative")

    qubit_ids = tuple(qubit_id for row in rows for qubit_id in row)
    for qubit_id in qubit_ids:
        if isinstance(qubit_id, bool) or not isinstance(qubit_id, int):
            raise TypeError("Chip-layout qubit identifiers must be integers")
        if qubit_id <= 0:
            raise ValueError("Chip-layout qubit identifiers must be positive")
    if len(set(qubit_ids)) != len(qubit_ids):
        raise ValueError("Chip-layout qubit identifiers must be unique")

    topology: Dict[Edge, str] = {}
    for row_index, upper_row in enumerate(rows[:-1]):
        lower_row = rows[row_index + 1]
        lower_offset = offsets[row_index + 1]
        lower_by_x = {
            lower_offset + 2 * column: qubit_id
            for column, qubit_id in enumerate(lower_row)
        }

        # 只检查下一行的左下、右下两个格点，避免同一行误生成耦合器。
        upper_offset = offsets[row_index]
        for column, qubit0 in enumerate(upper_row):
            upper_x = upper_offset + 2 * column
            for lower_x in (upper_x - 1, upper_x + 1):
                if lower_x not in lower_by_x:
                    continue
                qubit1 = lower_by_x[lower_x]
                edge: Edge = (min(qubit0, qubit1), max(qubit0, qubit1))
                topology[edge] = f"G{edge[0]}{edge[1]}"
    return topology


# 该图包含 Q001...Q107 和 187 条几何最近邻边。CHIP_TOPOLOGY 继续作为
# 转换器的默认全局查找表，保持 infer_couplers 及上层调用接口不变。
CHIP_TOPOLOGY = build_chip_topology(
    ZUCHONGZHI3_QOS3_QUBIT_ROWS,
    ZUCHONGZHI3_QOS3_ROW_OFFSETS,
)
SUPPORTED_QOS3_DATA_TYPES = ("EVENT_STATE", "EVENT_IQ", "P01", "IQ")
SUPPORTED_QOS3_READOUT_MODES = ("01", "0_other")


class DdTiming(NamedTuple):
    """Integer-sample timing for a symmetric two-pulse CPMG idle window."""

    total_samples: int
    x2_samples: int
    edge_delay_samples: int
    middle_delay_samples: int


class _TranspiledInstruction(NamedTuple):
    """保存一条已从 Qiskit DAG 规范化的基础门。"""

    name: str
    qubit_indices: Tuple[int, ...]
    parameters: Tuple[float, ...]
    clbit_indices: Tuple[int, ...] = ()
    time_unit: Optional[str] = None


# 这些名称描述转换器接受的 Qiskit 指令，而不是 QOS3 硬件门集的上限。
# 除原生旋转和半转门外，常见单比特门会等价降低为 RX/RY/RZ；全局相位丢弃。
_QOS3_FIXED_SINGLE_QUBIT_GATES: Mapping[str, str] = {
    "x": "X",
    "y": "Y",
    "sx": "X2P",
    "sxdg": "X2M",
    # 允许调用方用 Qiskit 自定义 Gate 保留 QCIS 原生名称。
    "x2p": "X2P",
    "x2m": "X2M",
    "y2p": "Y2P",
    "y2m": "Y2M",
    "x12": "X12",
}
_QOS3_ROTATION_GATES: Mapping[str, str] = {
    "rx": "RX",
    "ry": "RY",
    "rz": "RZ",
}
_QOS3_FIXED_Z_ROTATIONS: Mapping[str, float] = {
    "z": math.pi,
    "s": math.pi / 2,
    "sdg": -math.pi / 2,
    "t": math.pi / 4,
    "tdg": -math.pi / 4,
}
_QOS3_SINGLE_QUBIT_PARAMETER_COUNTS: Mapping[str, int] = {
    **{name: 0 for name in _QOS3_FIXED_SINGLE_QUBIT_GATES},
    **{name: 1 for name in _QOS3_ROTATION_GATES},
    **{name: 0 for name in _QOS3_FIXED_Z_ROTATIONS},
    "h": 0,
    "id": 0,
    "p": 1,
    "u1": 1,
    "u2": 2,
    "u": 3,
    "u3": 3,
    "r": 2,
}


def load_gate_sequences(source_path: Path = SOURCE_PATH) -> Dict[int, GateSequence]:
    """Read the four literal gate lists without importing or executing Qiskit."""
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    named: Dict[str, GateSequence] = {}
    for node in tree.body:
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        if not node.target.id.startswith("gate_sequence") or node.value is None:
            continue
        # literal_eval 只读取字面量，不执行源文件顶层代码，也不需要导入 Qiskit。
        value = ast.literal_eval(node.value)
        named[node.target.id] = tuple((str(name), float(angle)) for name, angle in value)

    sequences = {index: named[f"gate_sequence{index}"] for index in range(1, 5)}
    return sequences


def normalize_qubit_ids(qubit_ids: Sequence[QubitId]) -> Tuple[str, ...]:
    """Validate QOS3 names while preserving zero padding supplied as text.

    字符串只统一为大写 ``Q`` 前缀，不改变数字部分的宽度，因此 ``Q099``、
    ``q099`` 和 ``099`` 均得到 ``Q099``。整数没有文本宽度信息，仍按普通十进制
    输出，例如 ``99`` 得到 ``Q99``。唯一性按数值编号检查，所以 ``Q099`` 与
    ``Q99`` 不能在同一组映射中同时出现。
    """
    normalized: List[str] = []
    numeric_ids: List[int] = []
    for value in qubit_ids:
        if isinstance(value, bool) or not isinstance(value, (int, str)):
            raise TypeError("Qubit identifiers must be integers or strings")
        suffix = str(value).strip()
        if suffix[:1].upper() == "Q":
            suffix = suffix[1:]
        if not suffix.isdigit():
            raise ValueError(f"Invalid qubit identifier: {value!r}")
        numeric_ids.append(int(suffix))
        normalized.append(f"Q{suffix if isinstance(value, str) else int(suffix)}")

    if len(normalized) < 2:
        raise ValueError("At least two qubits are required")
    if len(set(numeric_ids)) != len(numeric_ids):
        raise ValueError("Qubit identifiers must be unique")
    return tuple(normalized)


def _coupler_name_for_edge(
    edge: Edge,
    qubits: Sequence[str],
    registered_name: str,
) -> str:
    """Apply endpoint zero padding to automatically derived coupler names.

    ``CHIP_TOPOLOGY`` 的默认名字由整数编号拼接得到。若调用方提供了带前导零的
    qubit 名称，则用同一文本后缀重建名字；人工配置的非默认注册表名称不改写。
    """
    default_name = f"G{edge[0]}{edge[1]}"
    if registered_name != default_name:
        return registered_name
    suffix_by_number = {int(qubit[1:]): qubit[1:] for qubit in qubits}
    try:
        return f"G{suffix_by_number[edge[0]]}{suffix_by_number[edge[1]]}"
    except KeyError:
        return registered_name


def infer_couplers(
    qubits: Sequence[str],
    topology: Mapping[Tuple[int, int], str] = CHIP_TOPOLOGY,
) -> Tuple[str, ...]:
    """Look up adjacent couplers while preserving endpoint zero padding."""
    numbers = [int(qubit[1:]) for qubit in qubits]
    pairs = [tuple(sorted(pair)) for pair in zip(numbers, numbers[1:])]
    try:
        return tuple(
            _coupler_name_for_edge(pair, qubits, topology[pair])
            for pair in pairs
        )
    except KeyError as exc:
        raise ValueError(
            f"Qubit pair {exc.args[0]} is not present in CHIP_TOPOLOGY"
        ) from exc


def make_dd_timing(
    total_duration_ns: float,
    sample_rate_hz: float = 2e9,
    x2_duration_ns: float = 40.0,
) -> DdTiming:
    """Convert finite-pulse symmetric CPMG timing from nanoseconds to samples.

    The two physical X-pulse centers are placed at one quarter and three
    quarters of the idle window. Pulse widths are included in the timing, so
    the free-evolution delays are ``T/4 - t_X/2`` and ``T/2 - t_X``.
    """
    values = {
        "total_duration_ns": total_duration_ns,
        "sample_rate_hz": sample_rate_hz,
        "x2_duration_ns": x2_duration_ns,
    }
    for name, value in values.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a real number")
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")

    samples_per_ns = sample_rate_hz / 1e9
    raw_total_samples = total_duration_ns * samples_per_ns
    raw_x2_samples = x2_duration_ns * samples_per_ns
    total_samples = round(raw_total_samples)
    x2_samples = round(raw_x2_samples)
    if not math.isclose(raw_total_samples, total_samples, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("DD total duration must map to an integer number of DAC samples")
    if not math.isclose(raw_x2_samples, x2_samples, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("X/2 duration must map to an integer number of DAC samples")

    # 实验约定一个物理 X 门由两个连续的已标定 X/2 脉冲组成。
    x_samples = 2 * x2_samples
    if total_samples < 2 * x_samples:
        raise ValueError("DD total duration must accommodate two physical X pulses")
    if total_samples % 4:
        raise ValueError("DD total duration must preserve quarter-window sample symmetry")

    # 有限脉宽修正保证两个 X 门的中心（而不是前沿）位于 T/4 和 3T/4。
    edge_delay_samples = total_samples // 4 - x_samples // 2
    middle_delay_samples = total_samples // 2 - x_samples
    if edge_delay_samples < 0 or middle_delay_samples < 0:
        raise ValueError("DD delays cannot be negative")
    if 2 * edge_delay_samples + middle_delay_samples + 2 * x_samples != total_samples:
        raise ValueError("DD timing does not fill the requested idle window")
    return DdTiming(
        total_samples=total_samples,
        x2_samples=x2_samples,
        edge_delay_samples=edge_delay_samples,
        middle_delay_samples=middle_delay_samples,
    )


def _object_fields(value: Any) -> Optional[MappingABC]:
    """将字典或普通对象统一转换为可递归检查的字段映射。"""
    if isinstance(value, MappingABC):
        return value
    fields = getattr(value, "__dict__", None)
    return fields if isinstance(fields, MappingABC) else None


def _waveform_leaf_lengths(value: Any) -> List[int]:
    """递归收集 QOS3 波形数据中所有数值采样数组的长度。"""
    fields = _object_fields(value)
    if fields is not None:
        lengths: List[int] = []
        for child in fields.values():
            lengths.extend(_waveform_leaf_lengths(child))
        return lengths

    if isinstance(value, (str, bytes, bytearray)):
        return []

    shape = getattr(value, "shape", None)
    if shape and all(isinstance(size, int) for size in shape):
        return [int(shape[-1])] if shape[-1] > 0 else []

    if isinstance(value, SequenceABC):
        if value and all(
            isinstance(sample, Number) and not isinstance(sample, bool)
            for sample in value
        ):
            return [len(value)]
        lengths = []
        for child in value:
            lengths.extend(_waveform_leaf_lengths(child))
        return lengths
    return []


def _normalized_record_fields(fields: MappingABC) -> Dict[str, Any]:
    """将序列记录字段名规范化为小写无分隔符形式。"""
    return {
        "".join(char for char in str(key).lower() if char.isalnum()): value
        for key, value in fields.items()
    }


def _finite_record_number(value: Any, field_name: str) -> float:
    """校验并返回波形序列记录中的有限实数。"""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"QOS3 sequence field {field_name!r} must be a real number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"QOS3 sequence field {field_name!r} must be finite")
    return number


def _sequence_intervals(value: Any) -> List[Tuple[float, float]]:
    """递归提取波形序列记录的起止位置。"""
    fields = _object_fields(value)
    intervals: List[Tuple[float, float]] = []
    if fields is not None:
        normalized = _normalized_record_fields(fields)
        start_aliases = ("start", "tstart", "startsample", "offset", "begin")
        length_aliases = ("length", "duration", "numsamples", "nsamples", "size")
        end_aliases = ("end", "stop", "endsample")
        start_key = next((key for key in start_aliases if key in normalized), None)
        length_key = next((key for key in length_aliases if key in normalized), None)
        end_key = next((key for key in end_aliases if key in normalized), None)

        if start_key is not None and (length_key is not None or end_key is not None):
            start = _finite_record_number(normalized[start_key], start_key)
            if length_key is not None:
                length = _finite_record_number(normalized[length_key], length_key)
                if length < 0:
                    raise ValueError("QOS3 sequence length cannot be negative")
                end = start + length
            else:
                end = _finite_record_number(normalized[end_key], end_key)  # type: ignore[index]
                if end < start:
                    raise ValueError("QOS3 sequence end cannot precede start")
            intervals.append((start, end))

        for child in fields.values():
            intervals.extend(_sequence_intervals(child))
        return intervals

    if isinstance(value, SequenceABC) and not isinstance(
        value, (str, bytes, bytearray)
    ):
        for child in value:
            intervals.extend(_sequence_intervals(child))
    return intervals


def extract_qos3_window_ns(
    waveform_data: Any = None,
    waveform_sequence_data: Any = None,
    *,
    sequence_time_unit: str = "ns",
    sample_rate_hz: float = 2e9,
) -> float:
    """从 QOS3 波形数据或波形序列数据提取探针窗口总时长（ns）。

    波形数据取最长数值采样数组；波形序列数据取所有识别记录的最早
    起点到最晚终点。两类数据同时提供时必须在一个 DAC sample 内一致。
    """
    if isinstance(sample_rate_hz, bool) or not isinstance(sample_rate_hz, (int, float)):
        raise TypeError("sample_rate_hz must be a real number")
    if not math.isfinite(sample_rate_hz) or sample_rate_hz <= 0:
        raise ValueError("sample_rate_hz must be finite and positive")
    sample_period_ns = 1e9 / sample_rate_hz

    waveform_duration_ns: Optional[float] = None
    if waveform_data is not None:
        lengths = _waveform_leaf_lengths(waveform_data)
        if not lengths:
            raise ValueError("QOS3 waveform data does not contain numeric sample arrays")
        waveform_duration_ns = max(lengths) * sample_period_ns

    sequence_duration_ns: Optional[float] = None
    if waveform_sequence_data is not None:
        intervals = _sequence_intervals(waveform_sequence_data)
        if not intervals:
            raise ValueError("QOS3 waveform sequence data has no recognized timing records")
        span = max(end for _, end in intervals) - min(start for start, _ in intervals)
        if span <= 0:
            raise ValueError("QOS3 waveform sequence window must be positive")
        unit = sequence_time_unit.strip().lower()
        if unit == "samples":
            sequence_duration_ns = span * sample_period_ns
        elif unit == "ns":
            sequence_duration_ns = span
        else:
            raise ValueError("sequence_time_unit must be 'samples' or 'ns'")

    if waveform_duration_ns is None and sequence_duration_ns is None:
        raise ValueError("Provide QOS3 waveform data or waveform sequence data")
    if waveform_duration_ns is not None and sequence_duration_ns is not None:
        if not math.isclose(
            waveform_duration_ns,
            sequence_duration_ns,
            rel_tol=0.0,
            abs_tol=sample_period_ns,
        ):
            raise ValueError(
                "QOS3 waveform data and waveform sequence data durations disagree: "
                f"{waveform_duration_ns} ns versus {sequence_duration_ns} ns"
            )
        return waveform_duration_ns
    return (
        waveform_duration_ns
        if waveform_duration_ns is not None
        else sequence_duration_ns  # type: ignore[return-value]
    )


def validate_hardware_names(qubits: Sequence[str], couplers: Sequence[str]) -> None:
    """Validate the ordered QOS3 qubit chain and its nearest-neighbour couplers."""
    if len(qubits) < 2:
        raise ValueError("At least two qubits are required")
    if len(couplers) != len(qubits) - 1:
        raise ValueError(f"{len(qubits)} qubits require {len(qubits) - 1} couplers")
    if len(set(qubits)) != len(qubits) or len(set(couplers)) != len(couplers):
        raise ValueError("Qubit and coupler names must be unique")
    if any(not name or any(char.isspace() for char in name) for name in (*qubits, *couplers)):
        raise ValueError("QOS3 component names must be non-empty and contain no whitespace")
    for left, (q0, q1) in enumerate(zip(qubits, qubits[1:])):
        suffix0, suffix1 = q0[1:], q1[1:]
        coupler_suffix = couplers[left][1:]
        if suffix0.isdigit() and suffix1.isdigit() and coupler_suffix.isdigit():
            expected = {suffix0 + suffix1, suffix1 + suffix0}
            if coupler_suffix not in expected:
                raise ValueError(
                    f"Coupler {couplers[left]} does not connect adjacent chain pair {q0}-{q1}; "
                    f"expected a registry name ending in {suffix0 + suffix1} or {suffix1 + suffix0}"
                )


class QcisCompiler:
    """Compile the source gate layers to chronological QOS3 QCIS instructions."""

    def __init__(
        self,
        qubits: Sequence[str],
        couplers: Sequence[str],
        dd_timing: Optional[DdTiming] = None,
    ) -> None:
        """Initialize the compiler with an ordered chain and optional DD timing."""
        validate_hardware_names(qubits, couplers)
        self.qubits = tuple(qubits)
        self.couplers = tuple(couplers)
        self.dd_timing = dd_timing
        self.lines: List[str] = []

    def align(self) -> None:
        """Align all qubit and coupler timelines at a source-circuit barrier."""
        self.lines.append("B " + " ".join((*self.qubits, *self.couplers)))

    def rz(self, qubit: int, angle: float) -> None:
        """Append a QOS3 virtual-Z rotation in radians."""
        self.lines.append(f"RZ {self.qubits[qubit]} {angle:.12g}")

    def ry(self, qubit: int, angle: float) -> None:
        """Append a QOS3 Y-axis rotation in radians."""
        self.lines.append(f"RY {self.qubits[qubit]} {angle:.12g}")

    def h(self, qubit: int) -> None:
        """Implement Qiskit H as RZ(pi) then RY(pi/2), up to global phase."""
        self.rz(qubit, math.pi)
        self.ry(qubit, math.pi / 2)

    def h_all(self) -> None:
        """Apply the H decomposition independently to every chain qubit."""
        for qubit in range(len(self.qubits)):
            self.h(qubit)

    def rx_all(self, angle: float) -> None:
        """Apply the same QOS3 RX rotation to every chain qubit."""
        for qubit in self.qubits:
            self.lines.append(f"RX {qubit} {angle:.12g}")

    def cpmg(self, qubit: int) -> None:
        """Fill one idle window with symmetric CPMG using two physical X gates."""
        if self.dd_timing is None:
            raise ValueError("CPMG requires configured DD timing")
        name = self.qubits[qubit]
        edge = self.dd_timing.edge_delay_samples
        middle = self.dd_timing.middle_delay_samples
        if edge:
            self.lines.append(f"I {name} {edge}")
        self.lines.extend((f"X2P {name}", f"X2P {name}"))
        if middle:
            self.lines.append(f"I {name} {middle}")
        self.lines.extend((f"X2P {name}", f"X2P {name}"))
        if edge:
            self.lines.append(f"I {name} {edge}")

    def cz(self, edge: Edge) -> None:
        """Apply CZ through the coupler corresponding to an adjacent chain edge."""
        left, right = edge
        if right != left + 1:
            raise ValueError(f"Only adjacent edges are supported, got {edge}")
        self.lines.append(f"CZ {self.couplers[left]}")

    def rzz(self, angle: float, edge: Edge) -> None:
        """Implement RZZ(angle) using H-CZ-H-RZ-H-CZ-H on the target."""
        control, target = edge
        self.h(target)
        self.cz((control, target))
        self.h(target)
        self.rz(target, angle)
        self.h(target)
        self.cz((control, target))
        self.h(target)

    def rzz_sublayer(self, angle: float, edges: Sequence[Edge]) -> None:
        """Compile disjoint RZZ edges and decouple qubits idle in this sublayer."""
        if not edges:
            return

        # QOS3 为不同组件建立独立时间线。子层首尾的 B 保证活动 RZZ 轨道与
        # 空闲 qubit 的 CPMG 轨道共享同一个并行时间窗口。
        self.align()
        active_qubits = {qubit for edge in edges for qubit in edge}
        for edge in edges:
            self.rzz(angle, edge)
        if self.dd_timing is not None:
            for qubit in range(len(self.qubits)):
                if qubit not in active_qubits:
                    self.cpmg(qubit)
        self.align()

    def zz_layer(self, angle: float) -> None:
        """Compile the source circuit's even-bond then odd-bond RZZ layer."""
        even_edges = tuple(
            (left, left + 1) for left in range(0, len(self.qubits) - 1, 2)
        )
        odd_edges = tuple(
            (left, left + 1) for left in range(1, len(self.qubits) - 1, 2)
        )
        self.rzz_sublayer(angle, even_edges)
        self.rzz_sublayer(angle, odd_edges)

    def xx_layer(self, angle: float) -> None:
        """Compile RXX by conjugating the complete RZZ layer with H on all qubits."""
        self.h_all()
        self.zz_layer(angle)
        self.h_all()

    def yy_layer(self, angle: float) -> None:
        """Compile the exact S-H-RZZ-H-Sdg decomposition used by the source."""
        for qubit in range(len(self.qubits)):
            self.rz(qubit, math.pi / 2)
            self.h(qubit)
        self.zz_layer(angle)
        for qubit in range(len(self.qubits)):
            self.h(qubit)
            self.rz(qubit, -math.pi / 2)

    def compile_sequence(self, sequence: GateSequence, add_barriers: bool = True) -> None:
        """Append every abstract source layer and optionally preserve its barrier."""
        for gate_name, angle in sequence:
            if gate_name == "H":
                self.h_all()
            elif gate_name == "Rx":
                self.rx_all(angle)
            elif gate_name == "Rzz":
                self.zz_layer(angle)
            elif gate_name == "Rxx":
                self.xx_layer(angle)
            elif gate_name == "Ryy":
                self.yy_layer(angle)
            else:
                raise ValueError(f"Unsupported source gate: {gate_name}")
            if add_barriers:
                self.align()

    def add_measurements(self) -> None:
        """Append simultaneous native-Z readout after aligning preparation timelines."""
        self.align()
        self.lines.extend(f"M {qubit}" for qubit in self.qubits)

    def text(self) -> str:
        """Return newline-terminated, comment-free QCIS accepted by the QOS3 parser."""
        return "\n".join(self.lines) + "\n"


def build_qos3_dd_timing_probe(
    qubit_ids: Sequence[QubitId],
    *,
    rzz_angle: float = -0.25,
) -> str:
    """Build one undecoupled odd-bond RZZ window for QOS3 timing inspection.

    Compile the returned QCIS in the QOS3 waveform viewer and read the elapsed
    time in ns between its two ``B`` instructions. That measured value includes
    calibrated gate waveforms and hardware padding and should be passed to
    ``convert_qiskit_file_to_qcis`` as ``dd_total_duration_ns``.
    """
    qubits = normalize_qubit_ids(qubit_ids)
    if len(qubits) != 4:
        raise ValueError("The confirmed DD timing probe requires exactly four qubits")
    couplers = infer_couplers(qubits)
    compiler = QcisCompiler(qubits, couplers)

    # 对 Q99-Q106-Q100-Q107，(1, 2) 是唯一奇数键。端点 qubit 故意留空，
    # 使查看器显示由 B 自动补齐后的真实空闲窗口。
    compiler.rzz_sublayer(rzz_angle, ((1, 2),))
    return compiler.text()


def save_qos3_dd_timing_probe(
    qubit_ids: Sequence[QubitId],
    *,
    output_file: Optional[Union[str, Path]] = None,
    rzz_angle: float = -0.25,
) -> Path:
    """Save an isolated RZZ probe for measuring the DD window in QOS3."""
    probe = build_qos3_dd_timing_probe(qubit_ids, rzz_angle=rzz_angle)
    path = (
        Path(output_file).expanduser()
        if output_file is not None
        else SOURCE_PATH.parent / "qcis_preparation_circuits" / "dd_timing_probe_N4.qcis"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(probe, encoding="utf-8")
    return path


def build_qcis(
    sequence: GateSequence,
    qubits: Sequence[str],
    couplers: Sequence[str],
    *,
    dd_timing: Optional[DdTiming] = None,
    add_barriers: bool = True,
    measure: bool = False,
) -> str:
    """Build one complete QCIS preparation program from a source gate sequence."""
    compiler = QcisCompiler(qubits, couplers, dd_timing)
    compiler.compile_sequence(sequence, add_barriers=add_barriers)
    if measure:
        compiler.add_measurements()
    return compiler.text()


def save_qcis_files(
    sequences: Mapping[int, GateSequence],
    output_dir: Path,
    qubits: Sequence[str],
    couplers: Sequence[str],
    *,
    dd_timing: Optional[DdTiming] = None,
    add_barriers: bool = True,
    measure: bool = False,
) -> List[Path]:
    """Write one UTF-8 .qcis file per preparation sequence and return their paths."""
    # 先在内存中编译全部程序，再创建目录；验证或拓扑错误不会留下半套文件。
    programs = {
        sequence_id: build_qcis(
            sequences[sequence_id],
            qubits,
            couplers,
            dd_timing=dd_timing,
            add_barriers=add_barriers,
            measure=measure,
        )
        for sequence_id in sorted(sequences)
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for sequence_id, program in programs.items():
        path = output_dir / f"sequence_{sequence_id}_preparation_N{len(qubits)}.qcis"
        path.write_text(program, encoding="utf-8")
        paths.append(path)
    return paths


def _extract_transpiled_layers(
    circuit: Any,
    circuit_to_dag: Any,
) -> Tuple[Tuple[_TranspiledInstruction, ...], ...]:
    """从 Qiskit 电路提取不改变 wire 位序的 DAG 并行层。

    本函数通过参数接收 ``circuit_to_dag``，自身不导入 Qiskit，使旧转换功能和
    离线测试不依赖 Qiskit 安装。线路不能含经典条件，参数必须完全绑定；每个
    参数在这里转换为有限 ``float``。经典位只用于可选的终端测量。
    """
    if circuit.parameters:
        raise ValueError("transpiled circuit parameters must be fully bound")

    dag = circuit_to_dag(circuit)
    normalized_layers: List[Tuple[_TranspiledInstruction, ...]] = []
    for layer in dag.layers():
        instructions: List[_TranspiledInstruction] = []
        for node in layer["graph"].op_nodes():
            operation = node.op
            name = str(operation.name).lower()
            if getattr(operation, "condition", None) is not None:
                raise ValueError(
                    f"transpiled instruction {name!r} has a classical condition"
                )

            # find_bit().index 明确使用 circuit.qubits 的 wire 顺序；这里不读取
            # circuit.layout，也不猜测任何 backend 物理比特编号。
            qubit_indices = tuple(
                circuit.find_bit(qarg).index for qarg in node.qargs
            )
            clbit_indices = tuple(
                circuit.find_bit(carg).index for carg in node.cargs
            )
            parameters: List[float] = []
            for parameter in operation.params:
                try:
                    if isinstance(parameter, bool):
                        raise TypeError
                    numeric_parameter = float(parameter)
                except (TypeError, ValueError, OverflowError) as error:
                    raise ValueError(
                        f"parameters of transpiled instruction {name!r} must be "
                        "finite and fully bound"
                    ) from error
                if not math.isfinite(numeric_parameter):
                    raise ValueError(
                        f"parameters of transpiled instruction {name!r} must be "
                        "finite and fully bound"
                    )
                parameters.append(numeric_parameter)

            instructions.append(
                _TranspiledInstruction(
                    name,
                    qubit_indices,
                    tuple(parameters),
                    clbit_indices,
                    (
                        str(operation.unit).lower()
                        if name == "delay" and getattr(operation, "unit", None)
                        else None
                    ),
                )
            )
        normalized_layers.append(tuple(instructions))
    return tuple(normalized_layers)


def _qiskit_delay_to_qos3_samples(
    duration: float,
    unit: Optional[str],
    *,
    sample_rate_hz: float,
    qiskit_dt_ns: Optional[float],
) -> int:
    """把 Qiskit delay 精确换算成 QOS3 ``I`` 指令使用的 DAC samples。"""
    if duration < 0:
        raise ValueError("delay duration cannot be negative")
    normalized_unit = "dt" if unit is None else unit.lower()
    nanoseconds_by_unit = {
        "s": 1e9,
        "ms": 1e6,
        "us": 1e3,
        "ns": 1.0,
        "ps": 1e-3,
    }
    if normalized_unit == "dt":
        if qiskit_dt_ns is None:
            raise ValueError(
                "qiskit_dt_ns is required to convert a Qiskit delay in dt"
            )
        duration_ns = duration * qiskit_dt_ns
    elif normalized_unit in nanoseconds_by_unit:
        duration_ns = duration * nanoseconds_by_unit[normalized_unit]
    else:
        raise ValueError(f"unsupported Qiskit delay unit: {unit!r}")

    raw_samples = duration_ns * sample_rate_hz / 1e9
    samples = round(raw_samples)
    if not math.isclose(raw_samples, samples, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("Qiskit delay must map to an integer number of DAC samples")
    return samples


def _single_qubit_qos3_lines(
    instruction: _TranspiledInstruction,
    qubit: str,
) -> Tuple[str, ...]:
    """把一个已验证的 Qiskit 单比特门降低为等价 QCIS 指令。"""
    name = instruction.name
    parameters = instruction.parameters
    if name in _QOS3_FIXED_SINGLE_QUBIT_GATES:
        return (f"{_QOS3_FIXED_SINGLE_QUBIT_GATES[name]} {qubit}",)
    if name in _QOS3_ROTATION_GATES:
        return (
            f"{_QOS3_ROTATION_GATES[name]} {qubit} {parameters[0]:.12g}",
        )
    if name in _QOS3_FIXED_Z_ROTATIONS:
        return (f"RZ {qubit} {_QOS3_FIXED_Z_ROTATIONS[name]:.12g}",)
    if name in {"p", "u1"}:
        return (f"RZ {qubit} {parameters[0]:.12g}",)
    if name == "h":
        return (
            f"RZ {qubit} {math.pi:.12g}",
            f"RY {qubit} {math.pi / 2:.12g}",
        )
    if name == "u2":
        phi, lam = parameters
        return (
            f"RZ {qubit} {lam:.12g}",
            f"RY {qubit} {math.pi / 2:.12g}",
            f"RZ {qubit} {phi:.12g}",
        )
    if name in {"u", "u3"}:
        theta, phi, lam = parameters
        return (
            f"RZ {qubit} {lam:.12g}",
            f"RY {qubit} {theta:.12g}",
            f"RZ {qubit} {phi:.12g}",
        )
    if name == "r":
        theta, phi = parameters
        return (
            f"RZ {qubit} {-phi:.12g}",
            f"RX {qubit} {theta:.12g}",
            f"RZ {qubit} {phi:.12g}",
        )
    if name == "id":
        return ()
    raise AssertionError(f"unhandled validated single-qubit gate: {name}")


def _compile_transpiled_layers(
    layers: Sequence[Sequence[_TranspiledInstruction]],
    qubits: Sequence[str],
    *,
    add_barriers: bool,
    add_measurements: bool = False,
    sample_rate_hz: float = 2e9,
    qiskit_dt_ns: Optional[float] = None,
) -> str:
    """把经过规范化的 Qiskit 并行层编译为 QCIS 文本。

    外部负责 transpile 和门集选择；本函数只做确定性的语义映射。常见 Qiskit
    单比特门会直接映射或等价降低为 QOS3 RX/RY/RZ 与半转门，CZ 通过芯片拓扑
    解析 coupler，delay 精确换算为 DAC samples。测量默认禁用；启用后只接受
    无经典反馈的终端测量。
    """
    normalized_layers = tuple(tuple(layer) for layer in layers)
    if (
        isinstance(sample_rate_hz, bool)
        or not isinstance(sample_rate_hz, Real)
        or not math.isfinite(float(sample_rate_hz))
        or sample_rate_hz <= 0
    ):
        raise ValueError("sample_rate_hz must be finite and positive")
    if qiskit_dt_ns is not None and (
        isinstance(qiskit_dt_ns, bool)
        or not isinstance(qiskit_dt_ns, Real)
        or not math.isfinite(float(qiskit_dt_ns))
        or qiskit_dt_ns <= 0
    ):
        raise ValueError("qiskit_dt_ns must be finite and positive")
    if not isinstance(add_measurements, bool):
        raise TypeError("add_measurements must be a boolean")

    supported_names = set(_QOS3_SINGLE_QUBIT_PARAMETER_COUNTS) | {
        "barrier",
        "cz",
        "delay",
        "global_phase",
        "measure",
    }
    measured_qubits = set()
    measured_clbits = set()
    measurements_started = False
    for layer in normalized_layers:
        layer_has_measurement = any(
            instruction.name == "measure" for instruction in layer
        )
        layer_has_quantum_operation = any(
            instruction.name not in {"barrier", "global_phase", "measure"}
            for instruction in layer
        )
        if layer_has_measurement and layer_has_quantum_operation:
            raise ValueError(
                "measurements must be in terminal layers without quantum gates"
            )
        if measurements_started and layer_has_quantum_operation:
            raise ValueError("quantum gates cannot follow a measurement")
        if layer_has_measurement:
            measurements_started = True

        active_qubits = set()
        active_clbits = set()
        for instruction in layer:
            name = instruction.name
            indices = instruction.qubit_indices
            parameters = instruction.parameters
            clbit_indices = instruction.clbit_indices
            if name not in supported_names:
                raise ValueError(
                    f"unsupported transpiled instruction: {name!r}; transpile it "
                    "externally or add an explicit QOS3 mapping"
                )
            if any(
                isinstance(parameter, bool)
                or not isinstance(parameter, Real)
                or not math.isfinite(float(parameter))
                for parameter in parameters
            ):
                raise ValueError(f"{name} parameters must be finite real numbers")

            if name in _QOS3_SINGLE_QUBIT_PARAMETER_COUNTS:
                expected_parameters = _QOS3_SINGLE_QUBIT_PARAMETER_COUNTS[name]
                if len(indices) != 1:
                    raise ValueError(f"{name} requires one qubit")
                if len(parameters) != expected_parameters:
                    raise ValueError(
                        f"{name} requires {expected_parameters} parameter(s)"
                    )
                if clbit_indices:
                    raise ValueError(f"{name} does not accept classical bits")
            elif name == "cz":
                if len(indices) != 2 or parameters or clbit_indices:
                    raise ValueError(
                        "cz requires two qubits and no parameters or classical bits"
                    )
                if indices[0] == indices[1]:
                    raise ValueError("cz requires two distinct qubits")
            elif name == "delay":
                if len(indices) != 1 or len(parameters) != 1 or clbit_indices:
                    raise ValueError(
                        "delay requires one duration on one qubit and no classical bits"
                    )
                _qiskit_delay_to_qos3_samples(
                    parameters[0],
                    instruction.time_unit,
                    sample_rate_hz=float(sample_rate_hz),
                    qiskit_dt_ns=(
                        float(qiskit_dt_ns) if qiskit_dt_ns is not None else None
                    ),
                )
            elif name == "measure":
                if not add_measurements:
                    raise ValueError(
                        "Qiskit measurements require add_measurements=True"
                    )
                if len(indices) != 1 or len(clbit_indices) != 1 or parameters:
                    raise ValueError(
                        "measure requires one qubit, one classical bit, and no parameters"
                    )
                if indices[0] in measured_qubits:
                    raise ValueError("a qubit cannot be measured more than once")
                if clbit_indices[0] in measured_clbits:
                    raise ValueError("a classical bit cannot be written more than once")
                measured_qubits.add(indices[0])
                measured_clbits.add(clbit_indices[0])
            elif name == "barrier":
                if not indices or parameters or clbit_indices:
                    raise ValueError(
                        "barrier requires at least one qubit and no parameters"
                    )
            else:
                if indices or clbit_indices or len(parameters) != 1:
                    raise ValueError(
                        "global_phase requires one parameter and no bits"
                    )

            for qubit_index in indices:
                if isinstance(qubit_index, bool) or not isinstance(qubit_index, int):
                    raise TypeError("transpiled qubit index must be an integer")
                if not 0 <= qubit_index < len(qubits):
                    raise ValueError("transpiled qubit index is outside the circuit")
                if qubit_index in active_qubits:
                    raise ValueError(
                        "a transpiled layer cannot use the same qubit more than once"
                    )
                active_qubits.add(qubit_index)
            for clbit_index in clbit_indices:
                if isinstance(clbit_index, bool) or not isinstance(clbit_index, int):
                    raise TypeError("transpiled classical-bit index must be an integer")
                if clbit_index < 0:
                    raise ValueError("transpiled classical-bit index cannot be negative")
                if clbit_index in active_clbits:
                    raise ValueError(
                        "a transpiled layer cannot use the same classical bit more than once"
                    )
                active_clbits.add(clbit_index)

    # B 只同步这条线路实际使用的 coupler，并按首次出现顺序固定输出。
    coupler_by_instruction: Dict[Tuple[int, int], str] = {}
    used_couplers: List[str] = []
    coupler_edges: Dict[str, Edge] = {}
    for layer in normalized_layers:
        for instruction in layer:
            if instruction.name != "cz":
                continue
            left_index, right_index = instruction.qubit_indices
            edge = tuple(
                sorted((int(qubits[left_index][1:]), int(qubits[right_index][1:])))
            )
            try:
                registered_coupler = CHIP_TOPOLOGY[edge]
            except KeyError as error:
                raise ValueError(
                    f"Qubit pair {edge} is not present in CHIP_TOPOLOGY"
                ) from error
            coupler = _coupler_name_for_edge(
                edge,
                qubits,
                registered_coupler,
            )
            coupler_by_instruction[(left_index, right_index)] = coupler
            coupler_edges[coupler] = edge
            if coupler not in used_couplers:
                used_couplers.append(coupler)

    full_barrier = "B " + " ".join((*qubits, *used_couplers))
    lines: List[str] = []

    def append_barrier(barrier: str = full_barrier) -> None:
        """仅在上一条不是同一同步边界时追加 B。"""
        if not lines or lines[-1] != barrier:
            lines.append(barrier)

    def explicit_barrier(indices: Sequence[int]) -> str:
        selected_qubits = tuple(qubits[index] for index in indices)
        selected_numbers = {int(qubit[1:]) for qubit in selected_qubits}
        selected_couplers = tuple(
            coupler
            for coupler in used_couplers
            if selected_numbers.intersection(coupler_edges[coupler])
        )
        return "B " + " ".join((*selected_qubits, *selected_couplers))

    for layer in normalized_layers:
        if not layer:
            continue
        has_measurement = any(
            instruction.name == "measure" for instruction in layer
        )
        has_executable_gate = any(
            instruction.name not in {"barrier", "global_phase", "id", "measure"}
            for instruction in layer
        )
        if has_measurement or (add_barriers and has_executable_gate):
            append_barrier()

        for instruction in layer:
            name = instruction.name
            if name in _QOS3_SINGLE_QUBIT_PARAMETER_COUNTS:
                lines.extend(
                    _single_qubit_qos3_lines(
                        instruction,
                        qubits[instruction.qubit_indices[0]],
                    )
                )
            elif name == "cz":
                key = (instruction.qubit_indices[0], instruction.qubit_indices[1])
                lines.append(f"CZ {coupler_by_instruction[key]}")
            elif name == "delay":
                samples = _qiskit_delay_to_qos3_samples(
                    instruction.parameters[0],
                    instruction.time_unit,
                    sample_rate_hz=float(sample_rate_hz),
                    qiskit_dt_ns=(
                        float(qiskit_dt_ns) if qiskit_dt_ns is not None else None
                    ),
                )
                if samples:
                    lines.append(
                        f"I {qubits[instruction.qubit_indices[0]]} {samples}"
                    )
            elif name == "measure":
                lines.append(f"M {qubits[instruction.qubit_indices[0]]}")
            elif name == "barrier":
                append_barrier(explicit_barrier(instruction.qubit_indices))
            # global_phase 对任何测量概率均无影响，QCIS 无需输出。

        if add_barriers and has_executable_gate:
            append_barrier()

    if add_measurements and not measured_qubits:
        append_barrier()
        lines.extend(f"M {qubit}" for qubit in qubits)

    return "\n".join(lines) + ("\n" if lines else "")


def _load_qiskit_runtime() -> Tuple[Any, Any, Any]:
    """延迟加载新入口所需的 Qiskit 类型、QPY 模块和 DAG 转换函数。

    Qiskit 不是旧 QCIS 工作流的必需依赖，因此不能在模块顶层导入。只有实验人员
    调用 ``convert_transpiled_qiskit_to_qcis`` 时，本函数才检查运行环境。
    """
    try:
        from qiskit import QuantumCircuit, qpy
        from qiskit.converters import circuit_to_dag
    except ImportError as error:
        raise ImportError(
            "convert_transpiled_qiskit_to_qcis requires Qiskit; install qiskit "
            "in the conversion environment"
        ) from error
    return QuantumCircuit, qpy, circuit_to_dag


def _write_qcis_atomic(output_path: Path, text: str) -> Path:
    """通过同目录临时文件原子写入 QCIS，并在失败时保留原目标。

    同目录临时文件保证 ``os.replace`` 不跨文件系统。若写入或替换失败，临时文件
    会被删除；已有目标文件只有在完整文本成功关闭后才会被替换。
    """
    if output_path.is_dir():
        raise ValueError("output_file must not be a directory")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_name: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            delete=False,
            dir=str(output_path.parent),
            prefix=f".{output_path.name}.",
            suffix=".tmp",
        ) as temporary_file:
            temporary_name = temporary_file.name
            temporary_file.write(text)
        os.replace(temporary_name, output_path)
    except BaseException:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)
        raise
    return output_path


def convert_transpiled_qiskit_to_qcis(
    source: object,
    qubit_ids: Sequence[QubitId],
    output_file: Union[str, Path],
    *,
    add_barriers: bool = False,
    add_measurements: bool = False,
    sample_rate_hz: float = 2e9,
    qiskit_dt_ns: Optional[float] = None,
) -> Path:
    """把 transpile 后的单条 Qiskit 线路转换为 QOS3 QCIS 文件。

    ``source`` 可以是 ``QuantumCircuit`` 对象，或只保存一条线路的 ``.qpy``
    文件。``qubit_ids[i]`` 严格对应 ``circuit.qubits[i]``；函数不会读取或猜测
    Qiskit layout。单比特门支持 ``rx/ry/rz``、``x/y/sx/sxdg``、QOS3 半转门
    别名以及常见相位门和 U 门；还支持 ``cz``、``delay`` 与局部/全局
    ``barrier``。未知双比特门应由调用方在外部 transpile。

    默认生成无 ``M`` 的制备线路。``add_measurements=True`` 时，若 Qiskit 线路
    已有测量则保留无经典反馈的终端测量；否则对全部 qubit 追加测量。SI 时间单位
    的 delay 会按 ``sample_rate_hz`` 转成 DAC samples；``dt`` 单位还必须提供
    ``qiskit_dt_ns``。本入口不为通用 transpiled layer 插入 DD。

    Args:
        source: Transpile 后的 ``QuantumCircuit``，或单线路 QPY 文件路径。
        qubit_ids: 按 ``circuit.qubits`` 顺序排列的 QOS3 qubit 编号。
        output_file: 后缀必须为 ``.qcis`` 的单个输出文件路径。
        add_barriers: 是否为每个 DAG 执行层加入全组件 ``B--B`` 同步边界。
        add_measurements: 是否允许或追加终端原生 Z 测量。
        sample_rate_hz: QOS3 ``I`` 指令使用的 DAC 采样率。
        qiskit_dt_ns: Qiskit 一个 ``dt`` 对应的纳秒数；仅转换 dt delay 时需要。

    Returns:
        实际写入的 QCIS ``Path``。

    Raises:
        ImportError: 当前运行环境没有安装 Qiskit。
        TypeError: 参数类型不符合接口合同。
        ValueError: QPY 数量、wire 映射、门、参数、拓扑或输出路径不合法。
    """
    if not isinstance(add_barriers, bool):
        raise TypeError("add_barriers must be a boolean")
    if isinstance(output_file, bool) or not isinstance(output_file, (str, Path)):
        raise TypeError("output_file must be a string or Path")
    output_path = Path(output_file).expanduser()
    if output_path.suffix.lower() != ".qcis":
        raise ValueError("output_file must use the .qcis suffix")
    if output_path.is_dir():
        raise ValueError("output_file must not be a directory")

    QuantumCircuit, qpy, circuit_to_dag = _load_qiskit_runtime()
    if isinstance(source, QuantumCircuit):
        circuit = source
    elif isinstance(source, (str, Path)) and not isinstance(source, bool):
        source_path = Path(source).expanduser()
        if source_path.suffix.lower() != ".qpy":
            raise ValueError("path source must use the .qpy suffix")
        if not source_path.is_file():
            raise ValueError("source must be an existing regular .qpy file")
        with source_path.open("rb") as qpy_file:
            circuits = tuple(qpy.load(qpy_file))
        if len(circuits) != 1:
            raise ValueError("QPY source must contain exactly one circuit")
        circuit = circuits[0]
        if not isinstance(circuit, QuantumCircuit):
            raise ValueError("QPY source must contain one QuantumCircuit")
    else:
        raise TypeError("source must be a QuantumCircuit or .qpy path")

    qubits = normalize_qubit_ids(qubit_ids)
    if len(qubits) != circuit.num_qubits:
        raise ValueError(
            "qubit_ids length must equal the number of circuit.qubits"
        )

    layers = _extract_transpiled_layers(circuit, circuit_to_dag)
    text = _compile_transpiled_layers(
        layers,
        qubits,
        add_barriers=add_barriers,
        add_measurements=add_measurements,
        sample_rate_hz=sample_rate_hz,
        qiskit_dt_ns=qiskit_dt_ns,
    )
    return _write_qcis_atomic(output_path, text)


def convert_qiskit_file_to_qcis(
    source_file: object,
    qubit_ids: Sequence[QubitId],
    add_dynamic_decoupling: bool,
    *,
    # 首选：传入 QOS3 编译无 DD 测时探针后得到的波形或控制序列对象。
    # 手工 dd_total_duration_ns 仅作回退；两类来源禁止同时传入，避免歧义。
    dd_total_duration_ns: Optional[float] = None,
    qos3_waveform_data: Any = None,
    qos3_waveform_sequence_data: Any = None,
    qos3_sequence_time_unit: str = "ns",
    sample_rate_hz: float = 2e9,
    qiskit_dt_ns: Optional[float] = None,
    x2_duration_ns: float = 40.0,
    output_dir: Optional[Union[str, Path]] = None,
    add_barriers: bool = True,
    add_measurements: bool = False,
) -> List[Path]:
    """Convert legacy source sequences or a transpiled Qiskit circuit to QCIS.

    ``source_file`` 为旧 ``.py`` 文件时，保留基于抽象 RZZ 子层的四线路及 DD
    工作流；为 transpile 后的 ``QuantumCircuit`` 或单线路 ``.qpy`` 时，转发到
    广覆盖同步分层转换器并返回单元素路径列表。Transpile 输入不支持 DD。
    """
    is_path_source = isinstance(source_file, (str, Path)) and not isinstance(
        source_file, bool
    )
    source_path = Path(source_file).expanduser() if is_path_source else None
    is_transpiled_source = source_path is None or source_path.suffix.lower() == ".qpy"

    if is_transpiled_source:
        if add_dynamic_decoupling:
            raise ValueError(
                "dynamic decoupling is unavailable for transpiled Qiskit circuits"
            )
        if any(
            value is not None
            for value in (
                dd_total_duration_ns,
                qos3_waveform_data,
                qos3_waveform_sequence_data,
            )
        ):
            raise ValueError(
                "DD timing inputs are only valid for the legacy Python source workflow"
            )
        if output_dir is None:
            target_dir = (
                source_path.resolve().parent if source_path is not None else SOURCE_PATH.parent
            ) / "qcis_preparation_circuits"
        else:
            target_dir = Path(output_dir).expanduser()

        raw_stem = (
            source_path.stem
            if source_path is not None
            else str(getattr(source_file, "name", "transpiled_circuit"))
        )
        safe_stem = "".join(
            character
            if character.isascii()
            and (character.isalnum() or character in ("-", "_"))
            else "_"
            for character in raw_stem
        ).strip("_")
        if not safe_stem:
            safe_stem = "transpiled_circuit"
        output_path = target_dir / f"{safe_stem}.qcis"
        return [
            convert_transpiled_qiskit_to_qcis(
                source_file,
                qubit_ids,
                output_path,
                add_barriers=add_barriers,
                add_measurements=add_measurements,
                sample_rate_hz=sample_rate_hz,
                qiskit_dt_ns=qiskit_dt_ns,
            )
        ]

    if source_path is None or source_path.suffix.lower() != ".py":
        raise ValueError("source_file must be a .py/.qpy path or QuantumCircuit")
    if qiskit_dt_ns is not None:
        raise ValueError("qiskit_dt_ns is only valid for Qiskit circuit conversion")

    sequences = load_gate_sequences(source_path)
    qubits = normalize_qubit_ids(qubit_ids)
    couplers = infer_couplers(qubits)

    dd_timing: Optional[DdTiming] = None
    if add_dynamic_decoupling:
        # 正式实验的首选路径是 QOS3 返回数据 -> 真实 B-B 窗口 -> CPMG。
        # 这里不能从未调度的 Qiskit/QCIS 文本估算硬件 buffer；只有拿不到
        # QOS3 返回数据时，才允许实验人员把查看器读出的 ns 时长手工回填。
        has_exported_timing = (
            qos3_waveform_data is not None
            or qos3_waveform_sequence_data is not None
        )
        if dd_total_duration_ns is not None and has_exported_timing:
            raise ValueError(
                "dd_total_duration_ns and QOS3 exported timing are mutually exclusive"
            )
        if has_exported_timing:
            resolved_duration_ns = extract_qos3_window_ns(
                waveform_data=qos3_waveform_data,
                waveform_sequence_data=qos3_waveform_sequence_data,
                sequence_time_unit=qos3_sequence_time_unit,
                sample_rate_hz=sample_rate_hz,
            )
        elif dd_total_duration_ns is not None:
            resolved_duration_ns = dd_total_duration_ns
        else:
            raise ValueError(
                "dd_total_duration_ns or QOS3 exported timing is required when "
                "dynamic decoupling is enabled"
            )
        dd_timing = make_dd_timing(
            resolved_duration_ns,
            sample_rate_hz=sample_rate_hz,
            x2_duration_ns=x2_duration_ns,
        )

    target_dir = (
        Path(output_dir).expanduser()
        if output_dir is not None
        else source_path.resolve().parent / "qcis_preparation_circuits"
    )
    return save_qcis_files(
        sequences,
        target_dir,
        qubits,
        couplers,
        dd_timing=dd_timing,
        add_barriers=add_barriers,
        measure=add_measurements,
    )


def append_readout_to_qcis(
    qcis_text: str,
    qubit_ids: Sequence[QubitId],
    readout_mode: str,
) -> str:
    """在制备 QCIS 末尾追加选定的真机读取线路。

    ``01`` 模式先对齐所有读取 qubit，再直接追加 ``M``。``0_other`` 模式
    先对齐并对所有读取 qubit 施加手册规定的 ``X12``，随后直接追加 ``M``。

    Args:
        qcis_text: 不含测量指令的 QCIS 制备线路文本。
        qubit_ids: 需要同时读取的 QOS3 qubit 编号。
        readout_mode: ``"01"`` 或 ``"0_other"``。

    Returns:
        以换行符结尾、可直接提交给 ``RunCircuits`` 的最终 QCIS 文本。

    Raises:
        TypeError: QCIS 文本、读取模式或 qubit 编号类型不合法。
        ValueError: qubit 编号不合法、读取模式未知，或线路已经含有 ``M``。
    """
    if not isinstance(qcis_text, str):
        raise TypeError("qcis_text must be a string")
    if not isinstance(readout_mode, str):
        raise TypeError("readout_mode must be a string")

    normalized_mode = readout_mode.strip().lower()
    if normalized_mode not in SUPPORTED_QOS3_READOUT_MODES:
        raise ValueError(
            f"readout_mode must be one of {SUPPORTED_QOS3_READOUT_MODES}, "
            f"got {readout_mode!r}"
        )

    qubits = normalize_qubit_ids(qubit_ids)
    lines = [line.strip() for line in qcis_text.splitlines() if line.strip()]
    if any(line.split(maxsplit=1)[0].upper() == "M" for line in lines):
        raise ValueError(
            "QCIS circuit already contains an M instruction; generate a "
            "preparation-only circuit before selecting runtime readout_mode"
        )

    barrier = "B " + " ".join(qubits)
    lines.append(barrier)
    if normalized_mode == "0_other":
        # X12 是手册定义的 |1> -> |2> 跃迁门，不是计算子空间的 X/2 门。
        lines.extend(f"X12 {qubit}" for qubit in qubits)
        # 同一 qubit 的 M 会排在自身 X12 后；手册未要求二者间再次全局对齐。
    lines.extend(f"M {qubit}" for qubit in qubits)
    return "\n".join(lines) + "\n"


def run_qcis_files_with_pyqos(
    qcis_files: Sequence[Union[str, Path]],
    qubit_ids: Sequence[QubitId],
    *,
    readout_mode: str,
    data_type: str,
    sampling_interval: Any,
    num_shots: int,
    wait: bool = True,
    runner_class: Any = None,
) -> Any:
    """通过 pyqos ``RunCircuits`` 提交生成的 QCIS 文件。

    提交前根据 ``readout_mode`` 生成最终读取线路：``01`` 直接读取，
    ``0_other`` 在读取前加入并行 X12 层。``EVENT_STATE``/``EVENT_IQ``
    返回逐 shot 数据，``P01``/``IQ`` 返回 shots 平均结果。默认调用
    ``wait()``，并返回原始 RunCircuits 对象。``runner_class`` 仅用于离线
    测试或兼容其他 QOS3 安装入口。``qcis_files`` 必须是无 ``M`` 的制备
    线路，且 ``readout_mode`` 是必须显式填写的 keyword-only 参数。
    """
    normalized_data_type = str(data_type).upper()
    if normalized_data_type not in SUPPORTED_QOS3_DATA_TYPES:
        raise ValueError(
            f"data_type must be one of {SUPPORTED_QOS3_DATA_TYPES}, got {data_type!r}"
        )
    if isinstance(num_shots, bool) or not isinstance(num_shots, int) or num_shots <= 0:
        raise ValueError("num_shots must be a positive integer")
    if not isinstance(wait, bool):
        raise TypeError("wait must be a boolean")
    if not qcis_files:
        raise ValueError("At least one QCIS file is required")

    qubits = normalize_qubit_ids(qubit_ids)
    circuits = [
        append_readout_to_qcis(
            Path(path).expanduser().read_text(encoding="utf-8"),
            qubits,
            readout_mode,
        )
        for path in qcis_files
    ]
    if runner_class is None:
        try:
            from pyqos.experiment.data_taking.scan_circuits import RunCircuits
        except ImportError as exc:
            raise ImportError(
                "pyqos is required to execute QCIS files; conversion remains available offline"
            ) from exc
        runner_class = RunCircuits

    runner = runner_class(
        qubits=[list(qubits)],
        use_template=False,
        circuits=circuits,
        data_type=normalized_data_type,
        sampling_interval=sampling_interval,
        num_shots=num_shots,
    )
    if wait:
        runner.wait()
    return runner


def parse_args() -> argparse.Namespace:
    """Parse command-line qubit, DD, measurement, and output options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE_PATH)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--qubits", nargs="+", default=list(DEFAULT_QUBITS))
    parser.add_argument(
        "--dynamic-decoupling",
        action="store_true",
        help="add symmetric CPMG to qubits idle in an RZZ sublayer",
    )
    parser.add_argument(
        "--dd-total-duration-ns",
        type=float,
        help="total physical duration of one DD idle window in ns",
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=float,
        default=2e9,
        help="DAC sample rate used for QCIS I instructions",
    )
    parser.add_argument(
        "--x2-duration-ns",
        type=float,
        default=40.0,
        help="calibrated duration of one physical X/2 pulse",
    )
    parser.add_argument("--measure", action="store_true", help="append native-Z M instructions")
    parser.add_argument("--no-barriers", action="store_true", help="omit source-layer B instructions")
    return parser.parse_args()


def main() -> None:
    """Convert all four source circuits using the requested QOS3 hardware mapping."""
    args = parse_args()
    paths = convert_qiskit_file_to_qcis(
        args.source,
        args.qubits,
        args.dynamic_decoupling,
        dd_total_duration_ns=args.dd_total_duration_ns,
        sample_rate_hz=args.sample_rate_hz,
        x2_duration_ns=args.x2_duration_ns,
        output_dir=args.output_dir,
        add_barriers=not args.no_barriers,
        add_measurements=args.measure,
    )
    for path in paths:
        print(path.resolve())


if __name__ == "__main__":
    main()
