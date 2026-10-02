import type { SeriesPoint } from "../api/types";
import s from "../App.module.css";
export function EnergyChart({
  points,
  time,
  onTime,
}: {
  points: SeriesPoint[];
  time?: number;
  onTime?: (time: number) => void;
}) {
  const valid = points.filter(
    (p) =>
      Number.isFinite(p.time_fs) &&
      typeof p.total_energy_eV === "number" &&
      Number.isFinite(p.total_energy_eV),
  );
  if (!valid.length)
    return <div className={s.smallEmpty}>运行后显示能量曲线</div>;
  const w = 650,
    h = 180,
    left = 72,
    right = 22,
    top = 24,
    bottom = 38,
    x0 = Math.min(...valid.map((p) => p.time_fs)),
    x1 = Math.max(...valid.map((p) => p.time_fs)),
    values = valid.map((p) => p.total_energy_eV as number),
    lo = Math.min(...values),
    hi = Math.max(...values),
    pad = Math.max((hi - lo) * 0.15, 1e-7),
    min = lo - pad,
    max = hi + pad,
    x = (t: number) => left + ((t - x0) / (x1 - x0 || 1)) * (w - left - right),
    y = (v: number) => top + ((max - v) / (max - min)) * (h - top - bottom),
    path = valid
      .map(
        (p, i) =>
          `${i ? "L" : "M"}${x(p.time_fs)},${y(p.total_energy_eV as number)}`,
      )
      .join(" "),
    current =
      time === undefined
        ? valid.at(-1)!
        : valid.reduce((a, b) =>
            Math.abs(a.time_fs - time) < Math.abs(b.time_fs - time) ? a : b,
          );
  return (
    <div className={s.chart}>
      <div className={s.chartTitle}>
        <h3>总能量</h3>
        <span>{Number(current.total_energy_eV).toFixed(5)} eV</span>
      </div>
      <svg
        viewBox={`0 0 ${w} ${h}`}
        role="img"
        aria-label="总能量随时间变化，单位 eV 与 fs"
        onClick={(e) => {
          const box = e.currentTarget.getBoundingClientRect();
          const t =
            x0 +
            Math.max(
              0,
              Math.min(
                1,
                (((e.clientX - box.left) / box.width) * w - left) /
                  (w - left - right),
              ),
            ) *
              (x1 - x0);
          onTime?.(t);
        }}
        style={{ cursor: onTime ? "crosshair" : "default" }}
      >
        {[0, 0.5, 1].map((f) => (
          <g key={f}>
            <path
              d={`M${left} ${top + f * (h - top - bottom)}H${w - right}`}
              stroke="#E7EDF5"
            />
            <text
              x={left - 10}
              y={top + f * (h - top - bottom) + 4}
              textAnchor="end"
              fontSize="11"
              fill="#718096"
            >
              {(max - f * (max - min)).toFixed(5)}
            </text>
          </g>
        ))}
        <path d={path} stroke="#2458D3" strokeWidth="2" fill="none" />
        {[x0, x1].map((t, i) => (
          <text
            key={i}
            x={x(t)}
            y={h - 13}
            textAnchor={i ? "end" : "start"}
            fontSize="11"
            fill="#718096"
          >
            {t.toFixed(2)} fs
          </text>
        ))}
        <path
          d={`M${x(current.time_fs)} ${top}V${h - bottom}`}
          stroke="#9EB6EB"
          strokeDasharray="3 4"
        />
        <circle
          cx={x(current.time_fs)}
          cy={y(current.total_energy_eV as number)}
          r="4"
          fill="#2458D3"
          stroke="white"
          strokeWidth="2"
        />
      </svg>
    </div>
  );
}
export function Probabilities({
  values,
  counts,
}: {
  values: Record<string, number>;
  counts?: Record<string, number>;
}) {
  const entries = Object.entries(values).sort((a, b) => b[1] - a[1]);
  return (
    <div className={s.probabilities}>
      <div className={s.chartTitle}>
        <h3>测量概率</h3>
        <span>位序 q0 → q(n−1)</span>
      </div>
      {entries.slice(0, 32).map(([state, p]) => (
        <div key={state} className={s.probability}>
          <code>|{state}⟩</code>
          <div>
            <i style={{ width: `${Math.max(0, Math.min(100, p * 100))}%` }} />
          </div>
          <strong>{(p * 100).toFixed(2)}%</strong>
          <span>{counts?.[state] ?? "—"} 次</span>
        </div>
      ))}
      {entries.length > 32 && (
        <p className={s.scope}>
          显示概率最高的 32 个状态；完整结果见输出文件。
        </p>
      )}
    </div>
  );
}
