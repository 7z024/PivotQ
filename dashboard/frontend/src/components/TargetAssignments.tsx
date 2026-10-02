import { actualDevice } from "../api/hardware";
import type { Plan, RunResult } from "../api/types";
import TargetDetails from "./TargetDetails";
import s from "../App.module.css";

export default function TargetAssignments({
  plan,
  result,
}: {
  plan: Plan;
  result?: RunResult | null;
}) {
  return (
    <div className={s.targetAssignments} aria-label="保存的目标硬件配置">
      {plan.stages.map((stage) => {
        const snapshot = stage.target_snapshot,
          execution = result?.stage_results?.find(
            (item) => item.stage_id === stage.id,
          ),
          actual =
            execution?.actual_device ||
            actualDevice(result) ||
            execution?.device ||
            "",
          label =
            actual.toLowerCase() === "cpu" &&
            result?.execution_mode === "local_cpu"
              ? "CPU 数值模拟"
              : actual.toUpperCase();
        return (
          <div key={stage.id} className={s.targetAssignment}>
            <strong>{stage.title}</strong>
            <div>
              <span>
                目标：
                {snapshot?.title ||
                  stage.target_id ||
                  stage.device.toUpperCase()}
              </span>
              {result && <span>实际执行：{label || "未记录"}</span>}
              {snapshot?.logical_qubits !== undefined && (
                <span>
                  实际电路：{snapshot.logical_qubits} 逻辑比特
                  {snapshot.parameters.qubits
                    ? ` / 芯片容量：${snapshot.parameters.qubits} 比特`
                    : ""}
                </span>
              )}
              {snapshot && (
                <span className={s.snapshotVersion}>
                  参数版本 {snapshot.profile_version} ·{" "}
                  {snapshot.profile_sha256.slice(0, 12)}
                </span>
              )}
              {snapshot && (
                <details>
                  <summary>
                    {snapshot.parameters.topology
                      ? "已保存的芯片参数"
                      : "已保存的目标参数"}
                  </summary>
                  <TargetDetails snapshot={snapshot} />
                </details>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
