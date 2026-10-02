import { duration } from "../api/client";
import type { CSSProperties } from "react";
import type { Prediction } from "../api/types";
import TargetAssignments from "../components/TargetAssignments";
import s from "./Pages.module.css";

const phaseNames: Record<string, string> = {
  actor_setup: "计算服务初始化",
  dataset_and_ood_setup: "数据初始化",
  statevector: "量子模拟",
  classical_actor: "经典推理",
  bookkeeping: "积分与记录",
  force_host: "求力",
  plot_artifacts: "结果图表",
  worker_setup_teardown: "计算进程初始化与收尾",
  feature_readout: "特征读出",
  http_submit: "QPU 提交",
  acquisition: "QPU 采样",
};

export default function PerformancePage({ prediction }: { prediction: Prediction | null }) {
  const value = prediction?.result?.prediction;
  const stages = Object.entries(value?.stage_seconds || value?.phase_seconds || {});
  const total = value?.latency_seconds || 0;
  const resources = prediction?.plan.stages || [];

  return (
    <div className={s.page}>
      <header className={s.hero}>
        <p className={s.eyebrow}>PERFORMANCE FORECAST</p>
        <h1>性能预测</h1>
        <p>按提交时保存的任务参数和硬件分配，查看完整的预测流程与阶段耗时。</p>
      </header>
      {!prediction || !value ? (
        <section className={`${s.card} ${s.empty}`}>
          <h2>暂无性能预测</h2>
          <p>返回工作台，选择硬件后点击“性能预测”。</p>
          <a className={s.button} href="/">返回工作台</a>
        </section>
      ) : (
        <>
          <section className={s.overview} aria-label="预测摘要">
            <div className={`${s.card} ${s.stat}`}><span>预计总耗时</span><strong>{duration(total)}</strong></div>
            <div className={`${s.card} ${s.stat}`}><span>预测阶段</span><strong>{stages.length}</strong></div>
            <div className={`${s.card} ${s.stat}`}><span>主要瓶颈</span><strong>{value.main_bottleneck ? phaseNames[value.main_bottleneck] || value.main_bottleneck : "—"}</strong></div>
            <div className={`${s.card} ${s.stat}`}><span>预测来源</span><strong>QPerfSim</strong></div>
          </section>
          <section className={`${s.card} ${s.section}`}>
            <div className={s.sectionHeader}><h2>执行流程与阶段耗时</h2><span className={s.muted}>总计 {duration(total)}</span></div>
            <div className={s.pipeline}>
              {stages.map(([name, seconds]) => {
                const amount = typeof seconds === "number" ? seconds : Number(seconds);
                return <div key={name} className={s.stage} style={{ "--stage-progress": `${Math.min(1, amount / (total || 1))}` } as CSSProperties}>
                  <span>{phaseNames[name] || name.replaceAll("_", " ")}</span>
                  <strong>{duration(amount)}</strong>
                  <small>{Math.round((amount / (total || 1)) * 100)}% of forecast</small>
                </div>;
              })}
            </div>
          </section>
          <section className={s.twoColumns}>
            <div className={`${s.card} ${s.section}`}>
              <div className={s.sectionHeader}><h2>任务与资源</h2><span className={s.muted}>{prediction.plan.task_id}</span></div>
              <div className={s.resourceList}>
                {resources.map((stage) => <div key={stage.id} className={s.resource}><div><strong>{stage.title}</strong><code>{stage.id}</code></div><span>{stage.target_snapshot?.title || stage.target_id || stage.device}</span></div>)}
              </div>
              <TargetAssignments plan={prediction.plan} />
            </div>
            <div className={`${s.card} ${s.section}`}>
              <div className={s.sectionHeader}><h2>预测范围</h2><span className={s.muted}>模型说明</span></div>
              {[...(prediction.model_scope?.notes || []), ...(prediction.request?.scope_notes || []), prediction.request?.validation_scope || "不包含外部平台排队时间。"].filter(Boolean).map((note, index) => <p key={index} className={s.muted} style={{ marginBottom: 10 }}>{note}</p>)}
            </div>
          </section>
        </>
      )}
    </div>
  );
}
