import { useEffect, useState } from "react";
import { api, terminal } from "../api/client";
import type { ResultResponse, Run, Series } from "../api/types";
import { pollUntil } from "./poll";
export function useRun(id: string | null) {
  const [run, setRun] = useState<Run | null>(null),
    [series, setSeries] = useState<Series>({ items: [] }),
    [error, setError] = useState("");
  useEffect(() => {
    setRun(null);
    setSeries({ items: [] });
    setError("");
    if (!id) return;
    return pollUntil(
      async (signal) => {
        const run = await api<Run>(
          `/runs/${encodeURIComponent(id)}`,
          undefined,
          signal,
        );
        const result = await api<ResultResponse>(
          `/runs/${encodeURIComponent(id)}/result`,
          undefined,
          signal,
        );
        let series: Series = { items: [] };
        if (run.task_id === "h2o-hybrid-aimd")
          series = await api<Series>(
            `/runs/${encodeURIComponent(id)}/series`,
            undefined,
            signal,
          );
        return {
          run: { ...run, result: result.result, events: result.events },
          series,
        };
      },
      (data) => {
        setRun(data.run);
        setSeries(data.series);
        setError("");
      },
      (e) => setError(e instanceof Error ? e.message : "同步失败"),
      (data) => terminal(data.run.status),
    );
  }, [id]);
  return { run, series, error };
}
