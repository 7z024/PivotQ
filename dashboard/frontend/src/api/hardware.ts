import type { RunResult, Stage, Target } from "./types";

/** A changed authoritative profile invalidates compilation even if its ID is unchanged. */
export function hardwareProfileDigests(
  hardware: Record<string, string>,
  targets: Target[],
) {
  return Object.fromEntries(
    Object.entries(hardware).flatMap(([stage, id]) => {
      const digest = targets.find((target) => target.id === id)?.target_snapshot
        ?.profile_sha256;
      return digest ? [[stage, digest]] : [];
    }),
  );
}

/** Actual execution must never be inferred from a logical hardware selection. */
export function actualDevice(result?: RunResult | null) {
  return (
    result?.actual_device ||
    result?.quantum_execution?.actual_device ||
    result?.device_name ||
    String(result?.summary?.device_name || "") ||
    (result?.execution_mode === "local_cpu" ? "cpu" : "")
  );
}
/** Device kind is fixed for coordinator stages; a removed target must not trap old drafts. */
export function hardwareChoices(
  stages: Stage[],
  targets: Target[],
  saved: Record<string, string> = {},
) {
  return Object.fromEntries(
    stages.map((stage) => {
      const previous = saved[stage.id];
      const compatible = targets.filter((target) =>
        stage.allowed_devices.includes(target.kind),
      );
      // A saved draft may refer to a target that has since been unregistered.
      // Keep it only when the current backend resource list still contains it;
      // otherwise the UI must not display or submit a stale CPU/GPU/QPU id.
      if (previous && compatible.some((target) => target.id === previous))
        return [stage.id, previous];
      const available = compatible.find(
        (target) => target.kind === stage.default_device && target.available,
      );
      return [stage.id, available?.id || ""];
    }),
  );
}
export function repairFixedChoices(
  stages: Stage[],
  targets: Target[],
  saved: Record<string, string>,
) {
  const proposed = hardwareChoices(stages, targets, saved);
  let changed = false;
  const next = { ...saved };
  for (const stage of stages)
    if (proposed[stage.id] !== saved[stage.id]) {
      next[stage.id] = proposed[stage.id];
      changed = true;
    }
  return changed ? next : saved;
}
