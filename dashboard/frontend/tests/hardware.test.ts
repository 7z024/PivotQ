import { expect, it } from "vitest";
import {
  actualDevice,
  hardwareChoices,
  hardwareProfileDigests,
  repairFixedChoices,
} from "../src/api/hardware";
import type { Stage, Target } from "../src/api/types";
const stages: Stage[] = [
  {
    id: "init",
    title: "初始化",
    allowed_devices: ["cpu"],
    default_device: "cpu",
    fixed_device: true,
    depends_on: [],
  },
  {
    id: "quantum",
    title: "量子计算",
    allowed_devices: ["gpu", "qpu"],
    default_device: "gpu",
    depends_on: ["init"],
  },
];
const targets: Target[] = [
  { id: "new-cpu", title: "CPU", kind: "cpu", available: true },
  { id: "new-qpu", title: "QPU", kind: "qpu", available: true },
];
it("clears removed targets instead of keeping stale device ids", () => {
  const selected = hardwareChoices(stages, targets, {
    init: "old-cpu",
    quantum: "old-gpu",
  });
  expect(selected).toEqual({ init: "new-cpu", quantum: "" });
  expect(hardwareChoices(stages, targets).quantum).toBe("");
});
it("clears removed targets during discovery refresh", () => {
  const previous = { init: "", quantum: "old-gpu" };
  const updated = repairFixedChoices(stages, targets, previous);
  expect(updated).toEqual({ init: "new-cpu", quantum: "" });
  expect(repairFixedChoices(stages, targets, updated)).toBe(updated);
});
it("uses the quantum default for new drafts and preserves an old CPU draft", () => {
  const stage = {
    ...stages[1],
    default_device: "qpu",
    allowed_devices: ["cpu", "gpu", "qpu"],
  };
  expect(hardwareChoices([stage], targets)).toEqual({ quantum: "new-qpu" });
  expect(hardwareChoices([stage], targets, { quantum: "new-cpu" })).toEqual({
    quantum: "new-cpu",
  });
});
it("refreshes profile digests without changing the logical target selection", () => {
  const target: Target = {
    ...targets[1],
    target_snapshot: {
      id: "new-qpu",
      title: "Fake SC-36",
      kind: "qpu",
      profile_version: "1",
      profile_sha256: "first",
      parameters: { qubits: 36 },
    },
  };
  const selection = { quantum: "new-qpu", init: "new-cpu" };
  expect(hardwareProfileDigests(selection, [target])).toEqual({
    quantum: "first",
  });
  target.target_snapshot!.profile_sha256 = "second";
  expect(hardwareProfileDigests(selection, [target])).toEqual({
    quantum: "second",
  });
  expect(hardwareProfileDigests(selection, targets)).toEqual({});
});
it("prefers explicit execution over logical or stale quantum metadata", () => {
  expect(
    actualDevice({
      actual_device: "cpu",
      quantum_execution: { actual_device: "qpu", requested_device: "qpu" },
    }),
  ).toBe("cpu");
  expect(actualDevice({ execution_mode: "local_cpu" })).toBe("cpu");
  expect(actualDevice({ quantum_execution: { requested_device: "qpu" } })).toBe(
    "",
  );
});
