import { afterEach, describe, expect, it, vi } from "vitest";
import { pollUntil } from "../src/hooks/poll";
import { api, ApiError, terminal } from "../src/api/client";
afterEach(() => {
  vi.useRealTimers();
  vi.unstubAllGlobals();
});
describe("sequential run polling", () => {
  it("retries an offline response, keeps updating and stops on terminal state", async () => {
    vi.useFakeTimers();
    const load = vi
      .fn()
      .mockRejectedValueOnce(new Error("offline"))
      .mockResolvedValueOnce("RUNNING")
      .mockResolvedValueOnce("SUCCEEDED");
    const receive = vi.fn(),
      error = vi.fn();
    const stop = pollUntil(load, receive, error, terminal);
    await vi.advanceTimersByTimeAsync(0);
    expect(error).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(1000);
    expect(receive).toHaveBeenLastCalledWith("RUNNING");
    await vi.advanceTimersByTimeAsync(1000);
    expect(receive).toHaveBeenLastCalledWith("SUCCEEDED");
    await vi.advanceTimersByTimeAsync(10000);
    expect(load).toHaveBeenCalledTimes(3);
    stop();
  });
  it("suppresses late updates when a different run is selected", async () => {
    vi.useFakeTimers();
    let resolve!: (s: string) => void;
    const receive = vi.fn();
    const stop = pollUntil(
      () => new Promise<string>((r) => (resolve = r)),
      receive,
      vi.fn(),
      terminal,
    );
    stop();
    resolve("SUCCEEDED");
    await vi.advanceTimersByTimeAsync(0);
    expect(receive).not.toHaveBeenCalled();
  });
  it("does not overlap slow requests", async () => {
    vi.useFakeTimers();
    let resolve!: (s: string) => void;
    const load = vi.fn(() => new Promise<string>((r) => (resolve = r)));
    const stop = pollUntil(load, vi.fn(), vi.fn(), terminal);
    await vi.advanceTimersByTimeAsync(3000);
    expect(load).toHaveBeenCalledTimes(1);
    resolve("RUNNING");
    await vi.advanceTimersByTimeAsync(1000);
    expect(load).toHaveBeenCalledTimes(2);
    stop();
  });
});
it("preserves compiler line diagnostics and uses same-origin API", async () => {
  const fetch = vi
    .fn()
    .mockResolvedValue(
      new Response(
        JSON.stringify({
          valid: false,
          diagnostics: [{ line: 4, message: "不支持的量子门" }],
        }),
        { status: 422, headers: { "Content-Type": "application/json" } },
      ),
    );
  vi.stubGlobal("fetch", fetch);
  try {
    await api("/projects/circuit/compile", { source: "code" });
    throw new Error("expected error");
  } catch (error) {
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).diagnostics[0].line).toBe(4);
    expect((error as Error).message).toContain("第 4 行");
  }
  expect(fetch.mock.calls[0][0]).toBe("/api/v1/projects/circuit/compile");
});
