import type { Diagnostic } from "./types";
export class ApiError extends Error {
  constructor(
    message: string,
    public diagnostics: Diagnostic[] = [],
    public status = 0,
  ) {
    super(message);
    this.name = "ApiError";
  }
}
export async function api<T>(
  path: string,
  body?: unknown,
  signal?: AbortSignal,
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`/api/v1${path}`, {
      ...(body === undefined
        ? {}
        : {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body),
          }),
      signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError")
      throw error;
    throw new ApiError("无法连接服务，正在等待恢复");
  }
  const data = await response
    .json()
    .catch(() => ({ message: `服务返回异常 (${response.status})` }));
  if (!response.ok || data.valid === false) {
    const diagnostics: Diagnostic[] = data.diagnostics || [];
    const message =
      diagnostics
        .map((d) => `${d.line ? `第 ${d.line} 行：` : ""}${d.message}`)
        .join("；") ||
      data.errors?.map((e: { message: string }) => e.message).join("；") ||
      data.message ||
      data.error ||
      "请求失败";
    throw new ApiError(message, diagnostics, response.status);
  }
  return data as T;
}
export async function apiDelete<T>(path: string): Promise<T> {
  let response: Response;
  try {
    response = await fetch(`/api/v1${path}`, { method: "DELETE" });
  } catch {
    throw new ApiError("无法连接服务，正在等待恢复");
  }
  const data = await response
    .json()
    .catch(() => ({ message: `服务返回异常 (${response.status})` }));
  if (!response.ok || data.valid === false) {
    const diagnostics: Diagnostic[] = data.diagnostics || [];
    const message =
      diagnostics
        .map((d) => `${d.line ? `第 ${d.line} 行：` : ""}${d.message}`)
        .join("；") ||
      data.errors?.map((e: { message: string }) => e.message).join("；") ||
      data.message ||
      data.error ||
      "删除失败";
    throw new ApiError(message, diagnostics, response.status);
  }
  return data as T;
}
export const terminal = (status: string) =>
  ["SUCCEEDED", "FAILED", "CANCELLED", "INTERRUPTED"].includes(status);
export const statusLabel = (status: string) =>
  ({
    QUEUED: "排队中",
    SUBMITTED: "已提交",
    RUNNING: "运行中",
    CANCELLING: "正在取消",
    SUCCEEDED: "已完成",
    FAILED: "失败",
    CANCELLED: "已取消",
    INTERRUPTED: "已中断",
  })[status] || status;
export function duration(value: unknown) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "—";
  return value < 1
    ? `${(value * 1000).toFixed(1)} ms`
    : value < 60
      ? `${value.toFixed(2)} s`
      : `${(value / 60).toFixed(2)} min`;
}
export function stored<T>(key: string): T | null {
  try {
    return JSON.parse(localStorage.getItem(key) || "null") as T | null;
  } catch {
    return null;
  }
}
export function store(key: string, value: unknown) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* Private mode and storage limits do not block computation. */
  }
}
export function setLocation(key: "run" | "prediction", value: string) {
  const url = new URL(location.href);
  url.searchParams.set(key, value);
  history.replaceState(null, "", url);
}
