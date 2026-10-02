#!/usr/bin/env python3
"""Start the QHAI workstation and API from any working directory."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import sys
import threading
import webbrowser


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="QHAI Dashboard · 本地计算工作台")
    parser.add_argument("--mode", choices=("local-cpu", "ray", "demo"), help="默认 local-cpu；兼容 FUSION_EXECUTOR")
    parser.add_argument("--host", help="默认 127.0.0.1")
    parser.add_argument("--port", type=int, help="默认 8787")
    parser.add_argument("--no-open", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args(argv)
    if args.port is not None and not 1 <= args.port <= 65535:
        parser.error("端口必须在 1–65535 之间")
    for key, value in (("FUSION_API_HOST", args.host), ("FUSION_API_PORT", args.port)):
        if value is not None:
            os.environ[key] = str(value)
    if args.mode:
        os.environ["FUSION_EXECUTOR"] = {"local-cpu": "local_cpu", "demo": "dry_run"}.get(args.mode, args.mode)
    from backend.paths import DASHBOARD_ROOT, configure_defaults, execution_mode
    configure_defaults()
    mode = execution_mode()
    if mode not in {"local_cpu", "ray", "dry_run"}:
        parser.error(f"未知执行模式：{mode}")
    os.environ["FUSION_EXECUTOR"] = mode
    if mode != "ray" and os.environ.get("FUSION_DEVICE_REGISTRATION") == "network":
        parser.error("网络设备登记需要 --mode ray")
    if not (DASHBOARD_ROOT / "frontend/dist/index.html").is_file():
        print("缺少前端构建资源。请使用包含 dist 的发行版，或执行：\n  cd dashboard/frontend && npm ci && npm run build", file=sys.stderr)
        return 1
    from backend import server
    httpd = None
    timer = None
    def stop(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    try:
        server.initialize()
        host, port = os.environ["FUSION_API_HOST"], int(os.environ["FUSION_API_PORT"])
        httpd = server.ThreadingHTTPServer((host, port), server.Handler)
        httpd.daemon_threads = True
        url = f"http://{'127.0.0.1' if host in ('0.0.0.0', '::') else host}:{port}"
        print(f"QHAI Dashboard: {url}\n执行模式: {mode}\n运行记录: {server.HISTORY_FILE}", flush=True)
        if server.EXECUTOR_ERROR:
            print(f"计算暂不可用: {server.EXECUTOR_ERROR}", flush=True)
        if not args.no_open and (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY") or sys.platform in {"darwin", "win32"}):
            timer = threading.Timer(0.4, webbrowser.open, args=(url,))
            timer.daemon = True
            timer.start()
        httpd.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("正在关闭工作台…", flush=True)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"无法启动工作台：{error}", file=sys.stderr)
        return 1
    finally:
        if timer:
            timer.cancel()
        if httpd:
            httpd.server_close()
        server.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
