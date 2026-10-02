/** A sequential poll loop: retries errors, never overlaps, and stops at terminal state. */
export function pollUntil<T>(
  load: (signal: AbortSignal) => Promise<T>,
  receive: (data: T) => void,
  failed: (error: unknown) => void,
  done: (data: T) => boolean,
  interval = 1000,
) {
  let stopped = false;
  let timer: ReturnType<typeof setTimeout> | undefined;
  const controller = new AbortController();
  async function tick() {
    let complete = false;
    try {
      const data = await load(controller.signal);
      if (stopped) return;
      receive(data);
      complete = done(data);
    } catch (error) {
      if (stopped) return;
      failed(error);
    }
    if (!stopped && !complete) timer = setTimeout(tick, interval);
  }
  void tick();
  return () => {
    stopped = true;
    controller.abort();
    clearTimeout(timer);
  };
}
