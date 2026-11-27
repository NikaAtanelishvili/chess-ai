"""Games and puzzles in worker processes, supervised so that a run can never hang.

Each worker builds its players once (networks, search trees, engine processes) and takes
tasks one at a time. The supervisor, in the calling process, hands out tasks, prints a
heartbeat line, and enforces three limits:

* ``move_timeout``: no progress (no move played) for that long;
* ``task_timeout``: one task (a game, a puzzle) taking longer than that;
* ``deadline``: the wall-clock end of the whole run.

A worker over a limit is killed together with its engine processes (it leads its own
process group) and replaced; its task is reported as :class:`Failed`. A task that raises,
such as an engine call timing out, fails the same way, and the worker restarts its
players. A worker that cannot build its players hands its task back; after
``MAX_FUTILE_RESTARTS`` workers in a row die without finishing anything, the remaining
tasks fail instead of being retried forever. Workers close their players themselves before
exiting: python-chess runs each engine in a non-daemon thread, which would otherwise keep
the process alive forever.
"""

from __future__ import annotations

import contextlib
import multiprocessing
import os
import signal
import time
from collections import deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from multiprocessing.connection import Connection, wait
from typing import Any, Generic, TypeVar

from chessnet.engine.player import Player
from chessnet.eval.players import PlayerSpec, build_players, close_players, resolve_device

Item = TypeVar("Item")
Result = TypeVar("Result")
Beat = Callable[[], None]
Task = Callable[[tuple[Player, ...], Any, Beat], Any]

#: Workers that die before finishing anything, in a row, before the run gives up.
MAX_FUTILE_RESTARTS = 3
#: Seconds between two progress messages from a worker.
BEAT_INTERVAL = 1.0


@dataclass(frozen=True)
class Watchdog:
    """Limits for a supervised run; times in seconds (``deadline``: ``time.monotonic()``)."""

    move_timeout: float | None = 300.0
    task_timeout: float | None = 3600.0
    deadline: float | None = None
    heartbeat: float = 60.0


@dataclass(frozen=True)
class Failed(Generic[Item]):
    """A task that did not finish; ``stopped`` means the deadline, not an error, stopped it."""

    item: Item
    reason: str
    stopped: bool = False


def _no_beat() -> None:
    pass


def map_with_players(
    specs: tuple[PlayerSpec, ...],
    function: Task,
    items: Sequence[Item],
    *,
    workers: int = 1,
    device: str = "cpu",
    threads: int | None = None,
    watchdog: Watchdog | None = None,
    label: str = "",
    report: Callable[[str], None] = print,
) -> Iterator[Any]:
    """``function(players, item, beat)`` for every item, as results or :class:`Failed`.

    Without a watchdog and with one worker everything runs in this process, and errors
    propagate. Otherwise ``workers`` processes run the tasks under the watchdog's limits
    (``threads`` torch threads each, by default the CPUs shared out). ``function`` must be a
    module-level function; it calls ``beat()`` whenever it makes progress.
    """
    if watchdog is None and workers <= 1:
        players = build_players(specs, resolve_device(device))
        try:
            for item in items:
                yield function(players, item, _no_beat)
        finally:
            close_players(players)
        return
    count = max(1, min(workers, len(items)))
    per_worker = threads or max(1, multiprocessing.cpu_count() // count)
    supervisor = _Supervisor(specs, function, items, device, per_worker, watchdog or Watchdog())
    yield from supervisor.run(count, label, report)


class _Worker:
    """One worker process with its own pipes, so killing it cannot corrupt the others'."""

    def __init__(
        self, ident: int, specs: tuple[PlayerSpec, ...], device: str, threads: int
    ) -> None:
        context = multiprocessing.get_context("spawn")
        self.ident = ident
        self.tasks, task_end = context.Pipe(duplex=False)  # (receiver, sender)
        result_end, self.results = context.Pipe(duplex=False)
        self.process = context.Process(
            target=_work,
            args=(ident, specs, device, threads, self.tasks, self.results),
            daemon=True,
            name=f"eval-worker-{ident}",
        )
        self.process.start()
        self.tasks.close()  # the worker's ends now live in the worker
        self.results.close()
        self.tasks, self.results = task_end, result_end
        self.task: int | None = None
        self.started = self.beat = 0.0
        self.finished_any = False
        self.broken = ""  # why it could not build its players, if it said so

    def assign(self, index: int, function: Task, item: Any) -> None:
        self.tasks.send((index, function, item))
        self.task = index
        self.started = self.beat = time.monotonic()

    def receive(self) -> list[tuple[str, int | None, Any]]:
        """The messages that have arrived; none once the worker is gone and they are read."""
        messages = []
        with contextlib.suppress(EOFError, OSError):
            while self.results.poll():
                messages.append(self.results.recv())
        return messages

    def stop(self) -> None:
        with contextlib.suppress(OSError, ValueError):
            self.tasks.send(None)

    def kill(self) -> None:
        """Kill the worker and its engine processes (its process group)."""
        pid = self.process.pid
        try:
            if pid is None:
                raise ProcessLookupError
            os.killpg(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            self.process.kill()
        self.process.join(timeout=10)
        self.close()

    def close(self) -> None:
        for connection in (self.tasks, self.results):
            connection.close()


def _work(
    ident: int,
    specs: tuple[PlayerSpec, ...],
    device: str,
    threads: int,
    tasks: Connection,
    results: Connection,
) -> None:
    """A worker: build the players, run tasks until told to stop or the supervisor is gone."""
    os.setsid()  # its own process group: the supervisor can kill it with its engines
    supervisor = os.getppid()
    import torch

    torch.set_num_threads(threads)
    players = _build(specs, device, ident, results)
    if players is None:
        return
    try:
        while True:
            if not tasks.poll(2.0):
                if os.getppid() != supervisor:
                    return  # orphaned: nobody will read the results
                continue
            task = tasks.recv()
            if task is None:
                return
            index, function, item = task
            last = [0.0]

            def beat(index: int = index, last: list[float] = last) -> None:
                now = time.monotonic()
                if now - last[0] >= BEAT_INTERVAL:
                    last[0] = now
                    results.send(("beat", index, None))

            try:
                value = function(players, item, beat)
            except Exception as error:
                results.send(("failed", index, f"{type(error).__name__}: {error}"))
                close_players(players)  # a stuck engine is quit, or killed
                players = _build(specs, device, ident, results) or ()
                if not players:
                    return
            else:
                results.send(("done", index, value))
    finally:
        close_players(players)  # before the interpreter waits for the engines' threads


def _build(
    specs: tuple[PlayerSpec, ...], device: str, ident: int, results: Connection
) -> tuple[Player, ...] | None:
    """The worker's players, or None after telling the supervisor why they cannot be built."""
    try:
        return build_players(specs, resolve_device(device, ident))
    except Exception as error:
        results.send(("broken", None, f"{type(error).__name__}: {error}"))
        return None


class _Supervisor(Generic[Item]):
    def __init__(
        self,
        specs: tuple[PlayerSpec, ...],
        function: Task,
        items: Sequence[Item],
        device: str,
        threads: int,
        watchdog: Watchdog,
    ) -> None:
        self.specs, self.function, self.items = specs, function, items
        self.device, self.threads, self.watchdog = device, threads, watchdog
        self.next_ident = 0
        self.finished = 0  # tasks done or failed
        self.futile = 0  # workers in a row that died without finishing a task

    def _start(self) -> _Worker:
        worker = _Worker(self.next_ident, self.specs, self.device, self.threads)
        self.next_ident += 1
        return worker

    def _collect(self, worker: _Worker) -> list[Any]:
        """Apply the messages ``worker`` sent; returns the results and failures to yield."""
        outcomes: list[Any] = []
        for kind, index, payload in worker.receive():
            if kind == "beat" and index == worker.task:
                worker.beat = time.monotonic()
            elif kind in ("done", "failed") and index is not None and index == worker.task:
                worker.task, worker.finished_any = None, True
                outcomes.append(payload if kind == "done" else Failed(self.items[index], payload))
            elif kind == "broken":
                worker.broken = payload
        if outcomes:
            self.finished += len(outcomes)
            self.futile = 0
        return outcomes

    def run(self, count: int, label: str, report: Callable[[str], None]) -> Iterator[Any]:
        watchdog, items = self.watchdog, self.items
        pending = deque(range(len(items)))
        workers = [self._start() for _ in range(count)]
        started = last_heartbeat = time.monotonic()
        try:
            while pending or any(worker.task is not None for worker in workers):
                if watchdog.deadline is not None and time.monotonic() >= watchdog.deadline:
                    for worker in workers:
                        yield from self._collect(worker)  # what finished in time counts
                        if worker.task is not None:
                            yield Failed(items[worker.task], "stopped at the wall-time limit", True)
                            worker.task = None
                            worker.kill()
                    for index in pending:
                        yield Failed(items[index], "not started: wall-time limit", True)
                    pending.clear()
                    break
                for worker in workers:
                    if worker.task is None and pending and not worker.broken:
                        index = pending.popleft()
                        worker.assign(index, self.function, items[index])
                wait([worker.results for worker in workers], timeout=1.0)

                now = time.monotonic()
                survivors: list[_Worker] = []
                for worker in workers:
                    alive = (
                        worker.process.is_alive()
                    )  # first, so a dead worker's last words are read
                    yield from self._collect(worker)
                    reason = None
                    if not alive:
                        reason = (
                            f"a worker could not start: {worker.broken}"
                            if worker.broken
                            else f"worker exited (code {worker.process.exitcode})"
                        )
                    elif worker.task is not None:
                        if watchdog.move_timeout and now - worker.beat > watchdog.move_timeout:
                            reason = f"no progress for {watchdog.move_timeout:g} s"
                        elif watchdog.task_timeout and now - worker.started > watchdog.task_timeout:
                            reason = f"took more than {watchdog.task_timeout:g} s"
                    if reason is None:
                        survivors.append(worker)
                        continue
                    task, worker.task = worker.task, None
                    worker.kill()
                    self.futile = 0 if worker.finished_any else self.futile + 1
                    if task is not None and worker.broken:
                        pending.appendleft(task)  # it never started the task
                        task = None
                    give_up = self.futile > MAX_FUTILE_RESTARTS
                    replace = bool(pending) and not give_up
                    if give_up:
                        action = f"; giving up after {self.futile} failed workers in a row"
                    else:
                        verbs = ["killed"] * alive + ["replaced"] * replace
                        action = f"; the worker is {' and '.join(verbs)}" if verbs else ""
                    report(f"[{label}] {reason}{action}")
                    if task is not None:
                        self.finished += 1
                        yield Failed(items[task], reason)
                    if give_up:
                        self.finished += len(pending)
                        for index in pending:
                            yield Failed(items[index], f"workers keep failing: {reason}")
                        pending.clear()
                    elif replace:
                        survivors.append(self._start())
                workers = survivors
                if now - last_heartbeat >= watchdog.heartbeat:
                    running = sum(worker.task is not None for worker in workers)
                    report(
                        f"[{label}] heartbeat: {self.finished}/{len(items)} done, "
                        f"{running} running, {(now - started) / 60:.1f} min"
                    )
                    last_heartbeat = now
        finally:
            for worker in workers:
                worker.stop()
            for worker in workers:
                worker.process.join(timeout=20)
                if worker.process.is_alive():
                    worker.kill()
                else:
                    worker.close()
