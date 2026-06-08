#!/usr/bin/env python3
"""Driver for the dflash daemon stdin/stdout protocol with real timeouts.

A blocking `proc.stdout.readline()` returns only on a new line or on EOF (daemon
death). If the daemon goes silent (stalls during model load, deadlocks, disk
stall) it returns neither, so a `time.time()` check placed after `readline()`
never runs and the caller hangs forever. That matters for unattended calibration
and validation runs.

Here a reader thread pumps stdout lines into a queue, so the main loop waits with
`queue.get(timeout=...)`: a read can never outlive its deadline. (A `select()` on
the fd would be subtly wrong because data already pulled into Python's text
buffer is invisible to `select`; the reader thread does the blocking `readline`
itself, which handles buffering correctly.)
"""
import queue
import subprocess
import threading
import time


class DaemonTimeout(Exception):
    """A stdout read exceeded its deadline (daemon alive but silent)."""


class Daemon:
    def __init__(self, cmd, env, capture_stderr=False):
        self.proc = subprocess.Popen(
            cmd, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=(subprocess.PIPE if capture_stderr else subprocess.DEVNULL),
            bufsize=1, universal_newlines=True)
        self.stderr_lines = []
        self._q = queue.Queue()
        threading.Thread(target=self._pump_stdout, daemon=True).start()
        if capture_stderr:
            threading.Thread(target=self._pump_stderr, daemon=True).start()

    def _pump_stdout(self):
        for line in self.proc.stdout:
            self._q.put(line)
        self._q.put(None)  # EOF sentinel

    def _pump_stderr(self):
        for line in self.proc.stderr:
            self.stderr_lines.append(line.rstrip())

    def readline(self, timeout):
        """Next stdout line (str), or None on EOF. Raises DaemonTimeout past `timeout` s."""
        try:
            return self._q.get(timeout=max(0.0, timeout))
        except queue.Empty:
            raise DaemonTimeout

    def send(self, cmd):
        self.proc.stdin.write(cmd if cmd.endswith("\n") else cmd + "\n")
        self.proc.stdin.flush()

    def wait_ready(self, banner="-daemon] ready", timeout=600, on_line=None):
        """Block until the ready banner. Kills + raises SystemExit on timeout/death."""
        deadline = time.time() + timeout
        while True:
            try:
                line = self.readline(deadline - time.time())
            except DaemonTimeout:
                self.kill()
                raise SystemExit(f"daemon did not become ready within {timeout}s (no output)")
            if line is None:
                raise SystemExit("daemon died before ready banner")
            if on_line:
                on_line(line)
            if banner in line:
                return

    def request(self, cmd, timeout=120):
        """Send a command and wait for its `ok ...`/`err ...` reply.

        Returns the reply line, or None if the daemon died or stalled past
        `timeout` (in which case the daemon is killed so the caller can stop)."""
        try:
            self.send(cmd)
        except (BrokenPipeError, OSError):
            return None
        deadline = time.time() + timeout
        while True:
            try:
                line = self.readline(deadline - time.time())
            except DaemonTimeout:
                self.kill()
                return None
            if line is None:
                return None
            if line.startswith("ok ") or line.startswith("err "):
                return line

    def kill(self):
        try:
            self.proc.kill()
        except Exception:
            pass

    def quit(self, timeout=20):
        try:
            self.send("quit")
            self.proc.wait(timeout=timeout)
        except Exception:
            self.kill()
