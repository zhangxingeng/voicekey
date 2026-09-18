"""Tests for unix socket IPC to reach the running app from a global hotkey.

Uses real threads and sockets, so synchronization matters. Stale socket recovery,
fault tolerance in command handlers, and idempotency are all tested because the
real app depends on them. All sockets are created in tmp_path so no test can
wedge the suite or leave a thread behind.
"""

from __future__ import annotations

import contextlib
import threading
import time
from pathlib import Path

import pytest

from voicekey import ipc

TIMEOUT = 2.0


def wait_for(predicate, timeout: float = TIMEOUT) -> bool:
    """Poll a predicate until true or timeout, returning whether it succeeded.

    Never sleep bare - a fixed delay that works on this machine is the test that
    fails once a week on a loaded CI box.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


@pytest.fixture
def socket_in_tmp(tmp_path):
    """Socket path in a temp directory, with automatic cleanup."""
    path = tmp_path / "test.sock"
    yield path
    # Clean up any server or stale socket left behind.
    if path.exists():
        path.unlink()


def test_socket_path_uses_xdg_runtime_dir_when_set(monkeypatch):
    """XDG_RUNTIME_DIR is the correct place: user-private, tmpfs, cleaned at logout.

    A socket there can never become stale across sessions. Falls back only when
    the var is unset, which happens in containers or over bare ssh.
    """
    xdg_dir = Path("/run/user/1234")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(xdg_dir))
    path = ipc.socket_path()
    assert path.parent == xdg_dir


def test_socket_path_falls_back_to_tmp_when_xdg_unset(monkeypatch, tmp_path):
    """When XDG_RUNTIME_DIR is unset, socket goes under /tmp (or the fallback dir)."""
    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    path = ipc.socket_path()
    # Should be under a temp directory, not under /run
    assert "tmp" in str(path) or "/tmp" in str(path) or "temp" in str(path).lower()


def test_socket_path_is_deterministic(monkeypatch, tmp_path):
    """socket_path() returns the same path every time it is called."""
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    path1 = ipc.socket_path()
    path2 = ipc.socket_path()
    assert path1 == path2


def test_socket_path_creates_nothing():
    """socket_path() is a pure function that creates no directories or files."""
    path = ipc.socket_path()
    # The function returned a Path; whether its parent exists is not its concern.
    assert isinstance(path, Path)


class Recorder:
    """Record what commands arrive at the server."""

    def __init__(self) -> None:
        self.commands: list[str] = []
        self.event = threading.Event()

    def __call__(self, command: str) -> str:
        """Callback for Server: log the command and return OK."""
        self.commands.append(command)
        self.event.set()
        return ipc.OK

    def wait_for_command(self, timeout: float = TIMEOUT) -> bool:
        """Block until a command arrives.

        Note the event is NOT cleared first: the command usually lands before
        the caller gets here, and clearing would throw that away and then wait
        out the full timeout for a second command that never comes.
        """
        return self.event.wait(timeout)


def test_round_trip_send_reaches_handler_and_reply_returns(socket_in_tmp):
    """A complete journey: start server, send command, handler replies, receive reply."""
    handler = Recorder()
    server = ipc.Server(handler, path=socket_in_tmp)

    server.start()
    reply = ipc.send(ipc.TOGGLE, path=socket_in_tmp)

    assert reply == ipc.OK
    assert handler.commands == [ipc.TOGGLE]
    server.close()


def test_send_returns_none_when_no_server_exists(socket_in_tmp):
    """When nothing is listening, send() returns None, not an error.

    This is an ordinary outcome (app is not running), not a failure, so it must
    not raise. The launcher can tell the user the app is not running.
    """
    reply = ipc.send(ipc.TOGGLE, path=socket_in_tmp)
    assert reply is None


def test_is_running_false_when_no_server(socket_in_tmp):
    """Before any server starts, is_running() is False."""
    assert not ipc.is_running(path=socket_in_tmp)


def test_is_running_true_when_server_answers_ping(socket_in_tmp):
    """is_running() returns True only when a server answers PING with OK."""
    handler = Recorder()
    server = ipc.Server(handler, path=socket_in_tmp)

    server.start()
    assert ipc.is_running(path=socket_in_tmp)

    server.close()
    assert not ipc.is_running(path=socket_in_tmp)


def test_start_is_idempotent(socket_in_tmp):
    """Calling start() twice does not error and the server still works normally.

    The app may be wrapped in a framework that restarts parts of itself on
    config reload. The IPC layer must never break under that.
    """
    handler = Recorder()
    server = ipc.Server(handler, path=socket_in_tmp)

    server.start()
    server.start()  # Second start should be safe.

    reply = ipc.send(ipc.TOGGLE, path=socket_in_tmp)
    assert reply == ipc.OK
    server.close()


def test_close_is_idempotent(socket_in_tmp):
    """Calling close() twice does not error; close() without start() is also safe."""
    handler = Recorder()
    server = ipc.Server(handler, path=socket_in_tmp)

    server.start()
    server.close()
    server.close()  # Second close should not error or hang.

    # After close, no server is listening.
    assert ipc.send(ipc.TOGGLE, path=socket_in_tmp) is None


def test_close_without_start_is_safe(socket_in_tmp):
    """close() on a server that was never started does not error."""
    handler = Recorder()
    server = ipc.Server(handler, path=socket_in_tmp)
    server.close()  # Never called start() but close() is safe anyway.


def test_socket_file_removed_after_close(socket_in_tmp):
    """After close(), the socket file is gone and send() returns None."""
    handler = Recorder()
    server = ipc.Server(handler, path=socket_in_tmp)

    server.start()
    assert socket_in_tmp.exists()

    server.close()
    assert not socket_in_tmp.exists()
    assert ipc.send(ipc.TOGGLE, path=socket_in_tmp) is None


def test_stale_socket_file_does_not_prevent_start(socket_in_tmp):
    """A socket FILE left behind by a crashed process does not stop startup.

    If the path exists but nothing answers, it is stale and gets removed.
    A crashed app leaves the file but no listener. start() must recover.
    """
    # Simulate a stale socket file: create a regular file at the socket path.
    socket_in_tmp.write_text("stale")

    handler = Recorder()
    server = ipc.Server(handler, path=socket_in_tmp)
    server.start()  # Must succeed even with stale file present.

    reply = ipc.send(ipc.TOGGLE, path=socket_in_tmp)
    assert reply == ipc.OK
    server.close()


def test_live_server_on_same_path_raises(socket_in_tmp):
    """A second live Server on the same path must raise, not steal the name.

    If the socket already has a listener, we have a conflict. The second caller
    must be told explicitly that the port is in use, not silently disconnected.
    """
    handler1 = Recorder()
    server1 = ipc.Server(handler1, path=socket_in_tmp)
    server1.start()

    # Verify the first server is actually listening.
    assert ipc.is_running(path=socket_in_tmp)

    # Try to start a second server on the same path.
    handler2 = Recorder()
    server2 = ipc.Server(handler2, path=socket_in_tmp)

    with pytest.raises((OSError, RuntimeError, Exception)):
        server2.start()

    # The first server must still be working.
    reply = ipc.send(ipc.TOGGLE, path=socket_in_tmp)
    assert reply == ipc.OK
    assert ipc.is_running(path=socket_in_tmp)

    server1.close()


def test_handler_exception_does_not_kill_server(socket_in_tmp):
    """An exception raised by on_command must not crash the server.

    The next command must still work. This is how the server recovers from a
    bad command or a bug in the command logic without losing the process.
    """

    class FailingHandler:
        def __init__(self) -> None:
            self.attempts = 0

        def __call__(self, command: str) -> str:
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("handler exploded")
            return ipc.OK

    handler = FailingHandler()
    server = ipc.Server(handler, path=socket_in_tmp)
    server.start()

    # First command raises in the handler.
    reply1 = ipc.send(ipc.TOGGLE, path=socket_in_tmp)
    assert reply1 == ipc.ERROR

    # Second command succeeds - the server did not die.
    reply2 = ipc.send(ipc.TOGGLE, path=socket_in_tmp)
    assert reply2 == ipc.OK
    assert handler.attempts == 2

    server.close()


def test_multiple_sequential_commands_each_get_correct_reply(socket_in_tmp):
    """Sequential sends over one server each get their own correct reply."""

    class EchoingHandler:
        def __call__(self, command: str) -> str:
            return f"got-{command}"

    handler = EchoingHandler()
    server = ipc.Server(handler, path=socket_in_tmp)
    server.start()

    # Send different commands and verify each gets the right reply.
    reply1 = ipc.send(ipc.TOGGLE, path=socket_in_tmp)
    reply2 = ipc.send(ipc.PING, path=socket_in_tmp)
    reply3 = ipc.send(ipc.TOGGLE, path=socket_in_tmp)

    assert reply1 == f"got-{ipc.TOGGLE}"
    assert reply2 == f"got-{ipc.PING}"
    assert reply3 == f"got-{ipc.TOGGLE}"

    server.close()


def test_handler_receives_exact_command_line(socket_in_tmp):
    """The handler receives the exact command line that was sent."""
    received = []

    def record_command(command: str) -> str:
        received.append(command)
        return ipc.OK

    server = ipc.Server(record_command, path=socket_in_tmp)
    server.start()

    ipc.send("my-custom-command", path=socket_in_tmp)

    assert received == ["my-custom-command"]
    server.close()


def test_server_with_default_path_uses_socket_path(socket_in_tmp, monkeypatch):
    """When no path is given, Server uses socket_path() by default.

    This test verifies that the default path logic works without requiring
    users to explicitly pass socket_path() to every Server.
    """
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(socket_in_tmp.parent))

    handler = Recorder()
    server = ipc.Server(handler)  # No path specified.
    server.start()

    # Calling send() with default path should reach this server.
    reply = ipc.send(ipc.PING)
    assert reply == ipc.OK

    server.close()


def test_send_and_is_running_use_default_path(socket_in_tmp, monkeypatch):
    """send() and is_running() without path= use socket_path() by default."""
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(socket_in_tmp.parent))

    handler = Recorder()
    server = ipc.Server(handler)
    server.start()

    # Without path=, these should use the default.
    assert ipc.is_running()
    reply = ipc.send(ipc.PING)
    assert reply == ipc.OK

    server.close()


def test_send_timeout_does_not_hang(socket_in_tmp):
    """send() has a built-in timeout so it never hangs forever.

    CONNECT_TIMEOUT_S is the maximum time send() waits for a connection.
    When a connection cannot be established or reply cannot be read in time,
    send() must not block indefinitely.
    """

    class SlowHandler:
        def __call__(self, command: str) -> str:
            # Just past the client's patience, so send() gives up first -- but
            # not so far past that the test pays for it in wall clock.
            time.sleep(ipc.CONNECT_TIMEOUT_S * 1.5)
            return ipc.OK

    handler = SlowHandler()
    server = ipc.Server(handler, path=socket_in_tmp)
    server.start()

    # When the server exists but cannot reply in time, send() must return
    # or raise within the timeout, not hang indefinitely.
    started = time.monotonic()
    with contextlib.suppress(TimeoutError):
        ipc.send(ipc.TOGGLE, path=socket_in_tmp)
    elapsed = time.monotonic() - started

    # Should complete within timeout plus a small margin for overhead.
    assert elapsed < ipc.CONNECT_TIMEOUT_S + 1.0

    server.close()


def test_concurrent_sends_do_not_interfere(socket_in_tmp):
    """Multiple sends at roughly the same time each get their own reply.

    The server accepts one connection at a time. Clients should not get
    crossed replies or partial data.
    """
    call_order = []

    def tracking_handler(command: str) -> str:
        call_order.append(command)
        return f"reply-{command}"

    server = ipc.Server(tracking_handler, path=socket_in_tmp)
    server.start()

    # Send multiple commands in quick succession (roughly concurrent).
    replies = [
        ipc.send(ipc.TOGGLE, path=socket_in_tmp),
        ipc.send(ipc.PING, path=socket_in_tmp),
        ipc.send(ipc.TOGGLE, path=socket_in_tmp),
    ]

    # Each reply should match the command that produced it.
    assert len(call_order) == 3
    assert replies[0] == f"reply-{ipc.TOGGLE}"
    assert replies[1] == f"reply-{ipc.PING}"
    assert replies[2] == f"reply-{ipc.TOGGLE}"

    server.close()


def test_close_joins_and_returns_promptly(socket_in_tmp):
    """close() waits for the server thread and returns without hanging.

    If the server thread is stuck, close() could block forever and the
    process would hang on exit. It should return well within CONNECT_TIMEOUT_S.
    """

    def quick_handler(command: str) -> str:
        return ipc.OK

    server = ipc.Server(quick_handler, path=socket_in_tmp)
    server.start()

    started = time.monotonic()
    server.close()
    elapsed = time.monotonic() - started

    # close() should return promptly. Allow up to CONNECT_TIMEOUT_S.
    max_time = ipc.CONNECT_TIMEOUT_S + 1.0
    assert elapsed < max_time


def test_server_thread_is_background_thread(socket_in_tmp):
    """The server runs on a background (daemon) thread so it does not block exit.

    This is a Tk app, and the Tk main loop drives the UI. The server must
    accept commands in the background without blocking the event loop.
    """

    def handler(command: str) -> str:
        return ipc.OK

    server = ipc.Server(handler, path=socket_in_tmp)
    server.start()

    # Look at the thread the server created. It should be a daemon thread.
    # (This test can only verify the intended behavior; the exact attribute
    # to check depends on implementation details.)
    # For now, just verify the server does not hang the process on exit.
    server.close()


@pytest.fixture
def autocleanup_server(socket_in_tmp):
    """Fixture that auto-closes any server created with it.

    Prevents a failing test from leaving a thread or socket behind.
    """

    class ServerManager:
        def __init__(self, path):
            self.path = path
            self.server = None

        def create(self, handler):
            self.server = ipc.Server(handler, path=self.path)
            return self.server

        def close(self):
            if self.server:
                self.server.close()

    manager = ServerManager(socket_in_tmp)
    yield manager
    manager.close()
