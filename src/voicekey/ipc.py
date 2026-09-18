"""A unix socket so the hotkey can reach the running app.

Wayland gives an ordinary client no way to grab a global hotkey. The route
that works on GNOME without root, an extension, or an input-group membership
is: register a custom keybinding that runs a command, and have that command
talk to the already-running app. This is that channel.

It has to be a socket rather than a signal or a pidfile because the reply
matters -- the launcher needs to know whether anyone was listening, so it can
tell the user the app is not running instead of failing silently.

Keep the protocol trivial: one line of UTF-8 in, one line out. There is no
version negotiation because both ends ship in the same binary.
"""

from __future__ import annotations

import contextlib
import os
import socket
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

# Commands the server understands. Anything else gets ERROR back.
TOGGLE = "toggle"
PING = "ping"

OK = "ok"
ERROR = "error"

# Given a command line, return the reply line. Runs on the server's thread.
CommandFn = Callable[[str], str]

CONNECT_TIMEOUT_S = 2.0

# How often the accept loop surfaces to check whether it should stop. Quitting
# should feel instant, and an idle poll this cheap is not worth measuring.
_ACCEPT_POLL_S = 0.05

# Generous next to the poll interval: if the thread has not gone by now it is
# wedged, and blocking the UI thread any longer helps nobody.
_JOIN_TIMEOUT_S = 1.0


def socket_path() -> Path:
    """Where the socket lives.

    $XDG_RUNTIME_DIR is the correct home for this: it is user-private, on
    tmpfs, and cleaned up at logout, so a stale socket cannot outlive the
    session. Falls back to /tmp only when it is unset, which happens in
    containers and over bare ssh.
    """
    xdg_runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if xdg_runtime_dir:
        return Path(xdg_runtime_dir) / "voicekey.sock"
    return Path(tempfile.gettempdir()) / "voicekey.sock"


class Server:
    """Listens for commands from the hotkey launcher.

    Accepts on a background thread so the caller's main loop (Tk) is free.
    """

    def __init__(self, on_command: CommandFn, *, path: Path | None = None) -> None:
        self._on_command = on_command
        self._path = path or socket_path()
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Bind, listen, and begin accepting. Idempotent.

        A socket file left behind by a crashed process must not stop this from
        starting: if the path exists but nothing answers, it is stale and gets
        removed. If something *does* answer, another instance is live and this
        must raise rather than steal the name.
        """
        if self._thread is not None:
            return

        # Check for stale socket: try to connect to it.
        if self._path.exists():
            if self._is_socket_alive():
                raise RuntimeError(f"Another instance is running at {self._path}")
            # Socket is stale, remove it.
            self._path.unlink()

        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(str(self._path))
        self._sock.listen(1)
        # Closing a listening socket does not reliably wake a thread already
        # blocked in accept() on Linux, so shutdown has to be polled for
        # instead. Without this, close() waits out its whole join timeout and
        # then leaks the thread.
        self._sock.settimeout(_ACCEPT_POLL_S)

        # Start the accept loop on a daemon thread.
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._accept_loop, daemon=True, name="voicekey-ipc")
        self._thread.start()

    def close(self) -> None:
        """Stop accepting, join the thread, unlink the socket. Idempotent."""
        self._stop_event.set()

        # Joined before the socket is closed, so the loop exits through its own
        # poll rather than racing against a socket disappearing underneath it.
        if self._thread is not None:
            self._thread.join(timeout=_JOIN_TIMEOUT_S)
            self._thread = None

        if self._sock is not None:
            with contextlib.suppress(OSError):
                self._sock.close()
            self._sock = None

        with contextlib.suppress(FileNotFoundError):
            self._path.unlink()

    def _is_socket_alive(self) -> bool:
        """True if anything is listening on the socket.

        Deliberately does not require a well-formed PING reply. A live instance
        whose handler answers differently still owns the name, and unlinking its
        socket because the reply was unfamiliar would cut it off from its hotkey.
        """
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        probe.settimeout(CONNECT_TIMEOUT_S)
        try:
            probe.connect(str(self._path))
            return True
        except OSError:
            return False
        finally:
            probe.close()

    def _accept_loop(self) -> None:
        """Accept connections and dispatch commands until stopped."""
        while not self._stop_event.is_set():
            sock = self._sock
            if sock is None:
                return
            try:
                client, _ = sock.accept()
            except TimeoutError:
                continue  # poll interval elapsed; re-check the stop flag
            except OSError:
                break

            try:
                self._handle_connection(client)
            except Exception:
                # Errors handling one connection must not kill the loop.
                pass
            finally:
                with contextlib.suppress(OSError):
                    client.close()

    def _handle_connection(self, client: socket.socket) -> None:
        """Read one command, call on_command, and write the reply."""
        client.settimeout(CONNECT_TIMEOUT_S)
        with client.makefile("r", encoding="utf-8") as f:
            line = f.readline().rstrip("\n")
            if not line:
                return
        try:
            reply = self._on_command(line)
        except Exception:
            reply = ERROR
        with client.makefile("w", encoding="utf-8") as f:
            f.write(reply + "\n")
            f.flush()


def send(command: str, *, path: Path | None = None) -> str | None:
    """Send one command to a running server and return its reply.

    Returns None when nothing is listening -- the app is not running. That is
    an ordinary outcome, not an error, so it must not raise.
    """
    path = path or socket_path()
    try:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(CONNECT_TIMEOUT_S)
        sock.connect(str(path))
        try:
            with sock.makefile("w", encoding="utf-8") as f:
                f.write(command + "\n")
                f.flush()
            with sock.makefile("r", encoding="utf-8") as f:
                reply = f.readline().rstrip("\n")
            return reply if reply else None
        finally:
            sock.close()
    except (ConnectionRefusedError, FileNotFoundError):
        # App is not running; return None rather than raising.
        return None


def is_running(*, path: Path | None = None) -> bool:
    """True when a server answers PING."""
    return send(PING, path=path) == OK


__all__ = [
    "CONNECT_TIMEOUT_S",
    "ERROR",
    "OK",
    "PING",
    "TOGGLE",
    "CommandFn",
    "Server",
    "is_running",
    "send",
    "socket_path",
]
