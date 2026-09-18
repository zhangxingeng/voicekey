"""Build the frozen, zero-dependency artifact.

This exists because the plain PyInstaller invocation produces a binary that
looks fine and dies on startup.

uv's managed CPython ships `_tkinter.so` with no RPATH or RUNPATH, so nothing
can resolve `libtcl9.0.so` / `libtcl9tk9.0.so` by the normal rules -- `ldd`
reports them as "not found" even against the original interpreter, which works
only because the `python` executable itself carries an RPATH that covers the
whole dependency chain. PyInstaller's scanner follows the normal rules, finds
neither library, and silently ships without them. The bundle then contains all
the Tcl/Tk *data* files and none of the code, and the first `import tkinter`
fails with ImportError at startup.

Since tkinter was chosen precisely because uv guarantees Tk on every platform,
a frozen build that cannot import it defeats the whole packaging story. So the
libraries are located here and passed in explicitly.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ENTRY = Path("src/voicekey/__main__.py")
NAME = "voicekey"

# Covers both the interpreter library and the Tk library, whose names begin
# with the same prefix in Tcl 9 (libtcl9.0.so and libtcl9tk9.0.so). Windows
# keeps its DLLs where PyInstaller already looks, so it matches nothing there
# and needs nothing.
_PATTERNS = ("libtcl*.so*", "libtk*.so*", "libtcl*.dylib", "libtk*.dylib")


def tcl_tk_libraries(base: Path | None = None) -> list[Path]:
    """Every Tcl/Tk shared library shipped with this interpreter."""
    lib_dir = (base or Path(sys.base_prefix)) / "lib"
    if not lib_dir.is_dir():
        return []
    found: set[Path] = set()
    for pattern in _PATTERNS:
        # Only the top level: the nested itcl/thread packages are extensions
        # that Tcl loads by path at runtime, not link-time dependencies.
        found.update(p for p in lib_dir.glob(pattern) if p.is_file())
    return sorted(found)


def command() -> list[str]:
    argv = [
        "pyinstaller",
        "--noconfirm",
        "--clean",
        "--name",
        NAME,
        "--windowed",
        # sounddevice loads PortAudio through ctypes, which no static scan can
        # see, so the hook has to be asked for it explicitly.
        "--collect-binaries",
        "sounddevice",
    ]
    separator = ";" if sys.platform == "win32" else ":"
    for lib in tcl_tk_libraries():
        argv += ["--add-binary", f"{lib}{separator}."]
    argv.append(str(ENTRY))
    return argv


def main() -> int:
    libs = tcl_tk_libraries()
    print(f"Tcl/Tk libraries to bundle: {[p.name for p in libs] or 'none (Windows ships DLLs)'}")
    argv = command()
    print(" ".join(argv))
    return subprocess.call(argv)


if __name__ == "__main__":
    sys.exit(main())
