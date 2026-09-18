"""Guards on the frozen build.

The failure these exist to prevent is specific and was hit for real: the build
succeeded, CI went green, the artifact was the right size, and the binary died
on its first `import tkinter` because PyInstaller had shipped every Tcl/Tk data
file and neither shared library. Nothing short of running the artifact caught
it, so these at least hold the part that can be checked cheaply.
"""

from __future__ import annotations

import sys

from tools.freeze import command, tcl_tk_libraries


def test_finds_the_tk_libraries_on_this_interpreter():
    """uv's managed Python is the one that guarantees Tk, so it must have them.

    If this fails, either the interpreter is not a managed uv build (see
    python-preference in pyproject.toml) or the library layout changed.
    """
    if sys.platform == "win32":
        return  # Windows ships DLLs where PyInstaller already looks
    names = [p.name for p in tcl_tk_libraries()]
    assert any("tcl" in n for n in names), names
    assert any("tk" in n for n in names), names


def test_returns_nothing_for_an_interpreter_without_a_lib_dir(tmp_path):
    assert tcl_tk_libraries(tmp_path) == []


def test_command_passes_every_library_to_pyinstaller():
    argv = command()
    added = [argv[i + 1] for i, a in enumerate(argv) if a == "--add-binary"]
    assert len(added) == len(tcl_tk_libraries())


def test_command_collects_portaudio():
    # sounddevice reaches PortAudio through ctypes, which no static scan sees.
    argv = command()
    assert "--collect-binaries" in argv
    assert "sounddevice" in argv
