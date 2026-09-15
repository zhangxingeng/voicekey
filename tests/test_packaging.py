"""Guarantees the packaging story depends on.

tkinter was chosen over GTK4 specifically because uv's managed Python ships Tk
on every platform, so users install nothing. That only holds for *managed*
builds -- a system interpreter satisfies `requires-python` equally well and may
have no Tk (Homebrew's python@3.14 splits it into a `python-tk` formula). This
test is what catches the environment drifting off that guarantee.
"""


def test_tkinter_is_available():
    import tkinter  # noqa: F401


def test_tk_is_new_enough_for_the_ui():
    import tkinter

    # Tk 8.6 works but looks dated; 9.0 is what the managed builds provide and
    # what the UI's sizing and fonts were tuned against.
    assert tkinter.TkVersion >= 8.6


def test_onnxruntime_is_importable():
    import onnxruntime as ort

    assert "CPUExecutionProvider" in ort.get_available_providers()
