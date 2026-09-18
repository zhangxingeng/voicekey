"""Registering the global hotkey with GNOME.

Wayland has no global-hotkey protocol for ordinary clients, and X11-era
grabbing does not work in a Wayland session. What does work, verified on
GNOME Shell 50.1, is a custom keybinding in gsettings: GNOME owns the grab and
runs a command when the key fires. No root, no extension, no `input` group,
nothing installed.

The awkward part is that custom keybindings are stored as an array of dconf
paths under arbitrary `customN` names, with the settings for each living at
that path under a *relocatable* schema. So registering means: read the array,
find ours if it is already there, otherwise pick an unused slot and append.
Blindly writing `custom0` would silently destroy whatever binding the user
already had there.

Everything shells out to `gsettings` rather than importing Gio, because
PyGObject is exactly the system dependency this app refuses to require.
"""

from __future__ import annotations

import ast
import re
import subprocess
from collections.abc import Callable, Sequence

# The key. Ctrl+Shift+D is free in GNOME's own media-keys on a stock Ubuntu
# install (checked), though a *global* grab does shadow the app underneath --
# in browsers this is "bookmark all tabs", in VS Code the debug panel.
BINDING = "<Control><Shift>d"

# How the binding identifies itself in GNOME's Settings UI, and how we find
# our own entry again among the user's other bindings.
NAME = "voicekey dictation"

PARENT_SCHEMA = "org.gnome.settings-daemon.plugins.media-keys"
CHILD_SCHEMA = "org.gnome.settings-daemon.plugins.media-keys.custom-keybinding"
PATH_PREFIX = "/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/"

# Runs a command and returns its stdout. Injected so tests never touch the
# real dconf database -- a test that rewrites the user's keybindings would be
# an unforgivable thing to ship.
RunFn = Callable[[Sequence[str]], str]


def _parse_gvariant_array(value: str) -> list[str]:
    """Parse a GVariant array string into a list of strings.

    Handles @as [] (type annotation form), [], and ['a', 'b'] (normal form).
    """
    value = value.strip()
    if value.startswith("@as "):
        value = value[4:].strip()
    if value == "[]":
        return []
    try:
        parsed = ast.literal_eval(value)
        return list(parsed) if isinstance(parsed, (list, tuple)) else []
    except (ValueError, SyntaxError):
        return []


def _unquote_gvariant(value: str) -> str:
    """Strip GVariant quoting and trailing newline from a value.

    gsettings returns strings with surrounding single quotes and trailing
    newline, like 'voicekey dictation' followed by \n.
    """
    value = value.rstrip("\n")
    if value.startswith("'") and value.endswith("'"):
        value = value[1:-1]
    return value


def run_gsettings(argv: Sequence[str]) -> str:
    """Execute `gsettings` and return stdout. The default runner."""
    result = subprocess.run(
        ["gsettings", *argv],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"gsettings failed: {result.stderr.strip()}")
    return result.stdout


def available(*, run: RunFn = run_gsettings) -> bool:
    """True when this looks like a GNOME session we can register with.

    False is not an error: on KDE, sway, macOS or Windows the app still works,
    it just has no global hotkey and the UI should say so rather than pretend.
    """
    try:
        run(["get", PARENT_SCHEMA, "custom-keybindings"])
        return True
    except Exception:
        # Catching broadly here: schema missing, gsettings not found, dconf service
        # unavailable. All mean this is not a GNOME session we can register with.
        return False


def find_slot(*, run: RunFn = run_gsettings) -> str | None:
    """Return the dconf path of our existing binding, or None if unregistered."""
    try:
        array_str = run(["get", PARENT_SCHEMA, "custom-keybindings"])
        paths = _parse_gvariant_array(array_str)
        for path in paths:
            try:
                name_str = run(["get", f"{CHILD_SCHEMA}:{path}", "name"])
                name = _unquote_gvariant(name_str)
                if name == NAME:
                    return path
            except RuntimeError:
                # This path is missing or unreadable; skip it.
                continue
        return None
    except RuntimeError:
        return None


def register(command: str, *, binding: str = BINDING, run: RunFn = run_gsettings) -> str:
    """Point `binding` at `command`, and return the dconf path used.

    Idempotent: registering twice updates our entry in place rather than
    appending a duplicate. Other users' bindings in the array are preserved
    exactly -- this reads the existing array and appends, never overwrites.
    """
    # Check if we are already registered; if so, update in place.
    path = find_slot(run=run)
    if path:
        run(["set", f"{CHILD_SCHEMA}:{path}", "name", NAME])
        run(["set", f"{CHILD_SCHEMA}:{path}", "binding", binding])
        run(["set", f"{CHILD_SCHEMA}:{path}", "command", command])
        return path

    # Not registered yet: find the lowest unused customN slot.
    array_str = run(["get", PARENT_SCHEMA, "custom-keybindings"])
    paths = _parse_gvariant_array(array_str)

    # Extract which customN slots are in use by parsing the path suffix.
    used_slots = set()
    for p in paths:
        match = re.search(r"custom(\d+)", p)
        if match:
            used_slots.add(int(match.group(1)))

    # Find the lowest unused slot.
    slot = 0
    while slot in used_slots:
        slot += 1

    # Build the new path and append it to the array.
    new_path = f"{PATH_PREFIX}custom{slot}/"
    new_paths = [*paths, new_path]

    # Write the updated array back, preserving all existing entries.
    array_repr = repr(new_paths)
    run(["set", PARENT_SCHEMA, "custom-keybindings", array_repr])

    # Set the binding's properties at its new path.
    run(["set", f"{CHILD_SCHEMA}:{new_path}", "name", NAME])
    run(["set", f"{CHILD_SCHEMA}:{new_path}", "binding", binding])
    run(["set", f"{CHILD_SCHEMA}:{new_path}", "command", command])

    return new_path


def unregister(*, run: RunFn = run_gsettings) -> bool:
    """Remove our binding from the array. True if one was there to remove."""
    path = find_slot(run=run)
    if not path:
        return False

    # Read the array, remove our path, and write it back. All other entries
    # are preserved in their original order.
    array_str = run(["get", PARENT_SCHEMA, "custom-keybindings"])
    paths = _parse_gvariant_array(array_str)
    new_paths = [p for p in paths if p != path]

    array_repr = repr(new_paths)
    run(["set", PARENT_SCHEMA, "custom-keybindings", array_repr])

    return True


__all__ = [
    "BINDING",
    "CHILD_SCHEMA",
    "NAME",
    "PARENT_SCHEMA",
    "PATH_PREFIX",
    "RunFn",
    "available",
    "find_slot",
    "register",
    "run_gsettings",
    "unregister",
]
