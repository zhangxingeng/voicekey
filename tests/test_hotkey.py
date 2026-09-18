"""Tests for global hotkey registration with GNOME.

The critical invariant: registering a hotkey must preserve every pre-existing
user binding in the dconf array. A test that clobbers the user's keybindings
would be unforgivable, so these tests use a fake gsettings that never touches
the real dconf database.

GNOME stores custom keybindings as an array of relocatable schema paths:
the array lives in a parent schema under the key `custom-keybindings`, and
each path holds a dict of {name, binding, command}. Registering means reading
the array, finding ours if it exists (by name), or picking the lowest unused
`customN` slot and appending it. Blindly writing `custom0` would silently
destroy the user's own bindings.
"""

from __future__ import annotations

import ast

import pytest

from voicekey import hotkey


class FakeGsettings:
    """In-memory dconf simulator for testing.

    Tracks:
    - The array of custom-keybinding paths (formatted as GVariant string)
    - Each path's {name, binding, command} dict

    Methods behave like `gsettings get/set`, including GVariant quoting.
    """

    def __init__(self) -> None:
        # The array of paths, stored as a list internally
        self.paths: list[str] = []
        # Each path's settings: {path: {name, binding, command}}
        self.bindings: dict[str, dict[str, str]] = {}
        # Track calls for inspection
        self.calls: list[tuple[str, list[str]]] = []

    def __call__(self, argv: list[str]) -> str:
        """Simulate gsettings command."""
        self.calls.append(("gsettings", argv))
        argv = list(argv)

        if not argv:
            raise ValueError("gsettings: no args")

        cmd = argv[0]

        if cmd == "get":
            return self._get(argv[1:])
        elif cmd == "set":
            return self._set(argv[1:])
        else:
            raise ValueError(f"gsettings: unknown command {cmd}")

    def _get(self, args: list[str]) -> str:
        """Handle `gsettings get SCHEMA [KEY]` or `get SCHEMA:PATH KEY`."""
        if len(args) < 2:
            raise ValueError("gsettings get: needs SCHEMA KEY or SCHEMA:PATH KEY")

        schema_or_loc = args[0]
        key = args[1]

        # Check if this is a relocatable schema: SCHEMA:PATH format
        if ":" in schema_or_loc:
            schema, path = schema_or_loc.split(":", 1)
            if schema != hotkey.CHILD_SCHEMA:
                raise ValueError(f"gsettings: invalid relocatable schema {schema}")
            if path not in self.bindings:
                raise ValueError(f"gsettings: no such path {path}")
            binding = self.bindings[path]
            if key == "name":
                return self._quote(binding.get("name", ""))
            elif key == "binding":
                return self._quote(binding.get("binding", ""))
            elif key == "command":
                return self._quote(binding.get("command", ""))
            else:
                raise ValueError(f"gsettings: unknown key {key}")

        # Non-relocatable (parent) schema
        schema = schema_or_loc
        if schema == hotkey.PARENT_SCHEMA and key == "custom-keybindings":
            return self._format_array()

        raise ValueError(f"gsettings get: invalid args {schema} {key}")

    def _set(self, args: list[str]) -> str:
        """Handle `gsettings set SCHEMA KEY VALUE` or `set SCHEMA:PATH KEY VALUE`."""
        if len(args) < 3:
            raise ValueError("gsettings set: needs SCHEMA KEY VALUE or SCHEMA:PATH KEY VALUE")

        schema_or_loc = args[0]
        key = args[1]
        value = args[2]

        # Check if this is a relocatable schema: SCHEMA:PATH format
        if ":" in schema_or_loc:
            schema, path = schema_or_loc.split(":", 1)
            if schema != hotkey.CHILD_SCHEMA:
                raise ValueError(f"gsettings: invalid relocatable schema {schema}")
            if path not in self.bindings:
                self.bindings[path] = {}
            self.bindings[path][key] = value
            return ""

        # Non-relocatable (parent) schema
        schema = schema_or_loc
        if schema == hotkey.PARENT_SCHEMA and key == "custom-keybindings":
            # Parse array from Python repr() format (what implementation uses)
            self.paths = self._parse_python_repr(value)
            return ""

        raise ValueError(f"gsettings set: invalid args {schema} {key}")

    def _format_array(self) -> str:
        """Format the paths array as a GVariant string (gsettings output)."""
        if not self.paths:
            return "@as []\n"
        quoted_paths = [f"'{p}'" for p in self.paths]
        return f"[{', '.join(quoted_paths)}]\n"

    def _parse_python_repr(self, value: str) -> list[str]:
        """Parse a Python repr() format array string into paths.

        The implementation uses repr([...]) for setting arrays.
        """
        try:
            result = ast.literal_eval(value)
            return list(result) if isinstance(result, (list, tuple)) else []
        except (ValueError, SyntaxError):
            return []

    def _quote(self, value: str) -> str:
        """Quote a string for GVariant and add trailing newline."""
        return f"'{value}'\n"

    def _unquote(self, value: str) -> str:
        """Remove quotes from a GVariant string."""
        value = value.strip()
        if value.startswith("'") and value.endswith("'"):
            return value[1:-1]
        return value


@pytest.fixture
def fake_gs():
    """Provide a fresh fake gsettings store for each test."""
    return FakeGsettings()


class TestAvailable:
    """Tests for available() -- detecting GNOME availability."""

    def test_available_returns_true_when_schema_exists(self, fake_gs):
        # A real GNOME session has the schema available.
        def fake_run(argv):
            fake_gs(argv)
            return ""

        assert hotkey.available(run=fake_run) is True

    def test_available_returns_false_on_file_not_found(self, fake_gs):
        # On systems without gsettings (KDE, Wayland, macOS, Windows),
        # the attempt to read the schema raises FileNotFoundError.
        def fake_run(argv):
            raise FileNotFoundError("gsettings: command not found")

        assert hotkey.available(run=fake_run) is False

    def test_available_returns_false_on_no_schema(self, fake_gs):
        # On non-GNOME desktops, the schema does not exist.
        def fake_run(argv):
            raise subprocess.CalledProcessError(
                1, argv, stderr="Schema 'org.gnome.settings-daemon.plugins.media-keys' not found"
            )

        assert hotkey.available(run=fake_run) is False

    def test_available_returns_false_on_generic_error(self, fake_gs):
        # Other errors (permission, dconf corruption) also mean unavailable.
        def fake_run(argv):
            raise RuntimeError("dconf: I/O error")

        assert hotkey.available(run=fake_run) is False


class TestFindSlot:
    """Tests for find_slot() -- locating our existing binding."""

    def test_find_slot_returns_none_when_unregistered(self, fake_gs):
        # If our name is not in any binding, we are not registered.
        fake_gs.paths = []
        assert hotkey.find_slot(run=fake_gs) is None

    def test_find_slot_returns_our_path_when_registered(self, fake_gs):
        # When our name exists in the array, find_slot returns its path.
        path = f"{hotkey.PATH_PREFIX}custom0/"
        fake_gs.paths = [path]
        fake_gs.bindings[path] = {
            "name": hotkey.NAME,
            "binding": hotkey.BINDING,
            "command": "echo test",
        }
        result = hotkey.find_slot(run=fake_gs)
        assert result == path

    def test_find_slot_ignores_other_bindings(self, fake_gs):
        # find_slot searches by our name, ignoring other users' bindings.
        path0 = f"{hotkey.PATH_PREFIX}custom0/"
        path1 = f"{hotkey.PATH_PREFIX}custom1/"
        fake_gs.paths = [path0, path1]
        fake_gs.bindings[path0] = {
            "name": "someone else",
            "binding": "<Control>q",
            "command": "some-app",
        }
        fake_gs.bindings[path1] = {
            "name": hotkey.NAME,
            "binding": hotkey.BINDING,
            "command": "echo test",
        }
        result = hotkey.find_slot(run=fake_gs)
        assert result == path1

    def test_find_slot_returns_none_when_array_has_other_bindings(self, fake_gs):
        # Just having a non-empty array does not mean we are registered.
        path = f"{hotkey.PATH_PREFIX}custom0/"
        fake_gs.paths = [path]
        fake_gs.bindings[path] = {
            "name": "my other binding",
            "binding": "<Control>x",
            "command": "other-cmd",
        }
        result = hotkey.find_slot(run=fake_gs)
        assert result is None


class TestRegister:
    """Tests for register() -- adding or updating our hotkey."""

    def test_register_into_empty_array(self, fake_gs):
        # Starting from nothing, registering should create custom0 and return
        # its path.
        fake_gs.paths = []
        path = hotkey.register("echo hello", run=fake_gs)
        assert path == f"{hotkey.PATH_PREFIX}custom0/"
        assert path in fake_gs.bindings
        assert fake_gs.bindings[path]["name"] == hotkey.NAME
        assert fake_gs.bindings[path]["binding"] == hotkey.BINDING
        assert fake_gs.bindings[path]["command"] == "echo hello"
        assert path in fake_gs.paths

    def test_register_sets_custom_binding(self, fake_gs):
        # When a custom binding is provided, it is used instead of the default.
        fake_gs.paths = []
        custom_binding = "<Control><Shift>x"
        path = hotkey.register("cmd", binding=custom_binding, run=fake_gs)
        assert fake_gs.bindings[path]["binding"] == custom_binding

    def test_register_preserves_preexisting_bindings(self, fake_gs):
        # If the user already has custom0 and custom1, we must append, not
        # overwrite. This is the critical invariant that protects the desktop.
        user0 = f"{hotkey.PATH_PREFIX}custom0/"
        user1 = f"{hotkey.PATH_PREFIX}custom1/"
        fake_gs.paths = [user0, user1]
        fake_gs.bindings[user0] = {
            "name": "user's binding 0",
            "binding": "<Control>a",
            "command": "cmd0",
        }
        fake_gs.bindings[user1] = {
            "name": "user's binding 1",
            "binding": "<Control>b",
            "command": "cmd1",
        }

        path = hotkey.register("echo register", run=fake_gs)

        # Our entry should be at custom2, and the user's should be untouched
        assert path == f"{hotkey.PATH_PREFIX}custom2/"
        assert fake_gs.paths == [user0, user1, path]
        assert fake_gs.bindings[user0] == {
            "name": "user's binding 0",
            "binding": "<Control>a",
            "command": "cmd0",
        }
        assert fake_gs.bindings[user1] == {
            "name": "user's binding 1",
            "binding": "<Control>b",
            "command": "cmd1",
        }

    def test_register_picks_lowest_unused_slot(self, fake_gs):
        # When custom0 and custom2 are taken but custom1 is not, pick custom1.
        # This ensures we don't fragment the namespace.
        path0 = f"{hotkey.PATH_PREFIX}custom0/"
        path2 = f"{hotkey.PATH_PREFIX}custom2/"
        fake_gs.paths = [path0, path2]
        fake_gs.bindings[path0] = {"name": "other0", "binding": "<Control>0", "command": "c0"}
        fake_gs.bindings[path2] = {"name": "other2", "binding": "<Control>2", "command": "c2"}

        path = hotkey.register("echo slot", run=fake_gs)

        assert path == f"{hotkey.PATH_PREFIX}custom1/"
        assert fake_gs.paths == [path0, path2, path]

    def test_register_is_idempotent(self, fake_gs):
        # Calling register twice with the same command should update in place,
        # not append a duplicate. The array gains exactly one entry.
        fake_gs.paths = []

        path1 = hotkey.register("echo first", run=fake_gs)
        assert len(fake_gs.paths) == 1

        path2 = hotkey.register("echo first", run=fake_gs)
        assert path1 == path2
        assert len(fake_gs.paths) == 1

    def test_register_updates_existing_entry_when_called_again(self, fake_gs):
        # Idempotency also means updating the binding if called a second time
        # with a different value.
        fake_gs.paths = []
        path = hotkey.register("echo first", run=fake_gs)
        original_command = fake_gs.bindings[path]["command"]
        assert original_command == "echo first"

        # Call again with different command
        path2 = hotkey.register("echo second", run=fake_gs)
        assert path == path2
        assert fake_gs.bindings[path]["command"] == "echo second"

    def test_register_with_changed_binding_updates_in_place(self, fake_gs):
        # If we re-register with a different binding, update our entry in place.
        fake_gs.paths = []
        path = hotkey.register("cmd", binding="<Control>a", run=fake_gs)
        assert fake_gs.bindings[path]["binding"] == "<Control>a"

        path2 = hotkey.register("cmd", binding="<Control>b", run=fake_gs)
        assert path == path2
        assert fake_gs.bindings[path]["binding"] == "<Control>b"
        assert len(fake_gs.paths) == 1

    def test_register_preserves_other_bindings_on_reregister(self, fake_gs):
        # When re-registering (idempotency), other users' bindings must still
        # be preserved exactly.
        user_path = f"{hotkey.PATH_PREFIX}custom0/"
        our_path = f"{hotkey.PATH_PREFIX}custom1/"
        fake_gs.paths = [user_path, our_path]
        fake_gs.bindings[user_path] = {
            "name": "user binding",
            "binding": "<Control>a",
            "command": "user-cmd",
        }
        fake_gs.bindings[our_path] = {
            "name": hotkey.NAME,
            "binding": hotkey.BINDING,
            "command": "echo v1",
        }

        path = hotkey.register("echo v2", run=fake_gs)

        assert path == our_path
        assert fake_gs.paths == [user_path, our_path]
        assert fake_gs.bindings[user_path] == {
            "name": "user binding",
            "binding": "<Control>a",
            "command": "user-cmd",
        }


class TestUnregister:
    """Tests for unregister() -- removing our binding."""

    def test_unregister_removes_our_entry_and_returns_true(self, fake_gs):
        # When we are registered, unregister removes our path from the array.
        path = f"{hotkey.PATH_PREFIX}custom0/"
        fake_gs.paths = [path]
        fake_gs.bindings[path] = {
            "name": hotkey.NAME,
            "binding": hotkey.BINDING,
            "command": "echo test",
        }

        result = hotkey.unregister(run=fake_gs)

        assert result is True
        assert path not in fake_gs.paths

    def test_unregister_returns_false_when_not_registered(self, fake_gs):
        # If we are not registered, unregister does nothing and returns False.
        fake_gs.paths = []
        result = hotkey.unregister(run=fake_gs)
        assert result is False

    def test_unregister_preserves_other_bindings(self, fake_gs):
        # Unregistering must not touch other entries in the array, and their
        # settings must remain readable.
        user0 = f"{hotkey.PATH_PREFIX}custom0/"
        our = f"{hotkey.PATH_PREFIX}custom1/"
        user2 = f"{hotkey.PATH_PREFIX}custom2/"

        fake_gs.paths = [user0, our, user2]
        fake_gs.bindings[user0] = {"name": "user0", "binding": "<Control>0", "command": "c0"}
        fake_gs.bindings[our] = {
            "name": hotkey.NAME,
            "binding": hotkey.BINDING,
            "command": "echo test",
        }
        fake_gs.bindings[user2] = {"name": "user2", "binding": "<Control>2", "command": "c2"}

        result = hotkey.unregister(run=fake_gs)

        assert result is True
        # Our path is removed from the array
        assert fake_gs.paths == [user0, user2]
        # Other paths remain unchanged and readable
        assert user0 in fake_gs.bindings
        assert user2 in fake_gs.bindings
        assert fake_gs.bindings[user0] == {
            "name": "user0",
            "binding": "<Control>0",
            "command": "c0",
        }
        assert fake_gs.bindings[user2] == {
            "name": "user2",
            "binding": "<Control>2",
            "command": "c2",
        }

    def test_unregister_is_idempotent(self, fake_gs):
        # Calling unregister twice is safe and idempotent.
        path = f"{hotkey.PATH_PREFIX}custom0/"
        fake_gs.paths = [path]
        fake_gs.bindings[path] = {
            "name": hotkey.NAME,
            "binding": hotkey.BINDING,
            "command": "echo test",
        }

        result1 = hotkey.unregister(run=fake_gs)
        assert result1 is True

        result2 = hotkey.unregister(run=fake_gs)
        assert result2 is False


class TestGVariantFormatting:
    """Tests that the fake gsettings correctly formats output like the real thing.

    This validates the fake implementation matches real gsettings behavior.
    """

    def test_empty_array_returns_at_as_format(self, fake_gs):
        # gsettings returns empty arrays in @as [] format with trailing newline.
        fake_gs.paths = []
        result = fake_gs._format_array()
        assert result == "@as []\n"

    def test_string_values_are_quoted_with_newline(self, fake_gs):
        # gsettings returns string values with surrounding quotes and newline.
        fake_gs.bindings["/path/"] = {"name": "test name"}
        # Call via the public _get interface using SCHEMA:PATH format
        output = fake_gs._get(
            ["org.gnome.settings-daemon.plugins.media-keys.custom-keybinding:/path/", "name"]
        )
        assert output == "'test name'\n"


class TestConstants:
    """Tests that the constants are correctly defined."""

    def test_binding_is_ctrl_shift_d(self):
        assert hotkey.BINDING == "<Control><Shift>d"

    def test_name_is_voicekey_dictation(self):
        assert hotkey.NAME == "voicekey dictation"

    def test_parent_schema(self):
        assert hotkey.PARENT_SCHEMA == "org.gnome.settings-daemon.plugins.media-keys"

    def test_child_schema(self):
        assert (
            hotkey.CHILD_SCHEMA == "org.gnome.settings-daemon.plugins.media-keys.custom-keybinding"
        )

    def test_path_prefix(self):
        assert (
            hotkey.PATH_PREFIX
            == "/org/gnome/settings-daemon/plugins/media-keys/custom-keybindings/"
        )


# Import subprocess for the FileNotFoundError test
import subprocess  # noqa: E402
