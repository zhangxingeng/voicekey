"""The CUDA preload must be harmless everywhere.

It runs on machines with no GPU, no NVIDIA wheels, and on macOS/Windows, so the
only universal guarantees are: it never raises, and it never reports loading
something it did not load.
"""

from voicekey import cuda


def test_preload_never_raises():
    cuda.preload()


def test_preload_is_idempotent():
    # Second call short-circuits; dlopen'ing the same libs twice would be
    # harmless but pointless, and the guard is what makes it cheap to call
    # from every session construction.
    cuda.preload()
    assert cuda.preload() == []


def test_reported_libraries_look_like_libraries():
    cuda._done = False
    try:
        for name in cuda.preload():
            assert ".so" in name
    finally:
        cuda._done = True


def test_lib_dir_scan_only_returns_directories():
    assert all(p.is_dir() for p in cuda._nvidia_lib_dirs())
