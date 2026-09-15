"""Loopback filtering.

A monitor source records system *output*. If one is picked as the microphone
the app records the speakers and transcribes whatever was playing -- which
fails silently, with plausible-looking text. These names are the real spellings
observed on a PipeWire/PulseAudio desktop.
"""

import pytest

from voicekey.audio import _is_loopback

LOOPBACK = [
    # PortAudio/PipeWire node names -- the `.monitor` suffix form.
    "alsa_output.usb-GeneralPlus_USB_Audio_Device-00.analog-stereo.monitor",
    "alsa_output.pci-0000_6a_00.1.hdmi-stereo.monitor",
    "bluez_output.B4:E7:B3:DB:16:D3.monitor",
    # PulseAudio descriptions -- the "Monitor of ..." prefix form.
    "Monitor of Radeon High Definition Audio Controller Digital Stereo (HDMI)",
    "Monitor of USB Audio Device Analog Stereo",
    "Loopback Capture",
]

REAL_MICS = [
    "alsa_input.usb-GeneralPlus_USB_Audio_Device-00.mono-fallback",
    "alsa_input.usb-Anker_Anker_PowerConf_C300_AR2MXP0B22401514-00.analog-stereo",
    "bluez_input.B4:E7:B3:DB:16:D3",
    "USB Audio Device Mono",
    "default",
    "pipewire",
]


@pytest.mark.parametrize("name", LOOPBACK)
def test_loopback_sources_are_rejected(name):
    assert _is_loopback(name)


@pytest.mark.parametrize("name", REAL_MICS)
def test_real_microphones_are_kept(name):
    assert not _is_loopback(name)


def test_matching_is_case_insensitive():
    assert _is_loopback("ALSA_OUTPUT.HDMI-STEREO.MONITOR")
    assert _is_loopback("monitor of Built-in Audio")


def test_a_device_merely_named_after_a_monitor_is_not_loopback():
    # "Monitor" as a substring is not enough -- an HDMI capture card or a
    # display's built-in mic would otherwise be filtered out wrongly.
    assert not _is_loopback("Dell Monitor Webcam Microphone")
