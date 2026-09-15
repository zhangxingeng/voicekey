from voicekey.backend import resolve


def test_cpu_only_falls_back_to_int8():
    be = resolve(["CPUExecutionProvider"])
    assert be.providers == ["CPUExecutionProvider"]
    assert be.quantization == "int8"
    assert not be.accelerated


def test_cuda_selects_fp16_and_keeps_cpu_as_fallback():
    be = resolve(["CUDAExecutionProvider", "CPUExecutionProvider"])
    assert be.providers == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    assert be.quantization == "int8"
    assert be.label == "CUDA"
    assert be.accelerated


def test_coreml_is_picked_on_macos():
    be = resolve(["CoreMLExecutionProvider", "CPUExecutionProvider"])
    assert be.label == "CoreML"
    assert be.quantization == "int8"


def test_cuda_wins_over_other_accelerators():
    be = resolve(["DmlExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"])
    assert be.label == "CUDA"


def test_cpu_is_always_present_as_the_last_resort():
    for available in (
        ["CPUExecutionProvider"],
        ["CUDAExecutionProvider", "CPUExecutionProvider"],
        ["CoreMLExecutionProvider", "CPUExecutionProvider"],
    ):
        assert resolve(available).providers[-1] == "CPUExecutionProvider"
