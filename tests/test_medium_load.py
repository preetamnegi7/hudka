"""Loading the medium model on a 12 GB card.

The library moves the 9.2 GB float32 checkpoint onto the GPU and only THEN halves it. That
is why "medium kills the process" and "medium peaks at 5-6.5 GB" were both true. Hudka's
loader stages on the CPU, casts, then moves - the same two lines in the other order - and
refuses, in words, when the free VRAM measured right before loading is not enough.

No real model or GPU here: the library and torch are stand-ins that record what they are
asked to do, which is the whole point under test.
"""

from __future__ import annotations

import sys
import types

import pytest

from hudka.engines import hardware
from hudka.engines.stable_audio3 import StableAudio3Engine


class Recorder:
    """An nn.Module stand-in: .to() records its argument and returns self."""

    def __init__(self):
        self.moves = []

    def to(self, target):
        self.moves.append(target)
        return self


class FakeStableAudioModel:
    """What StableAudioModel.from_pretrained hands back."""

    calls: list[dict] = []

    def __init__(self, device):
        self.model = Recorder()
        encoder = Recorder()
        cond = types.SimpleNamespace()
        cond.__dict__["model"] = encoder          # where the real conditioner keeps it
        self.model.conditioner = types.SimpleNamespace(conditioners={"prompt": cond})
        self.device = device
        self.model_half = False

    @staticmethod
    def from_pretrained(model_name, device=None, model_half=True):
        FakeStableAudioModel.calls.append({"name": model_name, "device": device, "model_half": model_half})
        return FakeStableAudioModel(device)


def roomy(**kw) -> hardware.Hardware:
    base = dict(device="cuda", gpu_name="RTX 4070", total_vram_gb=12.9, free_vram_gb=9.0,
                bf16=True, ram_gb=64.0, cores=24)
    base.update(kw)
    return hardware.Hardware(**base)


@pytest.fixture
def fake_torch(monkeypatch):
    torch = types.SimpleNamespace(
        float16="fp16", bfloat16="bf16",
        cuda=types.SimpleNamespace(is_bf16_supported=lambda: True, empty_cache=lambda: None),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    return torch


@pytest.fixture
def engine(monkeypatch, fake_torch):
    FakeStableAudioModel.calls = []
    monkeypatch.setattr(StableAudio3Engine, "_require_package", staticmethod(lambda: FakeStableAudioModel))
    monkeypatch.setattr(hardware, "detect", lambda refresh=False: roomy())
    return StableAudio3Engine("stable-audio-3-medium", device="cuda")


class TestLoadOrder:
    def test_stages_on_cpu_casts_then_moves(self, engine):
        model = engine._load()

        assert FakeStableAudioModel.calls == [{"name": "medium", "device": "cpu", "model_half": False}], \
            "the library must be asked for an fp32 model on the CPU, never on the card"
        assert model.model.moves == ["fp16", "cuda"], "cast FIRST, then move - the library does the reverse"
        assert model.device == "cuda" and model.model_half is True

    def test_the_text_encoder_is_cast_to_bf16(self, engine):
        model = engine._load()
        encoder = model.model.conditioner.conditioners["prompt"].__dict__["model"]
        assert encoder.moves == ["bf16"], \
            "it lives in __dict__, so the model's fp16 cast never reaches it; fp32 costs 1.13 GB"

    def test_without_bf16_the_encoder_is_left_alone(self, engine, fake_torch):
        fake_torch.cuda.is_bf16_supported = lambda: False
        model = engine._load()
        encoder = model.model.conditioner.conditioners["prompt"].__dict__["model"]
        assert encoder.moves == []

    def test_loads_once(self, engine):
        first = engine._load()
        assert engine._load() is first
        assert len(FakeStableAudioModel.calls) == 1

    def test_cpu_path_is_unchanged(self, monkeypatch, fake_torch):
        FakeStableAudioModel.calls = []
        monkeypatch.setattr(StableAudio3Engine, "_require_package", staticmethod(lambda: FakeStableAudioModel))
        cpu = StableAudio3Engine("stable-audio-3-small-music", device="cpu")
        model = cpu._load()
        assert FakeStableAudioModel.calls == [{"name": "small-music", "device": "cpu", "model_half": False}]
        assert model.model.moves == [] and model.device == "cpu" and model.model_half is False


class TestGuard:
    def test_refuses_before_touching_weights_when_vram_is_short(self, engine, monkeypatch):
        monkeypatch.setattr(hardware, "detect", lambda refresh=False: roomy(free_vram_gb=3.1))
        with pytest.raises(RuntimeError) as err:
            engine._load()
        text = str(err.value)
        assert "needs about" in text and "3.1 GB" in text and "other applications hold 9.8 GB" in text
        assert FakeStableAudioModel.calls == [], "the refusal must come before any weights load"

    def test_small_variants_never_guard(self, monkeypatch, fake_torch):
        FakeStableAudioModel.calls = []
        monkeypatch.setattr(StableAudio3Engine, "_require_package", staticmethod(lambda: FakeStableAudioModel))
        monkeypatch.setattr(hardware, "detect", lambda refresh=False: roomy(free_vram_gb=3.1))
        small = StableAudio3Engine("stable-audio-3-small-sfx", device="cuda")
        small._load()
        assert len(FakeStableAudioModel.calls) == 1

    def test_an_unknown_free_figure_does_not_refuse(self, engine, monkeypatch):
        monkeypatch.setattr(hardware, "detect", lambda refresh=False: roomy(free_vram_gb=0.0))
        engine._load()
        assert len(FakeStableAudioModel.calls) == 1
