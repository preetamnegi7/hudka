"""Hardware detection and the tier it produces.

The point under test is the one that mattered on the development machine: a 12.9 GB card
with 5 GB held by the desktop is, for model-loading purposes, a 7.9 GB card - and the
decision has to be made from that number, from nvidia-smi, without importing torch into the
GUI process.
"""

from __future__ import annotations

import subprocess

import pytest

from hudka.engines import hardware
from hudka.engines.hardware import Hardware, Tier


def hw(**kw) -> Hardware:
    base = dict(device="cuda", gpu_name="NVIDIA GeForce RTX 4070", total_vram_gb=12.9,
                free_vram_gb=8.0, bf16=True, ram_gb=64.0, cores=24, torch_build="2.7.1+cu128")
    base.update(kw)
    return Hardware(**base)


class TestTier:
    def test_uses_free_not_total(self):
        assert hw(free_vram_gb=5.0).tier is Tier.GPU_LITE, "12.9 GB total decides nothing"
        assert hw(free_vram_gb=8.0).tier is Tier.GPU_MEDIUM
        assert hw(free_vram_gb=16.5).tier is Tier.GPU_LARGE

    def test_cpu_is_cpu(self):
        assert hw(device="cpu", free_vram_gb=99.0).tier is Tier.CPU

    def test_medium_needs_ram_to_stage_the_checkpoint(self):
        assert hw(free_vram_gb=8.0, ram_gb=16.0).tier is Tier.GPU_LITE

    def test_held_by_others(self):
        assert hw(total_vram_gb=12.9, free_vram_gb=7.6).held_by_others_gb == pytest.approx(5.3)


class TestFitRule:
    def test_need_grows_with_duration(self):
        short, long = hardware.medium_need_gb(120.0, hw()), hardware.medium_need_gb(380.0, hw())
        assert short == pytest.approx(7.0, abs=0.15)
        assert long > short

    def test_an_fp32_text_encoder_costs_more(self):
        assert hardware.medium_need_gb(120.0, hw(bf16=False)) - hardware.medium_need_gb(120.0, hw()) \
            == pytest.approx((1.13 - 0.56) * hardware.HEADROOM, abs=0.01)

    def test_fits_is_the_need_against_free(self):
        assert hardware.medium_fits(60.0, hw(free_vram_gb=8.0))
        assert not hardware.medium_fits(380.0, hw(free_vram_gb=8.0))
        assert not hardware.medium_fits(60.0, hw(device="cpu"))


class TestTierTable:
    def test_cpu_row_is_todays_behaviour(self):
        from hudka.engines import DEFAULT_ENGINES

        for kind in ("music", "ambience", "sfx"):
            assert hardware.engine_for(kind, Tier.CPU) == DEFAULT_ENGINES[kind]
            assert hardware.steps_for(kind, Tier.CPU) == 8

    def test_gpu_lite_keeps_small_models_but_richer_beds(self):
        assert hardware.engine_for("music", Tier.GPU_LITE) == "stable-audio-3-small-music"
        assert hardware.steps_for("music", Tier.GPU_LITE) == 50
        assert hardware.steps_for("ambience", Tier.GPU_LITE) == 50
        assert hardware.steps_for("sfx", Tier.GPU_LITE) == 8

    def test_medium_tiers_put_beds_on_medium_and_leave_one_shots_alone(self):
        for tier in (Tier.GPU_MEDIUM, Tier.GPU_LARGE):
            assert hardware.engine_for("music", tier) == hardware.MEDIUM
            assert hardware.engine_for("ambience", tier) == hardware.MEDIUM
            assert hardware.engine_for("sfx", tier) == "stable-audio-3-small-sfx", \
                "medium's one-shot quality is unmeasured; small-sfx is post-trained for them"

    def test_fast_is_eight_steps_everywhere(self):
        for tier in Tier:
            for kind in ("music", "ambience", "sfx"):
                assert hardware.steps_for(kind, tier, quality="fast") == 8


class TestProbes:
    def test_nvidia_smi_csv_is_parsed(self, monkeypatch):
        calls = []

        def fake_run(cmd, **kw):
            calls.append(cmd)
            text = "8.9\n" if "compute_cap" in cmd[1] else "NVIDIA GeForce RTX 4070, 12282, 7539\n"
            return subprocess.CompletedProcess(cmd, 0, stdout=text, stderr="")

        monkeypatch.setattr(hardware.shutil, "which", lambda name: r"C:\Windows\nvidia-smi.exe")
        monkeypatch.setattr(hardware.subprocess, "run", fake_run)
        gpu = hardware._probe_nvidia_smi()
        assert gpu["name"] == "NVIDIA GeForce RTX 4070"
        assert gpu["total_gb"] == pytest.approx(12.878, abs=0.01)
        assert gpu["free_gb"] == pytest.approx(7.905, abs=0.01)
        assert gpu["compute_cap"] == 8.9

    def test_an_old_driver_without_compute_cap_still_reports_memory(self, monkeypatch):
        def fake_run(cmd, **kw):
            if "compute_cap" in cmd[1]:
                return subprocess.CompletedProcess(cmd, 2, stdout="", stderr="Field not found")
            return subprocess.CompletedProcess(cmd, 0, stdout="GTX 1080, 8192, 6000\n", stderr="")

        monkeypatch.setattr(hardware.shutil, "which", lambda name: "nvidia-smi")
        monkeypatch.setattr(hardware.subprocess, "run", fake_run)
        gpu = hardware._probe_nvidia_smi()
        assert gpu["free_gb"] == pytest.approx(6.29, abs=0.01)
        assert gpu["compute_cap"] == 0.0

    def test_no_nvidia_smi_means_no_gpu(self, monkeypatch):
        monkeypatch.setattr(hardware.shutil, "which", lambda name: None)
        assert hardware._probe_nvidia_smi() is None


class TestDetect:
    def test_never_raises(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("probe exploded")

        monkeypatch.setattr(hardware, "_probe_nvidia_smi", boom)
        monkeypatch.setattr(hardware, "_ram_gb", boom)
        monkeypatch.delenv("HUDKA_FORCE_TIER", raising=False)
        # _ram_gb is called outside the guarded block, so it must be safe on its own too.
        monkeypatch.setattr(hardware, "_ram_gb", lambda: 0.0)
        found = hardware.detect(refresh=True)
        assert found.device == "cpu" and found.tier is Tier.CPU

    def test_a_cpu_only_torch_wheel_means_no_gpu_even_with_a_card(self, monkeypatch):
        monkeypatch.setattr(hardware, "_probe_nvidia_smi",
                            lambda: {"name": "RTX 4070", "total_gb": 12.9, "free_gb": 8.0, "compute_cap": 8.9})
        monkeypatch.setattr(hardware, "_torch_build", lambda: "2.7.1+cpu")
        monkeypatch.delenv("HUDKA_FORCE_TIER", raising=False)
        found = hardware.detect(refresh=True)
        assert found.device == "cpu"
        assert "CPU-only build" in hardware.reason(found)

    def test_the_gui_never_imports_torch(self, monkeypatch):
        """The server lives all day; a CUDA context in it pins VRAM for that long."""
        import sys

        monkeypatch.setattr(hardware, "_probe_nvidia_smi", lambda: None)
        monkeypatch.delenv("HUDKA_ALLOW_TORCH_PROBE", raising=False)
        monkeypatch.delenv("HUDKA_FORCE_TIER", raising=False)
        monkeypatch.setitem(sys.modules, "torch", None)   # any import would now fail loudly
        found = hardware.detect(refresh=True)
        assert found.device == "cpu"

    def test_force_tier_is_consistent(self, monkeypatch):
        monkeypatch.setenv("HUDKA_FORCE_TIER", "gpu-medium")
        found = hardware.detect(refresh=True)
        assert found.tier is Tier.GPU_MEDIUM
        assert found.free_source == "forced" and found.free_vram_gb >= hardware.MEDIUM_MIN_FREE_GB
        monkeypatch.setenv("HUDKA_FORCE_TIER", "cpu")
        assert hardware.detect(refresh=True).tier is Tier.CPU


class TestWords:
    def test_summary_names_the_card_and_what_is_free(self):
        text = hardware.summary(hw(free_vram_gb=7.6))
        assert "RTX 4070" in text and "7.6 GB free of 12.9" in text and "medium" in text

    def test_reason_blames_other_apps_when_they_hold_the_card(self):
        text = hardware.reason(hw(free_vram_gb=5.1))
        assert "other applications" in text and "7.8 GB" in text and "reload" in text

    def test_reason_for_a_small_card_does_not_blame_the_user(self):
        text = hardware.reason(hw(total_vram_gb=6.0, free_vram_gb=5.5))
        assert "other applications" not in text and "6.0 GB card" in text
