"""Pipeline contract checks with small CPU images and model-free fake engines."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import weakref

import numpy as np
from PIL import Image

import protection_pipeline as pipeline_module
from protection_pipeline import ProtectionPipeline


class ModelToken:
    pass


class FakeEngine:
    def __init__(self, owner, name):
        self.owner, self.name = owner, name
        self.model = ModelToken()
        owner.model_refs.append(weakref.ref(self.model))
        owner.engine_refs.append(weakref.ref(self))
        owner.events.append(f"{name}:create")

    def protect(self, input_path, output_dir, progress, **kwargs):
        owner = self.owner
        owner.events.append(f"{self.name}:protect")
        progress(0, f"{self.name}: starting")
        with Image.open(input_path) as opened:
            source = opened.convert("RGB")
        owner.calls.append({"engine": self.name, "input_path": str(input_path),
                            "size": source.size, "pixel": source.getpixel((0, 0)), **kwargs})
        if owner.fail == self.name:
            raise RuntimeError(f"{self.name} failed")
        if self.name == "faceshield":
            source.thumbnail((kwargs["max_size"], kwargs["max_size"]), Image.Resampling.LANCZOS)
        if owner.resize == self.name:
            source = source.resize((source.width * 2, source.height * 2))
        change = np.asarray(owner.changes[self.name], dtype=np.int16)
        pixels = np.asarray(source).astype(np.int16)
        result = np.clip(pixels + change, 0, 255).astype(np.uint8)
        folder = Path(output_dir)
        folder.mkdir(parents=True, exist_ok=True)
        source.save(folder / "source.png")
        output = folder / "protected.png"
        Image.fromarray(result).save(output)
        progress(1, f"{self.name}: saved")
        return {"method": self.name, "source_path": str(folder / "source.png"),
                "output_path": str(output), "max_pixel_change": int(np.abs(change).max()),
                "peak_vram_gb": 1.25 if self.name == "phantomseal" else .75}

    def close(self):
        self.owner.events.append(f"{self.name}:close")
        self.model = None


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        self.input_path = self.root / "input.png"
        Image.new("RGB", (160, 80), (100, 110, 120)).save(self.input_path)
        self.events, self.calls, self.model_refs, self.engine_refs = [], [], [], []
        self.changes = {"phantomseal": (2, 1, -2), "faceshield": (1, -1, 3)}
        self.fail = self.resize = None
        self.job_number = 0
        self.pipeline = ProtectionPipeline(
            self.root, phantom_factory=lambda: FakeEngine(self, "phantomseal"),
            shield_factory=lambda: FakeEngine(self, "faceshield"))
        self.release_patch = patch.object(pipeline_module, "release_gpu", side_effect=self.release)
        self.release_patch.start()
        self.addCleanup(self.release_patch.stop)
        self.addCleanup(self.directory.cleanup)

    def release(self):
        # A cache flush alone is insufficient while a model still has a live owner.
        self.assertTrue(all(reference() is None for reference in self.model_refs))
        self.events.append("release")

    def protect(self, **kwargs):
        self.job_number += 1
        return self.pipeline.protect(self.input_path, self.root / f"job{self.job_number}",
                                     max_size=128, iterations=1, phantom_iterations=10, **kwargs)

    def test_chain_order_clean_baseline_same_geometry_and_cumulative_measurement(self):
        progress = []
        report = self.protect(progress=lambda fraction, message: progress.append(fraction))
        operations = [event for event in self.events if event != "release"]
        self.assertEqual(operations, ["phantomseal:create", "phantomseal:protect", "phantomseal:close",
                                      "faceshield:create", "faceshield:protect", "faceshield:close"])
        self.assertLess(self.events.index("phantomseal:close"), self.events.index("faceshield:create"))
        self.assertEqual(self.calls[0]["size"], (128, 64))
        self.assertEqual(self.calls[1]["size"], (128, 64))
        self.assertEqual(self.calls[1]["pixel"], (102, 111, 118))
        self.assertEqual(self.calls[1]["input_path"], report["stage1_path"])
        with Image.open(report["source_path"]) as source:
            self.assertEqual(source.size, (128, 64))
            self.assertEqual(source.getpixel((0, 0)), (100, 110, 120))
        with Image.open(report["output_path"]) as output:
            self.assertEqual(output.getpixel((0, 0)), (103, 110, 121))
        self.assertEqual(report["max_pixel_change"], 3)
        self.assertEqual(report["max_pixel_change_rgb"], [3, 0, 1])
        self.assertEqual([stage["name"] for stage in report["stages"]], ["PhantomSeal", "FaceShield"])
        self.assertEqual(report["peak_vram_gb"], 1.25)
        self.assertEqual(progress, sorted(progress))
        self.assertEqual((progress[0], progress[-1]), (0, 1))
        saved = json.loads((Path(report["output_path"]).parent / "metadata.json").read_text())
        self.assertEqual(saved["max_pixel_change_rgb"], [3, 0, 1])

    def test_combined_bound_is_measured_from_clean_not_relabelled_as_faceshield_bound(self):
        self.changes = {"phantomseal": (19, 8, 19), "faceshield": (12, 12, 12)}
        report = self.protect(noise_clamp=12)
        self.assertEqual(report["faceshield_noise_limit"], 12)
        self.assertEqual(report["max_pixel_change"], 31)
        self.assertEqual(report["max_pixel_change_rgb"], [31, 20, 31])

    def test_changed_stage_one_geometry_fails_before_faceshield_can_resize_it_back(self):
        self.resize = "phantomseal"
        with self.assertRaisesRegex(RuntimeError, "dimensions"):
            self.protect()
        self.assertNotIn("faceshield:create", self.events)
        self.assertIn("phantomseal:close", self.events)
        self.assertFalse(self.pipeline._loaded)

    def test_changed_final_geometry_is_rejected(self):
        self.resize = "faceshield"
        with self.assertRaisesRegex(RuntimeError, "dimensions"):
            self.protect()
        self.assertIn("faceshield:close", self.events)
        self.assertFalse((self.root / "job1" / "protected.png").exists())

    def test_errors_release_each_stage_and_pipeline_can_be_reused(self):
        for failing in ("phantomseal", "faceshield"):
            with self.subTest(engine=failing):
                self.fail = failing
                start = len(self.events)
                with self.assertRaisesRegex(RuntimeError, f"{failing} failed"):
                    self.protect()
                self.assertIn(f"{failing}:close", self.events[start:])
                self.assertFalse(self.pipeline._loaded)
                self.assertFalse((self.root / f"job{self.job_number}" / "protected.png").exists())
                self.assertTrue(all(reference() is None for reference in self.model_refs))
                self.fail = None
                self.assertEqual(self.protect()["max_pixel_change"], 3)
        self.assertTrue(all(reference() is None for reference in self.engine_refs))

    def test_single_modes_and_repeated_jobs_do_not_hold_models(self):
        for mode, expected in (("phantomseal", ["PhantomSeal"]), ("faceshield", ["FaceShield"]),
                               ("phantomseal_faceshield", ["PhantomSeal", "FaceShield"])):
            report = self.protect(mode=mode)
            self.assertEqual([stage["name"] for stage in report["stages"]], expected)
            self.assertEqual(report["stage1_path"] is None, mode == "faceshield")
            self.assertFalse(self.pipeline._loaded)
            self.assertTrue(all(reference() is None for reference in self.engine_refs))
            self.assertTrue(all(reference() is None for reference in self.model_refs))

    def test_invalid_stage_parameters_fail_before_model_factories(self):
        for arguments in ({"mode": "unknown"}, {"noise_clamp": 33}, {"step_size": 0},
                          {"noise_clamp": "12"}, {"step_size": True}, {"noise_clamp": float("nan")},
                          {"analysis_size": 65}, {"analysis_size": True}, {"analysis_size": 2048}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.protect(**arguments)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.events, [])

    def test_nullswap_is_independent_preserves_geometry_and_releases(self):
        self.changes["nullswap"] = (2, -3, 1)
        self.pipeline._nullswap_factory = lambda: FakeEngine(self, "nullswap")
        with patch.object(pipeline_module, "readiness", side_effect=AssertionError("No ensemble needed")):
            report = self.protect(mode="nullswap")
        self.assertEqual([call["engine"] for call in self.calls], ["nullswap"])
        self.assertEqual(self.calls[0]["size"], (128, 64))
        self.assertIsNone(report["stage1_path"])
        self.assertIsNone(report["faceshield_noise_limit"])
        self.assertEqual(report["max_pixel_change"], 3)
        self.assertFalse(self.pipeline._loaded)
        self.assertTrue(all(reference() is None for reference in self.engine_refs))

    def test_nullswap_controls_reach_only_its_engine(self):
        self.changes["nullswap"] = (2, -3, 1)
        self.pipeline._nullswap_factory = lambda: FakeEngine(self, "nullswap")
        self.protect(mode="nullswap", nullswap_strength=2.75, nullswap_seed=4294967295)
        self.assertEqual(self.calls[-1]["strength"], 2.75)
        self.assertEqual(self.calls[-1]["seed"], 4294967295)
        self.protect(mode="phantomseal_faceshield", nullswap_strength=4, nullswap_seed=123)
        for call in self.calls[1:]:
            self.assertNotIn("strength", call)
            self.assertNotIn("seed", call)

    def test_readiness_does_not_accept_config_without_required_weights(self):
        for relative in ("weights/faceshield/arcface50_checkpoint.tar", "weights/faceshield/arcface100_checkpoint.tar",
                         "weights/sd15/unet/config.json"):
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
        with patch.object(pipeline_module.importlib.util, "find_spec", return_value=object()):
            status = pipeline_module.readiness(self.root)
        self.assertFalse(status["faceshield"]["ready"])
        self.assertIn("ip-adapter_sd15.bin", status["faceshield"]["message"])
        self.assertIn("safetensors", status["faceshield"]["message"])
        self.assertEqual(self.events, [])


if __name__ == "__main__":
    unittest.main()
