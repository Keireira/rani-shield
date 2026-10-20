"""Editor API acceptance tests with an injected CPU-only stand-in."""
import base64
import io
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from PIL import Image

from gui import Studio, create_app


def image_payload(width=64, height=32, name="photo.png"):
    stream = io.BytesIO()
    Image.new("RGB", (width, height), (80, 120, 160)).save(stream, format="PNG")
    return {"name": name, "data": "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode(),
            "role": "source"}


class FakeProtector:
    _loaded = True

    def __init__(self, gate, calls=None):
        self.gate = gate
        self.calls = calls if calls is not None else []

    def protect(self, input_path, output_dir, max_size, progress,
                mode="phantomseal_faceshield", phantom_iterations=1000, **kwargs):
        self.calls.append({"input_path": input_path, "mode": mode,
                           "phantom_iterations": phantom_iterations, **kwargs})
        progress(0.2, "Test objective")
        if not self.gate.wait(10):
            raise TimeoutError("test gate timed out")
        with Image.open(input_path) as source:
            source = source.convert("RGB")
            source.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        output = Path(output_dir)
        output.mkdir(parents=True)
        source.save(output / "source.png")
        result = source.copy()
        stages = []
        stage1_path = None
        if mode in ("phantomseal", "phantomseal_faceshield"):
            result.putpixel((0, 0), (82, 120, 160))
            stage1_path = str(output / "phantomseal.png")
            result.save(stage1_path)
            stages.append({"method": "phantomseal", "max_pixel_change": 2})
        if mode in ("faceshield", "phantomseal_faceshield"):
            current = result.getpixel((0, 0))
            result.putpixel((0, 0), (current[0] + 1, current[1], current[2]))
            stages.append({"method": "faceshield", "max_pixel_change": 1})
        result.save(output / "protected.png")
        progress(1.0, "Saved")
        return {"source_path": str(output / "source.png"), "output_path": str(output / "protected.png"),
                "stage1_path": stage1_path, "method": mode, "stages": stages,
                "max_pixel_change": result.getpixel((0, 0))[0] - source.getpixel((0, 0))[0],
                "width": source.width, "height": source.height, "elapsed_seconds": 0.1, "peak_vram_gb": 0}


class FakeTestRunner:
    def __init__(self, gate):
        self.gate, self.calls, self.fail, self.outside = gate, [], False, None
        self.failed_detection = False

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        kwargs["progress"](.3, "CPU test evaluator")
        if not self.gate.wait(10):
            raise TimeoutError("test gate timed out")
        if self.fail:
            raise RuntimeError("SimSwap test failed after model loading")
        folder = Path(kwargs["output_dir"])
        folder.mkdir(parents=True)
        target = folder / "target.png"
        Image.new("RGB", (224, 224), (20, 60, 100)).save(target)
        conditions = []
        for name in (["png", "jpeg95"] if kwargs["jpeg95"] else ["png"]):
            rows = []
            keys = ["clean", "stage1", "final"] if kwargs["mode"] == "phantomseal_faceshield" else ["clean", "final"]
            for key in keys:
                swap = folder / (name + "_" + key + ".png")
                Image.new("RGB", (224, 224), (30, 70, 110)).save(swap)
                rows.append({"key": key, "label": key, "swap_path": str(swap),
                             "score": {"clean": .9, "stage1": .3, "final": .2}[key],
                             "source_detected": True, "swap_detected": not (self.failed_detection and key == "final"), "status": "ok"})
            conditions.append({"name": name, "rows": rows, "delta_final_minus_stage1": -.1})
        return {"target_path": str(self.outside or target), "conditions": conditions}


class EditorAPITests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        root = Path(self.folder.name)
        (root / "web").mkdir()
        (root / "web/index.html").write_text("<title>Test editor</title>")
        self.gate = threading.Event()
        self.gate.set()
        self.test_gate = threading.Event()
        self.test_gate.set()
        self.test_runner = FakeTestRunner(self.test_gate)
        self.factory_calls = 0
        self.protection_calls = []

        def factory():
            self.factory_calls += 1
            return FakeProtector(self.gate, self.protection_calls)

        self.studio = Studio(root, protector_factory=factory, test_runner=self.test_runner)
        self.client = TestClient(create_app(root, self.studio), base_url="http://127.0.0.1:7876")

    def tearDown(self):
        self.gate.set()
        self.test_gate.set()
        deadline = time.monotonic() + 3
        while self.studio.active is not None and time.monotonic() < deadline:
            time.sleep(0.01)
        self.client.close()
        self.folder.cleanup()

    def upload(self, width=64, height=32):
        response = self.client.post("/api/upload", json=image_payload(width, height))
        self.assertEqual(response.status_code, 200, response.text)
        return response.json()

    def wait_job(self, job_id):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            response = self.client.get(f"/api/job/{job_id}")
            self.assertEqual(response.status_code, 200)
            value = response.json()
            if value["status"] in ("done", "error"):
                return value
            time.sleep(0.01)
        self.fail("Job timed out")

    def completed_protection(self, mode="phantomseal_faceshield"):
        original = self.upload()
        response = self.client.post("/api/process", json={"source_id": original["id"],
                                                          "params": {"mode": mode, "iterations": 1}})
        self.assertEqual(response.status_code, 202, response.text)
        job = self.wait_job(response.json()["id"])
        self.assertEqual(job["status"], "done", job.get("error"))
        self.assertEqual(job["kind"], "protect")
        deadline = time.monotonic() + 2
        while self.studio.active is not None and time.monotonic() < deadline:
            time.sleep(.005)
        return job

    def test_protection_test_chain_registers_all_outputs_and_preserves_editor_photo(self):
        protection = self.completed_protection()
        before = self.client.get("/api/state").json()
        self.assertIn(protection["result"]["id"], before["testable_result_ids"])
        self.assertTrue(before["testing"]["ready"])
        self.assertEqual([target["key"] for target in before["testing"]["targets"]], ["cage", "wlh", "zxy"])
        response = self.client.post("/api/test", json={"result_id": protection["result"]["id"], "target_key": "cage"})
        self.assertEqual(response.status_code, 202, response.text)
        self.assertEqual(response.json()["kind"], "test")
        job = self.wait_job(response.json()["id"])
        self.assertEqual(job["status"], "done", job.get("error"))
        self.assertEqual(job["kind"], "test")
        self.assertEqual(job["result_id"], protection["result"]["id"])
        self.assertEqual(job["test"]["metric"], "FaceNet cosine to clean source")
        self.assertEqual(job["test"]["target"]["kind"], "test_target")
        self.assertEqual(self.client.get(job["test"]["target"]["url"]).status_code, 200)
        self.assertEqual([condition["name"] for condition in job["test"]["conditions"]], ["png", "jpeg95"])
        for condition in job["test"]["conditions"]:
            self.assertEqual([row["key"] for row in condition["rows"]], ["clean", "stage1", "final"])
            self.assertAlmostEqual(condition["delta_final_minus_stage1"], -.1)
            for row in condition["rows"]:
                self.assertIsNotNone(row["score"])
                self.assertTrue(row["source_detected"] and row["swap_detected"])
                self.assertNotIn("swap_path", row)
                self.assertEqual(row["swap"]["kind"], "test_swap")
                self.assertEqual(self.client.get(row["swap"]["url"]).headers["content-type"], "image/png")
        self.assertNotIn("target_path", job["test"])
        after = self.client.get("/api/state").json()
        for key in ("source", "result", "default_source_id", "default_result_id"):
            self.assertEqual(after[key], before[key])
        self.assertEqual(self.studio.job(protection["id"])["stage1"], protection["stage1"])
        self.assertEqual(after["latest_test_job_ids"][protection["result"]["id"]], job["id"])
        call = self.test_runner.calls[-1]
        self.assertEqual(call["source_path"], self.studio.asset_path(protection["source"]["id"]))
        self.assertEqual(call["stage1_path"], self.studio.asset_path(protection["stage1"]["id"]))
        self.assertTrue(call["jpeg95"])

    def test_protection_test_single_modes_show_two_distinct_variants_only(self):
        for mode in ("phantomseal", "faceshield"):
            with self.subTest(mode=mode):
                protection = self.completed_protection(mode)
                response = self.client.post("/api/test", json={"result_id": protection["result"]["id"],
                                                               "target_key": "wlh", "jpeg95": False})
                self.assertEqual(response.status_code, 202, response.text)
                job = self.wait_job(response.json()["id"])
                self.assertEqual(job["status"], "done", job.get("error"))
                self.assertEqual(len(job["test"]["conditions"]), 1)
                condition = job["test"]["conditions"][0]
                self.assertEqual([row["key"] for row in condition["rows"]], ["clean", "final"])
                self.assertIsNone(condition["delta_final_minus_stage1"])
                self.assertIsNone(self.test_runner.calls[-1]["stage1_path"])
                self.assertEqual(self.test_runner.calls[-1]["mode"], mode)

    def test_protection_and_test_share_one_busy_lock_in_both_directions(self):
        protection = self.completed_protection()
        test_payload = {"result_id": protection["result"]["id"], "target_key": "zxy"}
        protect_payload = {"source_id": protection["source_id"]}
        self.gate.clear()
        protecting = self.client.post("/api/process", json=protect_payload)
        self.assertEqual(protecting.status_code, 202)
        self.assertEqual(self.client.post("/api/test", json=test_payload).status_code, 409)
        self.assertEqual(self.client.get("/api/state").json()["active_job_kind"], "protect")
        self.gate.set()
        self.assertEqual(self.wait_job(protecting.json()["id"])["status"], "done")
        self.test_gate.clear()
        testing = self.client.post("/api/test", json=test_payload)
        self.assertEqual(testing.status_code, 202, testing.text)
        self.assertEqual(self.client.get("/api/state").json()["active_job_kind"], "test")
        self.assertEqual(self.client.post("/api/process", json=protect_payload).status_code, 409)
        self.assertEqual(self.client.post("/api/test", json=test_payload).status_code, 409)
        self.test_gate.set()
        self.assertEqual(self.wait_job(testing.json()["id"])["status"], "done")

    def test_protection_test_validates_ids_target_boolean_and_forbids_client_paths(self):
        protection = self.completed_protection()
        payload = {"result_id": protection["result"]["id"], "target_key": "cage"}
        unrelated = self.upload()
        self.studio.jobs["unfinished"] = {"id": "unfinished", "kind": "protect", "status": "running",
                                           "result": {"id": unrelated["id"]}}
        for result_id in (unrelated["id"], protection["source"]["id"], "../../models/weights.pth", "unknown"):
            self.assertEqual(self.client.post("/api/test", json={**payload, "result_id": result_id}).status_code, 404)
        invalid = [{"target_key": "../cage.jpg"}, {"target_key": []}, {"result_id": []},
                   {"result_id": True}, {"target_path": "C:/outside.png"}, {"source_path": "secret.png"}]
        invalid += [{"jpeg95": value} for value in (1, 0, None, "true", [])]
        for changes in invalid:
            response = self.client.post("/api/test", json={**payload, **changes})
            self.assertEqual(response.status_code, 400, (changes, response.text))
        self.assertEqual(self.test_runner.calls, [])

    def test_protection_test_failure_clears_busy_and_preserves_protected_result(self):
        protection = self.completed_protection()
        before = (self.studio.default_source, self.studio.default_result)
        self.test_runner.fail = True
        payload = {"result_id": protection["result"]["id"], "target_key": "cage", "jpeg95": False}
        with self.assertLogs("faceshield.editor", level="ERROR"):
            response = self.client.post("/api/test", json=payload)
            failed = self.wait_job(response.json()["id"])
        self.assertEqual(failed["status"], "error")
        self.assertIsNone(failed["test"])
        self.assertIn("SimSwap test failed", failed["error"])
        self.assertIsNone(self.studio.active)
        self.assertEqual((self.studio.default_source, self.studio.default_result), before)
        self.test_runner.fail = False
        response = self.client.post("/api/test", json=payload)
        self.assertEqual(response.status_code, 202)
        self.assertEqual(self.wait_job(response.json()["id"])["status"], "done")

    def test_protection_test_missing_runtime_returns_actionable_error_without_queueing(self):
        protection = self.completed_protection()
        self.studio._test_runner = None
        with patch("protection_test.testing_readiness", return_value={"ready": False,
                   "message": "Missing test files: weights/simswap/latest_net_G.pth", "targets": [],
                   "backends": {"simswap": {"ready": False, "message": "Missing weights/simswap/latest_net_G.pth"}}}):
            response = self.client.post("/api/test", json={"result_id": protection["result"]["id"], "target_key": "cage"})
        self.assertEqual(response.status_code, 503)
        self.assertIn("latest_net_G.pth", response.json()["error"])
        self.assertIsNone(self.studio.active)

    def test_protection_test_failed_detection_cannot_publish_a_finite_score(self):
        protection = self.completed_protection()
        self.test_runner.failed_detection = True
        response = self.client.post("/api/test", json={"result_id": protection["result"]["id"], "target_key": "cage"})
        job = self.wait_job(response.json()["id"])
        self.assertEqual(job["status"], "done", job.get("error"))
        for condition in job["test"]["conditions"]:
            self.assertFalse(condition["rows"][-1]["swap_detected"])
            self.assertIsNone(condition["rows"][-1]["score"])
            self.assertIsNone(condition["delta_final_minus_stage1"])

    def test_backend_selection_is_validated_and_preserved_in_jobs(self):
        protection = self.completed_protection()
        payload = {"result_id": protection["result"]["id"], "target_key": "cage", "jpeg95": False}
        for backend in ("../diffswap", "unknown", [], True, None):
            response = self.client.post("/api/test", json={**payload, "backend": backend})
            self.assertEqual(response.status_code, 400)
        self.assertEqual(self.test_runner.calls, [])
        for backend in ("simswap", "diffswap", "infoswap", "uniface"):
            response = self.client.post("/api/test", json={**payload, "backend": backend})
            self.assertEqual(response.status_code, 202, response.text)
            job = self.wait_job(response.json()["id"])
            self.assertEqual(job["status"], "done", job.get("error"))
            self.assertEqual(job["backend"], backend)
            self.assertEqual(job["test"]["backend"], backend)
            self.assertEqual(self.test_runner.calls[-1]["backend"], backend)

    def test_unavailable_selected_backend_cannot_fall_back_to_another(self):
        protection = self.completed_protection()
        statuses = {"ready": True, "backends": {
            "simswap": {"ready": True, "message": "Ready"},
            "diffswap": {"ready": False, "message": "Missing diffswap.pth"}}}
        with patch.object(self.studio, "testing_readiness", return_value=statuses):
            response = self.client.post("/api/test", json={
                "result_id": protection["result"]["id"], "target_key": "cage", "backend": "diffswap"})
        self.assertEqual(response.status_code, 503)
        self.assertIn("diffswap.pth", response.json()["error"])
        self.assertEqual(self.test_runner.calls, [])
        self.assertIsNone(self.studio.active)

    def test_protection_test_cannot_publish_a_runner_output_outside_its_test_folder(self):
        protection = self.completed_protection()
        self.test_runner.outside = self.studio.asset_path(protection["result"]["id"])
        with self.assertLogs("faceshield.editor", level="ERROR"):
            response = self.client.post("/api/test", json={"result_id": protection["result"]["id"], "target_key": "cage"})
            failed = self.wait_job(response.json()["id"])
        self.assertEqual(failed["status"], "error")
        self.assertIn("test folder", failed["error"])
        self.assertEqual(self.studio.default_result, protection["result"]["id"])

    def test_state_upload_pixels_identical_compare_do_not_load_models(self):
        self.assertTrue(self.client.get("/api/state").json()["runtime"]["ready"])
        asset = self.upload()
        response = self.client.get(f"/api/pixels/{asset['id']}")
        self.assertEqual(len(response.content), 64 * 32 * 4)
        self.assertEqual(response.content[:4], bytes([80, 120, 160, 255]))
        response = self.client.post("/api/compare", json={"source_id": asset["id"], "result_id": asset["id"]})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["identical"])
        self.assertIsNone(response.json()["psnr_db"])
        self.assertEqual(self.factory_calls, 0)

    def test_state_readiness_preserves_available_faceshield_without_loading_pipeline(self):
        engines = {
            "phantomseal": {"ready": False, "message": "PhantomSeal cloak images are missing."},
            "faceshield": {"ready": True, "message": "FaceShield models are available."},
        }
        checks = []

        def readiness(root):
            checks.append(root)
            return engines

        self.studio._readiness_provider = readiness
        state = self.client.get("/api/state").json()
        self.assertTrue(state["runtime"]["ready"])
        self.assertEqual(state["engines"], engines)
        self.assertEqual(state["defaults"]["mode"], "phantomseal_faceshield")
        self.assertEqual(state["defaults"]["phantom_iterations"], 1000)
        self.assertEqual(checks, [self.studio.root])
        self.assertEqual(self.factory_calls, 0)
        engines["faceshield"]["ready"] = False
        self.assertFalse(self.client.get("/api/state").json()["runtime"]["ready"])

    def test_process_resizes_matching_source_and_result_and_reuses_models(self):
        original = self.upload(1600, 900)
        for _ in range(2):
            response = self.client.post("/api/process", json={"source_id": original["id"],
                                                            "params": {"max_size": 1280, "iterations": 1}})
            self.assertEqual(response.status_code, 202, response.text)
            job = self.wait_job(response.json()["id"])
            self.assertEqual(job["status"], "done", job.get("error"))
            self.assertEqual(job["source_id"], original["id"])
            self.assertEqual((job["source"]["width"], job["source"]["height"]), (1280, 720))
            self.assertEqual((job["result"]["width"], job["result"]["height"]), (1280, 720))
            self.assertEqual(job["progress"], 100)
            self.assertFalse(job["metrics"]["identical"])
            self.assertEqual(self.client.get(job["result"]["url"]).headers["content-type"], "image/png")
        self.assertEqual(self.factory_calls, 1)

    def test_default_chain_preserves_clean_baseline_and_exposes_stage_one(self):
        original = self.upload()
        response = self.client.post("/api/process", json={
            "source_id": original["id"],
            "params": {"iterations": 1, "phantom_iterations": 37},
        })
        self.assertEqual(response.status_code, 202)
        job = self.wait_job(response.json()["id"])
        self.assertEqual(job["status"], "done", job.get("error"))
        self.assertEqual(job["source_id"], original["id"])
        self.assertEqual(job["method"], "phantomseal_faceshield")
        self.assertEqual([stage["method"] for stage in job["stages"]], ["phantomseal", "faceshield"])
        self.assertEqual(job["stage1"]["kind"], "intermediate")
        self.assertEqual(job["performance"]["max_pixel_change"], 3)
        self.assertNotIn("stage1_path", job["performance"])
        for key, pixel in (("source", 80), ("stage1", 82), ("result", 83)):
            response = self.client.get(f"/api/pixels/{job[key]['id']}")
            self.assertEqual(response.content[:4], bytes([pixel, 120, 160, 255]))
        self.assertAlmostEqual(job["metrics"]["mae"], 3 / (64 * 32 * 3), places=7)
        self.assertEqual(self.protection_calls[0]["mode"], "phantomseal_faceshield")
        self.assertEqual(self.protection_calls[0]["phantom_iterations"], 37)
        self.assertEqual(self.protection_calls[0]["analysis_size"], 512)

    def test_single_engine_modes_remain_available(self):
        original = self.upload()
        for mode in ("faceshield", "phantomseal"):
            response = self.client.post("/api/process", json={
                "source_id": original["id"], "params": {"mode": mode, "iterations": 1},
            })
            self.assertEqual(response.status_code, 202, response.text)
            job = self.wait_job(response.json()["id"])
            self.assertEqual(job["status"], "done", job.get("error"))
            self.assertEqual(job["method"], mode)
            self.assertEqual([stage["method"] for stage in job["stages"]], [mode])
            if mode == "faceshield":
                self.assertIsNone(job["stage1"])
            else:
                self.assertIsNotNone(job["stage1"])
        self.assertEqual(self.factory_calls, 1)

    def test_stage_failure_does_not_publish_partial_result_and_releases_job(self):
        class FailedPipeline:
            def protect(self, *args, **kwargs):
                raise RuntimeError("FaceShield stage failed after PhantomSeal")

        self.studio._protector = FailedPipeline()
        original = self.upload()
        with self.assertLogs("faceshield.editor", level="ERROR"):
            response = self.client.post("/api/process", json={"source_id": original["id"]})
            failed = self.wait_job(response.json()["id"])
        self.assertEqual(failed["status"], "error")
        self.assertIn("FaceShield stage failed", failed["error"])
        self.assertIsNone(failed["source"])
        self.assertIsNone(failed["result"])
        self.assertIsNone(failed["stage1"])
        deadline = time.monotonic() + 2
        while self.studio.active is not None and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertIsNone(self.studio.active)
        self.studio._protector = None
        response = self.client.post("/api/process", json={"source_id": original["id"]})
        self.assertEqual(response.status_code, 202)
        self.assertEqual(self.wait_job(response.json()["id"])["status"], "done")

    def test_busy_job_is_rejected(self):
        asset = self.upload()
        self.gate.clear()
        payload = {"source_id": asset["id"], "params": {"iterations": 1}}
        first = self.client.post("/api/process", json=payload)
        self.assertEqual(first.status_code, 202)
        second = self.client.post("/api/process", json=payload)
        self.assertEqual(second.status_code, 409)
        self.assertIn("error", second.json())
        self.gate.set()
        self.assertEqual(self.wait_job(first.json()["id"])["status"], "done")

    def test_origin_path_content_type_and_parameter_validation(self):
        self.assertEqual(self.client.get("/api/state", headers={"Origin": "https://example.com"}).status_code, 403)
        self.assertEqual(self.client.get("/api/state", headers={"Host": "evil.example:7876"}).status_code, 403)
        self.assertEqual(self.client.get("/api/state", headers={"Host": "127.0.0.1:9000"}).status_code, 403)
        self.assertEqual(self.client.get("/assets/not-an-asset").status_code, 404)
        self.assertEqual(self.client.get("/protector.py").status_code, 404)
        self.assertEqual(self.client.post("/api/upload", content="{}", headers={"Content-Type": "text/plain"}).status_code, 415)
        self.assertEqual(self.client.post("/api/upload", json={"data": "not an image"}).status_code, 400)
        asset = self.upload()
        for params in ({"iterations": True}, {"max_size": 1281}, {"noise_clamp": 99}, {"step_size": 0},
                       {"mode": "unknown"}, {"mode": True}, {"mode": []},
                       {"phantom_iterations": True}, {"phantom_iterations": 9},
                       {"phantom_iterations": 2001}, {"phantom_iterations": 10.5}):
            response = self.client.post("/api/process", json={"source_id": asset["id"], "params": params})
            self.assertEqual(response.status_code, 400, params)
            self.assertIn("error", response.json())
        self.assertEqual(self.factory_calls, 0)

    def test_compare_rejects_dimension_mismatch(self):
        a, b = self.upload(64, 32), self.upload(32, 32)
        response = self.client.post("/api/compare", json={"source_id": a["id"], "result_id": b["id"]})
        self.assertEqual(response.status_code, 400)

    def test_nullswap_controls_roundtrip_and_legacy_defaults(self):
        original = self.upload()
        for params, strength, seed in (({"mode": "nullswap"}, 1, 0),
                                      ({"mode": "nullswap", "nullswap_strength": 2.75,
                                        "nullswap_seed": 4294967295}, 2.75, 4294967295)):
            response = self.client.post("/api/process", json={"source_id": original["id"], "params": params})
            self.assertEqual(response.status_code, 202, response.text)
            job = self.wait_job(response.json()["id"])
            self.assertEqual(job["status"], "done", job.get("error"))
            for values in (job["params"], self.protection_calls[-1]):
                self.assertEqual(values["nullswap_strength"], strength)
                self.assertEqual(values["nullswap_seed"], seed)

    def test_invalid_nullswap_controls_rejected_before_queueing(self):
        original = self.upload()
        for key, values in (("nullswap_strength", (-.1, 4.1, "1", True)),
                            ("nullswap_seed", (-1, 2**32, .5, "0", True))):
            for value in values:
                response = self.client.post("/api/process", json={"source_id": original["id"],
                    "params": {"mode": "nullswap", key: value}})
                self.assertEqual(response.status_code, 400, response.text)
        self.assertEqual(self.factory_calls, 0)
        self.assertIsNone(self.studio.active)


if __name__ == "__main__":
    unittest.main()
