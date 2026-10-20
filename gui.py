"""FaceShield local canvas editor server. Run: python gui.py --port 7876."""
from __future__ import annotations

import argparse
import base64
import binascii
import io
import json
import logging
import math
import os
from pathlib import Path
import secrets
import sys
import threading
import time
from urllib.parse import urlsplit
import webbrowser

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException


ROOT = Path(__file__).resolve().parent
MAX_BODY = 48 * 1024 * 1024
MAX_IMAGE = 32 * 1024 * 1024
MAX_PIXELS = 36_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS
MODES = ("phantomseal_faceshield", "phantomseal", "faceshield")
DEFAULTS = {"mode": "phantomseal_faceshield", "max_size": 1280, "iterations": 30,
            "noise_clamp": 12, "step_size": 1, "phantom_iterations": 1000}
LOGGER = logging.getLogger("faceshield.editor")


class APIError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def number(value, name, low, high, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise APIError(f"{name}: a finite number is required.")
    if not low <= value <= high or integer and int(value) != value:
        raise APIError(f"{name}: expected {'an integer' if integer else 'a number'} between {low} and {high}.")
    return int(value) if integer else float(value)


def parameters(raw):
    if not isinstance(raw, dict):
        raise APIError("params must be an object.")
    mode = raw.get("mode", DEFAULTS["mode"])
    if not isinstance(mode, str) or mode not in MODES:
        raise APIError("mode must be phantomseal_faceshield, phantomseal or faceshield.")
    return {
        "mode": mode,
        "max_size": number(raw.get("max_size", 1280), "max_size", 128, 1280, True),
        "iterations": number(raw.get("iterations", 30), "iterations", 1, 200, True),
        "noise_clamp": number(raw.get("noise_clamp", 12), "noise_clamp", 1, 32, True),
        "step_size": number(raw.get("step_size", 1), "step_size", 0.25, 4),
        "phantom_iterations": number(raw.get("phantom_iterations", 1000), "phantom_iterations", 10, 2000, True),
    }


def compare_arrays(a, b):
    """Pixel metrics and Gaussian-window RGB SSIM; these do not measure protection efficacy."""
    if a.shape != b.shape:
        raise APIError("Image dimensions differ. Comparison requires images of the same size.")
    difference = a[..., :3].astype(np.float32) - b[..., :3].astype(np.float32)
    mse = float(np.square(difference).mean())
    changed = np.any(a != b, axis=2)
    identical = not bool(changed.any())
    score = 1.0 if identical else None
    if not identical and min(a.shape[:2]) >= 11:
        import cv2
        aa, bb = a[..., :3].astype(np.float32), b[..., :3].astype(np.float32)
        mean_a = cv2.GaussianBlur(aa, (11, 11), 1.5)
        mean_b = cv2.GaussianBlur(bb, (11, 11), 1.5)
        variance_a = cv2.GaussianBlur(aa * aa, (11, 11), 1.5) - mean_a * mean_a
        variance_b = cv2.GaussianBlur(bb * bb, (11, 11), 1.5) - mean_b * mean_b
        covariance = cv2.GaussianBlur(aa * bb, (11, 11), 1.5) - mean_a * mean_b
        c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        field = ((2 * mean_a * mean_b + c1) * (2 * covariance + c2) /
                 ((mean_a ** 2 + mean_b ** 2 + c1) * (variance_a + variance_b + c2)))
        score = float(np.clip(field[5:-5, 5:-5].mean(), -1, 1))
    return {"ssim": score, "psnr_db": 10 * math.log10(255 ** 2 / mse) if mse else None,
            "mae": float(np.abs(difference).mean()), "mse": mse,
            "changed_pct": float(changed.mean() * 100), "identical": identical,
            "mean_alpha": float(np.abs(a[..., 3].astype(np.float32) - b[..., 3]).mean())}


class Studio:
    def __init__(self, root_dir=ROOT, protector_factory=None, readiness_provider=None, test_runner=None):
        self.root = Path(root_dir).resolve()
        self.store = self.root / "results" / "editor"
        self.lock = threading.RLock()
        self.cpu_lock = threading.Lock()
        self.assets, self.paths, self.jobs = {}, {}, {}
        self.active = None
        self.default_source = self.default_result = None
        self._protector = None
        self._protector_factory = protector_factory
        self._readiness_provider = readiness_provider
        self._test_runner = test_runner
        self.latest_test_job_ids = {}

    def asset(self, asset_id):
        with self.lock:
            if not isinstance(asset_id, str) or asset_id not in self.assets:
                raise APIError("Image not found. Please upload it again.", 404)
            return dict(self.assets[asset_id])

    def asset_path(self, asset_id):
        self.asset(asset_id)
        with self.lock:
            return self.paths[asset_id]

    def register(self, path, kind, name=None):
        path = Path(path).resolve()
        if not path.is_relative_to(self.store):
            raise ValueError("Only editor-owned image files can be registered.")
        with Image.open(path) as image:
            width, height = image.size
        asset_id = secrets.token_hex(16)
        asset = {"id": asset_id, "name": name or path.name, "url": f"/api/image/{asset_id}",
                 "width": width, "height": height, "kind": kind}
        with self.lock:
            self.assets[asset_id], self.paths[asset_id] = asset, path
        return dict(asset)

    def image(self, asset_id):
        with Image.open(self.asset_path(asset_id)) as image:
            return image.convert("RGBA")

    def state(self):
        if self._readiness_provider is not None:
            engines = self._readiness_provider(self.root)
        elif self._protector_factory is not None:
            engines = {name: {"ready": True, "message": "Injected test runtime is available."}
                       for name in ("phantomseal", "faceshield")}
        else:
            # This helper only checks local files; it must not create GPU models.
            from protection_pipeline import readiness
            engines = readiness(self.root)
        ready = any(engine.get("ready", False) for engine in engines.values())
        testing = self.testing_readiness()
        with self.lock:
            loaded = bool(self._protector is not None and getattr(self._protector, "_loaded", False))
            return {"runtime": {"ready": ready, "loaded": loaded}, "ready": ready, "engines": engines,
                    "defaults": dict(DEFAULTS), "active_job": self.active, "active_job_id": self.active,
                    "active_job_kind": self.jobs.get(self.active, {}).get("kind"),
                    "testing": testing, "testable_result_ids": self._testable_result_ids(),
                    "latest_test_job_ids": dict(self.latest_test_job_ids),
                    "assets": [dict(item) for item in self.assets.values()],
                    "default_source_id": self.default_source, "default_result_id": self.default_result,
                    "source": self.assets.get(self.default_source), "result": self.assets.get(self.default_result)}

    def testing_readiness(self):
        from protection_test import target_options, testing_readiness
        if self._test_runner is not None:
            return {"ready": True, "message": "Injected test runtime is available.", "targets": target_options()}
        return testing_readiness(self.root)

    def _testable_result_ids(self):
        return [job["result"]["id"] for job in self.jobs.values()
                if job.get("kind") == "protect" and job["status"] == "done" and job.get("result")]

    def upload(self, payload):
        role = payload.get("role", "source")
        if role not in ("source", "result"):
            raise APIError("role must be source or result.")
        name = payload.get("name", "image.png")
        if not isinstance(name, str):
            raise APIError("name must be a string.")
        name = Path(name.replace("\\", "/")).name[:160] or "image.png"
        data = payload.get("data")
        if not isinstance(data, str) or not data.startswith("data:image/") or ";base64," not in data[:128]:
            raise APIError("A base64-encoded image data URL is required.")
        try:
            decoded = base64.b64decode(data.split(",", 1)[1], validate=True)
            if len(decoded) > MAX_IMAGE:
                raise APIError("File is too large. Maximum: 32 MB.", 413)
            with Image.open(io.BytesIO(decoded)) as source:
                if source.width * source.height > MAX_PIXELS:
                    raise APIError("Image is too large. Maximum: 36 megapixels.", 413)
                image = ImageOps.exif_transpose(source).convert("RGBA")
                image.load()
        except (binascii.Error, OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
            raise APIError("Cannot read this image. Use PNG, JPEG or WebP up to 36 megapixels.") from exc
        folder = self.store / "assets"
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"{secrets.token_hex(16)}.png"
        image.save(path, format="PNG")
        asset = self.register(path, role, name)
        with self.lock:
            if role == "source":
                self.default_source, self.default_result = asset["id"], None
            else:
                self.default_result = asset["id"]
        return asset

    def compare(self, source_id, result_id):
        with self.cpu_lock:
            a, b = self.image(source_id), self.image(result_id)
            if a.size != b.size:
                raise APIError("Image dimensions differ. Comparison requires images of the same size.")
            return compare_arrays(np.asarray(a), np.asarray(b))

    def process(self, payload):
        source = self.asset(payload.get("source_id"))
        params = parameters(payload.get("params", {}))
        with self.lock:
            if self.active is not None:
                raise APIError("Another image is being processed. Wait for it to finish.", 409)
            job_id = secrets.token_hex(16)
            job = {"id": job_id, "kind": "protect", "status": "queued", "stage": "Queued", "progress": 0,
                   "elapsed": 0.0, "source_id": source["id"], "source": None, "result": None,
                   "stage1": None, "method": params["mode"], "stages": [],
                   "metrics": None, "error": None, "params": params, "_started": time.monotonic()}
            self.jobs[job_id] = job
            self.active = job_id
        threading.Thread(target=self._run_job, args=(job_id, source, params), daemon=True,
                         name=f"faceshield-{job_id[:8]}").start()
        return {"id": job_id, "status": "queued", "kind": "protect"}

    def test(self, payload):
        from protection_test import TARGETS
        if not isinstance(payload, dict) or set(payload) - {"result_id", "target_key", "jpeg95"}:
            raise APIError("Use result_id, target_key and jpeg95 only; local paths are not accepted.")
        result_id, target_key = payload.get("result_id"), payload.get("target_key")
        jpeg95 = payload.get("jpeg95", True)
        if not isinstance(result_id, str) or not result_id:
            raise APIError("result_id must identify a completed protection result.")
        if not isinstance(target_key, str) or target_key not in TARGETS:
            raise APIError("target_key must be cage, wlh or zxy.")
        if type(jpeg95) is not bool:
            raise APIError("jpeg95 must be a boolean.")
        with self.lock:
            if self.active is not None:
                raise APIError("Another protection or test is running. Wait for it to finish.", 409)
            protect = next((job for job in self.jobs.values()
                if job.get("kind") == "protect" and job["status"] == "done"
                and job.get("result", {}).get("id") == result_id), None)
            if protect is None:
                raise APIError("Only a completed protection result from this editor can be tested.", 404)
            readiness = self.testing_readiness()
            if not readiness["ready"]:
                raise APIError("SimSwap test is unavailable. " + readiness["message"], 503)
            mode = protect["params"]["mode"]
            source = self.asset(protect["source"]["id"])
            result = self.asset(protect["result"]["id"])
            stage1 = self.asset(protect["stage1"]["id"]) if mode == "phantomseal_faceshield" and protect.get("stage1") else None
            if mode == "phantomseal_faceshield" and stage1 is None:
                raise APIError("The completed chain has no PhantomSeal intermediate image.")
            job_id = secrets.token_hex(16)
            self.jobs[job_id] = {"id": job_id, "kind": "test", "status": "queued", "stage": "Queued",
                "progress": 0, "elapsed": 0.0, "error": None, "test": None, "result_id": result_id,
                "protection_job_id": protect["id"], "target_key": target_key, "jpeg95": jpeg95,
                "_started": time.monotonic()}
            self.active = job_id
            self.latest_test_job_ids[result_id] = job_id
        threading.Thread(target=self._run_test, args=(job_id, source, result, stage1, mode, target_key, jpeg95),
                         daemon=True, name=f"faceshield-test-{job_id[:8]}").start()
        return {"id": job_id, "status": "queued", "kind": "test"}

    def _run_test(self, job_id, source, result, stage1, mode, target_key, jpeg95):
        from protection_test import METRIC, NOTE, TARGETS, finite_score, run_protection_test
        def update(**values):
            with self.lock:
                self.jobs[job_id].update(values)
        folder = (self.store / "tests" / job_id).resolve()
        def register_test(path, kind, name):
            if not Path(path).resolve().is_relative_to(folder):
                raise ValueError("Test output must stay inside its editor-owned test folder.")
            return self.register(path, kind, name)
        try:
            update(status="running", stage="Preparing SimSwap test")
            def progress(fraction, description):
                update(progress=max(0, min(98, float(fraction) * 98)), stage=str(description))
            report = (self._test_runner or run_protection_test)(root_dir=self.root, output_dir=folder,
                source_path=self.asset_path(source["id"]), result_path=self.asset_path(result["id"]),
                stage1_path=self.asset_path(stage1["id"]) if stage1 else None, mode=mode,
                target_key=target_key, jpeg95=jpeg95, progress=progress)
            target = register_test(report["target_path"], "test_target", TARGETS[target_key]["label"] + ".png")
            conditions = []
            expected_keys = ["clean", "stage1", "final"] if mode == "phantomseal_faceshield" else ["clean", "final"]
            if [condition["name"] for condition in report["conditions"]] != (["png", "jpeg95"] if jpeg95 else ["png"]):
                raise ValueError("Unexpected SimSwap test conditions.")
            for condition in report["conditions"]:
                if [row["key"] for row in condition["rows"]] != expected_keys:
                    raise ValueError("Unexpected SimSwap test variants.")
                rows = []
                for row in condition["rows"]:
                    swap = register_test(row["swap_path"], "test_swap", condition["name"] + "_" + row["key"] + "_swap.png") if row.get("swap_path") else None
                    detected = row.get("source_detected") is True and row.get("swap_detected") is True
                    rows.append({"key": row["key"], "label": row["label"], "swap": swap,
                        "score": finite_score(row.get("score")) if detected else None, "source_detected": row.get("source_detected") is True,
                        "swap_detected": row.get("swap_detected") is True, "status": str(row.get("status", "unknown"))})
                delta = None
                if mode == "phantomseal_faceshield" and all(row["score"] is not None and row["source_detected"] and row["swap_detected"] for row in rows):
                    delta = rows[-1]["score"] - rows[1]["score"]
                conditions.append({"name": condition["name"], "rows": rows, "delta_final_minus_stage1": delta})
            elapsed = round(time.monotonic() - self.jobs[job_id]["_started"], 1)
            update(status="done", stage="Test complete", progress=100, elapsed=elapsed,
                   test={"target": target, "conditions": conditions, "metric": METRIC, "note": NOTE})
        except Exception as exc:
            LOGGER.exception("FaceShield test job %s failed", job_id)
            message = str(exc) or type(exc).__name__
            if "out of memory" in message.lower():
                message = "GPU memory is full. Close other GPU applications and try the test again. " + message
            update(status="error", stage="Test failed", error=message[:1800],
                   elapsed=round(time.monotonic() - self.jobs[job_id]["_started"], 1))
        finally:
            with self.lock:
                if self.active == job_id:
                    self.active = None

    def job(self, job_id):
        with self.lock:
            if job_id not in self.jobs:
                raise APIError("Job not found.", 404)
            job = dict(self.jobs[job_id])
            if job["status"] in ("running", "queued"):
                job["elapsed"] = round(time.monotonic() - job["_started"], 1)
            job.pop("_started", None)
            return job

    def _run_job(self, job_id, original, params):
        def update(**values):
            with self.lock:
                self.jobs[job_id].update(values)

        try:
            update(status="running", stage="Preparing protection pipeline")
            if self._protector is None:
                if self._protector_factory:
                    self._protector = self._protector_factory()
                else:
                    from protection_pipeline import ProtectionPipeline
                    self._protector = ProtectionPipeline(root_dir=self.root)

            def progress(fraction, description):
                update(progress=max(0, min(98, float(fraction) * 98)), stage=str(description))

            report = self._protector.protect(
                str(self.asset_path(original["id"])), str(self.store / "jobs" / job_id),
                **params, analysis_size=512, progress=progress)
            source = self.register(report["source_path"], "source", original["name"])
            result = self.register(report["output_path"], "result", f"{Path(original['name']).stem}_protected.png")
            if (source["width"], source["height"]) != (result["width"], result["height"]):
                raise ValueError("The protected output dimensions do not match its reference image.")
            stage1 = None
            if report.get("stage1_path"):
                stage1 = self.register(report["stage1_path"], "intermediate",
                                       f"{Path(original['name']).stem}_phantomseal.png")
                if (source["width"], source["height"]) != (stage1["width"], stage1["height"]):
                    raise ValueError("The PhantomSeal output dimensions do not match the clean reference image.")
            update(stage="Comparing image pixels", progress=99)
            metrics = self.compare(source["id"], result["id"])
            performance = {key: value for key, value in report.items()
                           if key not in ("output_path", "source_path", "stage1_path")}
            with self.lock:
                self.default_source, self.default_result = source["id"], result["id"]
                elapsed = round(time.monotonic() - self.jobs[job_id]["_started"], 1)
            update(status="done", stage="Done", progress=100, source=source, result=result, stage1=stage1,
                   method=report.get("method", params["mode"]), stages=report.get("stages", []),
                   metrics=metrics, performance=performance, elapsed=elapsed)
        except Exception as exc:
            LOGGER.exception("FaceShield job %s failed", job_id)
            message = str(exc) or type(exc).__name__
            if "out of memory" in message.lower():
                message = "GPU memory is full. Close other GPU applications and try again. " + message
            with self.lock:
                elapsed = round(time.monotonic() - self.jobs[job_id]["_started"], 1)
            update(status="error", stage="Processing failed", error=message[:1800], elapsed=elapsed)
        finally:
            with self.lock:
                if self.active == job_id:
                    self.active = None


async def read_payload(request):
    if request.headers.get("content-type", "").split(";", 1)[0].strip() != "application/json":
        raise APIError("Content-Type: application/json is required.", 415)
    if request.headers.get("transfer-encoding"):
        raise APIError("Send Content-Length; Transfer-Encoding is not supported.")
    try:
        length = int(request.headers.get("content-length", "0"))
    except ValueError as exc:
        raise APIError("Invalid request length.") from exc
    if not 0 < length <= MAX_BODY:
        raise APIError("Request is empty or too large. Maximum: 48 MB.", 413)
    chunks, total = [], 0
    async for chunk in request.stream():
        total += len(chunk)
        if total > MAX_BODY or total > length:
            raise APIError("Request is too large.", 413)
        chunks.append(chunk)
    if total != length:
        raise APIError("Incomplete request body.")
    try:
        payload = json.loads(b"".join(chunks))
    except (ValueError, UnicodeError) as exc:
        raise APIError("Invalid JSON.") from exc
    if not isinstance(payload, dict):
        raise APIError("A JSON object is required.")
    return payload


def create_app(root_dir=ROOT, studio=None, port=7876):
    studio = studio or Studio(root_dir)
    web = Path(root_dir).resolve() / "web"
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
    app.state.studio = studio

    @app.middleware("http")
    async def local_only(request, call_next):
        host = request.headers.get("host", "")
        try:
            parsed = urlsplit("http://" + host)
            allowed = (parsed.hostname in ("localhost", "127.0.0.1", "::1") and parsed.port == port
                       and not parsed.username and not parsed.password)
        except ValueError:
            allowed = False
        origin = request.headers.get("origin")
        if (not allowed or origin and origin != "http://" + host or
                request.headers.get("sec-fetch-site") == "cross-site"):
            return JSONResponse({"error": "Only requests from this local editor are allowed."}, status_code=403)
        response = await call_next(request)
        response.headers.update({"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff",
                                 "X-Frame-Options": "DENY"})
        return response

    @app.exception_handler(APIError)
    async def api_error(request, exc):
        return JSONResponse({"error": str(exc)}, status_code=exc.status)

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        return JSONResponse({"error": str(exc.detail)}, status_code=exc.status_code)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return JSONResponse({"error": "Invalid request."}, status_code=400)

    @app.exception_handler(Exception)
    async def unexpected_error(request, exc):
        LOGGER.exception("Editor API failed")
        return JSONResponse({"error": "The request failed. See the FaceShield terminal for details."}, status_code=500)

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/api/state")
    @app.get("/api/status")
    def state():
        return studio.state()

    @app.post("/api/refresh")
    async def refresh(request: Request):
        await read_payload(request)
        return studio.state()

    @app.post("/api/upload")
    async def upload(request: Request):
        return await run_in_threadpool(studio.upload, await read_payload(request))

    @app.post("/api/compare")
    async def compare(request: Request):
        data = await read_payload(request)
        return await run_in_threadpool(studio.compare, data.get("source_id"), data.get("result_id"))

    @app.post("/api/process", status_code=202)
    async def process(request: Request):
        return studio.process(await read_payload(request))

    @app.post("/api/test", status_code=202)
    async def test(request: Request):
        return studio.test(await read_payload(request))

    @app.get("/api/job/{job_id}")
    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        return studio.job(job_id)

    @app.get("/api/image/{asset_id}")
    @app.get("/assets/{asset_id}")
    def image(asset_id: str):
        return FileResponse(studio.asset_path(asset_id), media_type="image/png")

    @app.get("/api/pixels/{asset_id}")
    def pixels(asset_id: str):
        return Response(studio.image(asset_id).tobytes(), media_type="application/octet-stream")

    @app.get("/favicon.ico")
    def favicon():
        return Response(status_code=204)

    @app.get("/{path:path}")
    def static(path: str):
        allowed = {"": ("index.html", "text/html"), "index.html": ("index.html", "text/html"),
                   "app.js": ("app.js", "text/javascript"), "style.css": ("style.css", "text/css"),
                   "styles.css": ("style.css", "text/css")}
        if path not in allowed:
            raise APIError("Not found.", 404)
        name, mime = allowed[path]
        file = web / name
        if not file.is_file():
            raise APIError("Editor files are missing.", 404)
        return FileResponse(file, media_type=mime)

    return app


app = create_app()


def main():
    import uvicorn
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="FaceShield local canvas editor")
    parser.add_argument("--host", default="127.0.0.1", choices=("127.0.0.1", "localhost", "::1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("FACESHIELD_PORT", "7876")))
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535.")
    application = create_app(port=args.port)
    url = f"http://{'[::1]' if args.host == '::1' else args.host}:{args.port}"
    print(f"FaceShield Studio: {url}\nStop: Ctrl+C", flush=True)
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    uvicorn.run(application, host=args.host, port=args.port, access_log=False)


if __name__ == "__main__":
    main()
