# Rani Shield on Windows

Use [README.md](README.md) for the installation and workflow, [QUICKSTART.md](QUICKSTART.md) for Russian commands, and [weights/README.md](weights/README.md) for model placement. This page covers Windows operation and troubleshooting.

## Local runtime

`manage.cmd setup` prepares `.runtime/python/` and the project's `.venv`, with Python 3.11 and the pinned CUDA 12.8 PyTorch wheels. The setup step can copy an installed Python 3.11 and Node.js into the project. Once prepared, processing uses local interpreter and package files; sibling source repositories and the bootstrap Python are unnecessary at runtime. The host supplies Windows, PowerShell, a compatible NVIDIA driver and a browser.

The GPU configuration exercised during development is an RTX 5070 Ti with 16 GB VRAM. Missing CUDA support, an incompatible driver or insufficient VRAM can prevent processing. `manage.cmd doctor` reports the configured environment and assets; it does not load the complete model tensors.

## Server lifecycle

Run commands in the repository root:

```powershell
.\manage.cmd run
```

The default address is [http://127.0.0.1:7876](http://127.0.0.1:7876). To select another port:

```powershell
.\manage.cmd run -Port 7877
```

Both `manage.cmd run` and `make run` keep the server in the current terminal and print its logs directly. The command returns when the server exits; press **Ctrl+C** to stop it. No background server, PID record or server log files are created. Ordinary setup, model and test commands do not start an editor server.

Inspect port occupancy from another terminal:

```powershell
.\manage.cmd status
.\manage.cmd status -Port 7877
```

`run` refuses to start if the selected port is occupied or a recorded legacy server is still running. It neither reuses nor stops the existing process. `manage.cmd stop` remains available only for background servers started by an older version: it verifies the saved executable, command line, PID and creation time before stopping them. It does not manage the new foreground server.

`launch.ps1` delegates to `manage.ps1 run` with the same foreground behavior. Neither launcher opens a browser automatically. `app.py` remains a compatibility entry point.

## Common issues

| Symptom | Action |
|---|---|
| Environment missing or wrong Python version | Run `manage.cmd setup`, then `manage.cmd doctor`. Preserve an incompatible environment until you have reviewed the setup message. |
| Port already occupied | Inspect `manage.cmd status`; choose `-Port 7877` if the occupant belongs to another app. |
| A configured model path points outside the project | Copy the resource into `weights/` and use repository-relative configuration. External runtime overrides are rejected. |
| Missing PhantomSeal branch or fewer than 214 bank images | Follow the exact archive placement map in [weights/README.md](weights/README.md); `models` alone does not fetch the two full archives. |
| A model loader reports an incompatible checkpoint | Verify the intended release and manifest hash; renaming an unrelated checkpoint does not make it compatible. |
| Out of GPU memory | Close other GPU jobs, reduce the output limit, or test one protection method at a time. Each setting still needs its own validation. |
| No face detected | Use a clear portrait with a sufficiently large, visible face; the current path optimizes the largest detected face. |
| Previous session disappeared after restart | Reupload the input. The asset/job registry lives in memory, while existing PNG files remain in `results/editor/`. |
| Test result is Inconclusive | Inspect the unprotected swap first. It must pass the numerical baseline gate and the separate visual likeness confirmation. |

## Output behavior

Images preserve proportions, with a long edge up to 1280 px and no enlargement of smaller inputs. Protection operates on RGB; transparency is flattened during processing. SSIM, PSNR and changed pixels compare the normalized original with the result at the same resolution.

**Export PNG** saves the original in Original view, the full-resolution difference in Diff view and the protected result in the other views. Pan, zoom and comparison slider position are not baked into the exported photo. Generated files stay under `results/editor/`; your original file is not overwritten.

PNG preserves the generated perturbation without lossy compression. JPEG, resizing, retouching and subsequent processing can change it. Current evidence is summarized in [Validation](docs/VALIDATION.md); successful execution does not establish reliable defense.
