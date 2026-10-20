"""Download the public model files used by FaceShield's local research demo."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import json
import shutil
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent


def download(url, target):
    target = Path(target)
    if target.is_file() and target.stat().st_size:
        print(f"Already present: {target.relative_to(ROOT)}", flush=True)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + '.part')
    print(f"Downloading: {target.relative_to(ROOT)}", flush=True)
    request = urllib.request.Request(url, headers={'User-Agent': 'FaceShield-local-setup'})
    with urllib.request.urlopen(request, timeout=120) as response, partial.open('wb') as output:
        shutil.copyfileobj(response, output, length=1024 * 1024)
    partial.replace(target)
    print(f"Ready: {target.relative_to(ROOT)} ({target.stat().st_size / 1e6:.1f} MB)", flush=True)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    sd_repo = 'stable-diffusion-v1-5/stable-diffusion-v1-5'
    sd_files = [
        'tokenizer/vocab.json', 'tokenizer/merges.txt', 'tokenizer/tokenizer_config.json',
        'tokenizer/special_tokens_map.json', 'text_encoder/config.json',
        'text_encoder/model.fp16.safetensors', 'vae/config.json',
        'vae/diffusion_pytorch_model.fp16.safetensors', 'unet/config.json',
        'unet/diffusion_pytorch_model.fp16.safetensors',
    ]
    jobs = [(f'https://huggingface.co/{sd_repo}/resolve/main/{name}',
             ROOT / 'weights/sd15' / name) for name in sd_files]
    for name in ('config.json', 'model.safetensors'):
        jobs.append((f'https://huggingface.co/h94/IP-Adapter/resolve/main/models/image_encoder/{name}',
                     ROOT / 'weights/ip-adapter/models/image_encoder' / name))
    jobs.append(('https://huggingface.co/h94/IP-Adapter/resolve/main/models/ip-adapter_sd15.bin',
                 ROOT / 'weights/ip-adapter/ip-adapter_sd15.bin'))
    jobs.append(('https://github.com/khrlimam/mtcnn-pytorch/releases/download/0.0.1/pnet-6b6ef92b.pth',
                 ROOT / 'weights/mtcnn-attack/pnet-6b6ef92b.pth'))
    archive = ROOT / 'weights/downloads/arcface-models.zip'
    arc_files = ['arcface50_checkpoint.tar', 'arcface100_checkpoint.tar']
    if not all((ROOT / 'weights/faceshield' / name).is_file() for name in arc_files):
        jobs.append(('https://kuaicv.synology.me/weights/iccv2025/FaceShield/models.zip', archive))
    errors = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download, url, path): str(path) for url, path in jobs}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as exc:
                errors.append(f'{futures[future]}: {exc}')
    if archive.is_file():
        with zipfile.ZipFile(archive) as bundle:
            for name in arc_files:
                target = ROOT / 'weights/faceshield' / name
                if target.is_file():
                    continue
                members = [m for m in bundle.infolist() if Path(m.filename).name == name]
                if len(members) != 1:
                    errors.append(f'Expected exactly one archive entry for {name}')
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(members[0]) as source, target.open('wb') as output:
                    shutil.copyfileobj(source, output)
                print(f'Extracted: weights/faceshield/{name}', flush=True)
    if errors:
        raise RuntimeError('\n'.join(errors))
    print('Public model downloads complete.', flush=True)


if __name__ == '__main__':
    main()
