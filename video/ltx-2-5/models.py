from pathlib import Path

from huggingface_hub import hf_hub_download

RUNPOD_COMFYUI = Path("/workspace/runpod-slim/ComfyUI")

MODELS = {
    "Lightricks/LTX-2.5": [
        "diffusion_models/ltx-2.5-22b-distilled-transformer-comfy-int8-convrot.safetensors",
        "text_encoders/gemma4-12b-with-proj-ltx-2.5-comfy-int8-convrot.safetensors",
        "vae/ltx-2.5-video-vae-bf16.safetensors",
        "vae/ltx-2.5-audio-vae-bf16.safetensors",
        "latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
    ],
}


def main():
    comfyui = next(
        (path for path in (Path.cwd(), RUNPOD_COMFYUI)
         if (path / "main.py").is_file() and (path / "folder_paths.py").is_file()),
        None,
    )
    if comfyui is None:
        raise SystemExit("ComfyUI not found. Run this script from your ComfyUI root directory.")

    models_dir = comfyui / "models"
    print(f"Model destination: {models_dir}", flush=True)

    for repo, filenames in MODELS.items():
        for filename in filenames:
            target = models_dir / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            print(f"\nDownloading: {target}", flush=True)
            # filename includes exactly one ComfyUI category folder.
            # Hugging Face preserves that path beneath local_dir.
            hf_hub_download(repo_id=repo, filename=filename, local_dir=str(models_dir))
            if not target.is_file() or target.stat().st_size == 0:
                raise SystemExit(f"Download missing or empty at expected location: {target}")
            print(f"Ready: {target}")
    print(f"\nAll five models are ready in {models_dir}. Refresh ComfyUI's model list.")


if __name__ == "__main__":
    main()
