"""Download the fixed ATML PA2 course assets."""

from pathlib import Path
from huggingface_hub import snapshot_download

HF_ASSET_REPO = "AbDu11aHHH/ATML-PA2-assets"
HF_ASSET_REVISION = "0b350481fb03f5525a35bcdec4131bd4fe487f98"

REPO_ROOT = Path(__file__).resolve().parents[1]


def main():
    print("Downloading ATML PA2 course assets...")
    print("Source:", HF_ASSET_REPO)
    print("Revision:", HF_ASSET_REVISION)

    snapshot_download(
        repo_id=HF_ASSET_REPO,
        repo_type="dataset",
        revision=HF_ASSET_REVISION,
        local_dir=str(REPO_ROOT),
        allow_patterns=[
            "data/**",
            "cached/**",
            "checkpoints/**",
            "manifests/**",
        ],
    )

    print("\nCourse assets downloaded.")
    print("Now run: python -m scripts.validate_assets")


if __name__ == "__main__":
    main()
