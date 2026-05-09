from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


def recover_checkpoint(extracted_dir: Path, output_file: Path) -> None:
    if not extracted_dir.exists() or not extracted_dir.is_dir():
        raise FileNotFoundError(f"Extracted checkpoint folder not found: {extracted_dir}")
    if output_file.exists():
        output_file.unlink()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    temp_zip = output_file.with_suffix(output_file.suffix + ".zip")
    if temp_zip.exists():
        temp_zip.unlink()

    archive_root = output_file.stem
    checkpoint_root = extracted_dir
    if not (checkpoint_root / "data.pkl").exists():
        candidates = [path for path in extracted_dir.iterdir() if path.is_dir() and (path / "data.pkl").exists()]
        if len(candidates) != 1:
            raise FileNotFoundError(
                f"Could not find checkpoint data.pkl inside {extracted_dir}. "
                "Expected either data.pkl at the root or exactly one child folder containing it."
            )
        checkpoint_root = candidates[0]

    with zipfile.ZipFile(temp_zip, "w", compression=zipfile.ZIP_STORED) as archive:
        for path in checkpoint_root.rglob("*"):
            if path.is_file():
                archive.write(path, Path(archive_root) / path.relative_to(checkpoint_root))
    shutil.move(str(temp_zip), str(output_file))


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild a .pt file from an extracted PyTorch checkpoint folder.")
    parser.add_argument("--input", default="shoe_detect_best.pt", help="Extracted checkpoint folder.")
    parser.add_argument("--output", default="models/footwear/shoe_detect_best.pt", help="Recovered .pt output path.")
    args = parser.parse_args()
    recover_checkpoint(Path(args.input), Path(args.output))
    print(f"Recovered checkpoint: {args.output}")


if __name__ == "__main__":
    main()
