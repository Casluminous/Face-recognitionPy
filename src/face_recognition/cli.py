"""CLI — port of FaceRecognitionCli.java."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .config import AppConfig
from .model_paths import ModelPaths
from .path_validator import is_valid_image_path
from .services.face_service import FaceDetectorService


def build_service() -> FaceDetectorService:
    mp = ModelPaths.resolve()
    if not (mp.has_ssd() and mp.has_gender()):
        raise FileNotFoundError(f"Missing SSD/gender models: {mp.missing_files()}")
    return FaceDetectorService(str(mp.ssd_model_path), str(mp.ssd_config_path),
                               str(mp.gender_model_path), str(mp.gender_config_path),
                               AppConfig.get_instance())


def run_cli(input_path: str, output_path: str) -> int:
    if not is_valid_image_path(input_path):
        print(f"Invalid input path: {input_path}")
        return 2
    inp = Path(input_path)
    if not inp.is_file():
        print(f"Input not found: {input_path}")
        return 2
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    svc = build_service()
    img = cv2.imread(str(inp))
    if img is None:
        print("Failed to read image")
        return 1
    n = svc.detect_and_draw(img)
    cv2.imwrite(str(out), img)
    print(f"Done: {n} face(s) -> {out}")
    return 0


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Face recognition CLI (Python port)")
    ap.add_argument("input", help="Input image path")
    ap.add_argument("output", nargs="?", default="Image_Output/result.jpg", help="Output image path")
    args = ap.parse_args(argv)
    raise SystemExit(run_cli(args.input, args.output))


if __name__ == "__main__":
    main()
