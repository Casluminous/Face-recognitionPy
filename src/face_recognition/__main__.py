"""Entry point — python -m face_recognition (also PyInstaller entry)."""
from __future__ import annotations

# NOTE: absolute import — PyInstaller runs this file as top-level script,
# relative import (from .ui...) crashes frozen exe with
# "ImportError: attempted relative import with no known parent package".
from face_recognition.ui.main_window import run_app


def main() -> None:
    run_app()


if __name__ == "__main__":
    main()
