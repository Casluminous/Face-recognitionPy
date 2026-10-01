# -*- mode: python ; coding: utf-8 -*-
# PyInstaller one-file build for Face-Recognition-Python 1.2.4
# Usage: pyinstaller face-recognition.spec
import os

JAVA_DATA = r"C:\Users\steal\Face-Recognition\data"

datas = [
    (os.path.join(JAVA_DATA, "models", "face", "yolov8n-face.onnx"), "data/models/face"),
    (os.path.join(JAVA_DATA, "models", "face", "res10_300x300_ssd_iter_140000.caffemodel"), "data/models/face"),
    (os.path.join(JAVA_DATA, "models", "face", "deploy.prototxt"), "data/models/face"),
    (os.path.join(JAVA_DATA, "models", "gender", "gender_net.caffemodel"), "data/models/gender"),
    (os.path.join(JAVA_DATA, "models", "gender", "gender_deploy.prototxt"), "data/models/gender"),
    (os.path.join(JAVA_DATA, "haarcascade", "haarcascade_frontalface_default.xml"), "data/haarcascade"),
    (os.path.join(os.path.dirname(os.path.abspath(SPEC)), "config", "application.properties"), "config"),
]

a = Analysis(
    ["src/face_recognition/__main__.py"],
    pathex=["src"],
    binaries=[],
    datas=datas,
    hiddenimports=["onnxruntime", "cv2", "numpy", "PySide6"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "scipy", "pandas", "pytest"],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="FaceRecognition",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
