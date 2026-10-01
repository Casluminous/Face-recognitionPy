"""SecurityService + UpdateService — port of Java versions (fail-closed)."""
from __future__ import annotations

import hashlib
import hmac
import logging
import platform
import re
import subprocess
import tempfile
import threading
from collections.abc import Callable
from pathlib import Path

import requests

from ..config import AppConfig

log = logging.getLogger(__name__)
_SHA_RE = re.compile(r"[0-9a-fA-F]{64}")


def calculate_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_checksum(path: Path, expected: str) -> bool:
    exp = (expected or "").strip().lower().replace(" ", "")
    if not re.fullmatch(r"[0-9a-f]{64}", exp):
        return False
    actual = calculate_sha256(path)
    return hmac.compare_digest(actual, exp)


def verify_signature(path: Path, subject: str = "", thumbprint: str = "") -> tuple[bool, str]:
    if path.suffix.lower() != ".exe" or not path.is_file():
        return False, "Not an exe file"
    if platform.system() != "Windows":
        return False, "Authenticode only supported on Windows"
    ps = (
        "$p=$env:FACE_UPDATE_PATH; "
        "$s=Get-AuthenticodeSignature -LiteralPath $p; "
        '"STATUS="+$s.Status; "SUBJECT="+$s.SignerCertificate.Subject; "THUMBPRINT="+$s.SignerCertificate.Thumbprint'
    )
    import os
    env = dict(os.environ)
    env["FACE_UPDATE_PATH"] = str(path)
    try:
        proc = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                              capture_output=True, text=True, timeout=10, env=env)
        out = proc.stdout or ""
        m_status = re.search(r"STATUS=(.*)", out)
        status = m_status.group(1).strip() if m_status else ""
        if status != "Valid":
            return False, f"Invalid signature status: {status}"
        if subject:
            m_sub = re.search(r"SUBJECT=(.*)", out)
            if not m_sub or subject not in m_sub.group(1):
                return False, "Signer subject mismatch"
        if thumbprint:
            m_th = re.search(r"THUMBPRINT=(.*)", out)
            norm = lambda s: re.sub(r"\s+", "", s).upper()
            if not m_th or norm(m_th.group(1)) != norm(thumbprint):
                return False, "Thumbprint mismatch"
        return True, "Valid"
    except Exception as e:
        return False, str(e)


def verify_downloaded_file(path: Path, expected_hash: str, cfg: AppConfig | None = None) -> tuple[bool, str]:
    cfg = cfg or AppConfig.get_instance()
    if path.suffix.lower() != ".exe":
        return False, "Not an exe file"
    if not verify_checksum(path, expected_hash):
        return False, "Checksum mismatch"
    if cfg.require_authenticode:
        ok, msg = verify_signature(path, cfg.get_string("update.trusted.signer.subject", ""),
                                   cfg.get_string("update.trusted.signer.thumbprint", ""))
        if not ok:
            return False, msg
    return True, "OK"


_ASSET_RE = re.compile(r'"name"\s*:\s*"([^"]+)"[^}]*?"browser_download_url"\s*:\s*"([^"]+)"', re.DOTALL)
_PUBLISHED_RE = re.compile(r'"published_at"\s*:\s*"([^"]+)"')
_HASH_LINE_RE = re.compile(r"([0-9a-fA-F]{64})\s+\*?(\S+)?")


def parse_checksum(content: str, exe_name: str) -> str:
    for m in _HASH_LINE_RE.finditer(content):
        _, fname = m.group(1), (m.group(2) or "")
        if not fname or fname == exe_name or exe_name in fname or fname in exe_name:
            return m.group(1).lower()
    m = _SHA_RE.search(content)
    return m.group(0).lower() if m else ""


class UpdateService:
    def __init__(self, cfg: AppConfig | None = None):
        self.cfg = cfg or AppConfig.get_instance()
        self.update_available = False
        self.download_url = ""
        self.expected_checksum = ""
        self.latest_release_time = ""

    def check_for_updates(self, on_found: Callable[[str], None] | None = None,
                          on_error: Callable[[str], None] | None = None) -> None:
        def _run():
            try:
                url = self.cfg.update_api_url
                if not url.startswith("https://"):
                    raise ValueError("Update URL must be HTTPS")
                r = requests.get(url, headers={"Accept": "application/vnd.github+json",
                                               "User-Agent": "FaceRecognition-App"},
                                 timeout=self.cfg.update_check_timeout)
                r.raise_for_status()
                body = r.text
                assets = _ASSET_RE.findall(body)
                exe_url = next((u for n, u in assets if n.lower().endswith(".exe")), "")
                sum_name = next((n for n, u in assets
                                 if n.lower().endswith((".sha256", ".sha256.txt")) or "checksum" in n.lower()), "")
                sum_url = next((u for n, u in assets if n == sum_name), "")
                if not exe_url or not sum_url:
                    if on_error:
                        on_error("Release missing exe or checksum asset (fail-closed)")
                    return
                exe_name = exe_url.rsplit("/", 1)[-1].split("?", 1)[0]
                cr = requests.get(sum_url, timeout=self.cfg.update_check_timeout)
                cr.raise_for_status()
                checksum = parse_checksum(cr.text, exe_name)
                if not checksum:
                    if on_error:
                        on_error("Checksum asset invalid")
                    return
                self.update_available = True
                self.download_url = exe_url
                self.expected_checksum = checksum
                m = _PUBLISHED_RE.search(body)
                self.latest_release_time = m.group(1) if m else ""
                if on_found:
                    on_found(exe_url)
            except Exception as e:
                log.warning("Update check failed: %s", e)
                if on_error:
                    on_error(str(e))

        threading.Thread(target=_run, daemon=True).start()

    def download_and_install(self, url: str, on_progress: Callable[[int], None] | None = None,
                             on_complete: Callable[[str], None] | None = None,
                             on_error: Callable[[str], None] | None = None) -> None:
        def _run():
            try:
                if url != self.download_url or not self.expected_checksum:
                    raise ValueError("URL mismatch or missing checksum (fail-closed)")
                tmp = Path(tempfile.gettempdir()) / f"FaceRecognition_Setup_{hash(url) & 0xffff}.exe"
                with requests.get(url, stream=True, timeout=self.cfg.update_download_timeout) as r:
                    r.raise_for_status()
                    total = int(r.headers.get("Content-Length", 0) or 0)
                    done = 0
                    with open(tmp, "wb") as f:
                        for chunk in r.iter_content(65536):
                            if chunk:
                                f.write(chunk)
                                done += len(chunk)
                                if on_progress and total:
                                    on_progress(min(100, int(done * 100 / total)))
                if tmp.stat().st_size < 10 * 1024 * 1024:
                    tmp.unlink(missing_ok=True)
                    raise ValueError("Downloaded file too small")
                ok, msg = verify_downloaded_file(tmp, self.expected_checksum, self.cfg)
                if not ok:
                    tmp.unlink(missing_ok=True)
                    raise ValueError(msg)
                import subprocess as sp
                sp.Popen([str(tmp)])
                if on_complete:
                    on_complete(str(tmp))
            except Exception as e:
                if on_error:
                    on_error(str(e))

        threading.Thread(target=_run, daemon=True).start()
