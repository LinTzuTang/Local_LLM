"""What machine the platform is running on, for the home page: a Mac, a GCP VM, or other Linux.

Read once per process (the app caches it): `system_profiler` on macOS; on Linux `nvidia-smi`,
/proc/cpuinfo, /etc/os-release and, on Google Compute Engine, the metadata server (instance
name, zone, machine type).
"""

import os
import platform
import re
import socket
import subprocess
from pathlib import Path

import httpx
import psutil

GCP_METADATA = "http://metadata.google.internal/computeMetadata/v1/instance/"


def _run(*cmd: str, timeout: float = 5) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def parse_hardware(text: str) -> dict:
    """`system_profiler SPHardwareDataType` -> {"Model Name": "MacBook Pro", "Chip": ..., ...}."""
    fields = {}
    for line in text.splitlines():
        key, sep, value = line.strip().partition(":")
        if sep and value.strip():
            fields[key.strip()] = value.strip()
    return fields


def parse_gpus(text: str) -> list[str]:
    """`nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits` -> labels."""
    gpus = []
    for line in text.strip().splitlines():
        name, _, mib = (part.strip() for part in line.partition(","))
        if name:
            gb = f" · {round(int(mib) / 1024)} GB" if mib.isdigit() else ""
            gpus.append(f"{name}{gb}")
    return gpus


def _os_release() -> str:
    try:
        text = Path("/etc/os-release").read_text()
    except OSError:
        return f"Linux {platform.release()}"
    m = re.search(r'^PRETTY_NAME="?([^"\n]+)', text, re.M)
    return m.group(1) if m else "Linux"


def _cpu_model() -> str:
    try:
        text = Path("/proc/cpuinfo").read_text()
    except OSError:
        return platform.processor()
    m = re.search(r"^model name\s*:\s*(.+)$", text, re.M)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else platform.processor()


def _gcp() -> dict | None:
    try:
        product = Path("/sys/class/dmi/id/product_name").read_text()
    except OSError:
        return None
    if "Google" not in product:
        return None
    out = {}
    try:
        with httpx.Client(timeout=1, headers={"Metadata-Flavor": "Google"}, trust_env=False) as c:
            for key in ("name", "zone", "machine-type"):
                r = c.get(GCP_METADATA + key)
                if r.status_code == 200:
                    out[key] = r.text.rsplit("/", 1)[-1]  # zone and type come as full paths
    except httpx.HTTPError:
        pass
    return out


def describe() -> dict:
    """{"kind": "mac" | "gcp" | "linux", "title", "facts": [(label key, value), ...]}.
    Label keys: chip, gpu, cpu, memory, os, zone, host (the page translates them)."""
    memory = f"{psutil.virtual_memory().total / 2**30:.0f} GB"
    host = socket.gethostname()
    if platform.system() == "Darwin":
        hw = parse_hardware(_run("system_profiler", "SPHardwareDataType"))
        chip = hw.get("Chip") or hw.get("Processor Name") or platform.machine()
        cores = hw.get("Total Number of Cores", "").split("(")[0].strip()
        facts = [
            ("chip", chip + (f" · {cores}-core CPU" if cores else "")),
            ("memory", hw.get("Memory", memory)),  # unified: shared by the CPU and the GPU
            ("os", f"macOS {platform.mac_ver()[0]}"),
            ("host", host),
        ]
        return {"kind": "mac", "title": hw.get("Model Name", "Mac"), "facts": facts}
    gpus = parse_gpus(
        _run("nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits")
    )
    facts = [("gpu", g) for g in gpus] or [("gpu", "—")]
    facts += [
        ("cpu", f"{_cpu_model()} · {os.cpu_count()} vCPU"),
        ("memory", memory),
        ("os", _os_release()),
    ]
    gcp = _gcp()
    if gcp is not None:
        zone = " · ".join(v for v in (gcp.get("zone"), gcp.get("machine-type")) if v)
        facts.insert(0, ("zone", zone or "—"))
        return {"kind": "gcp", "title": gcp.get("name") or host, "facts": facts}
    return {"kind": "linux", "title": host, "facts": facts}


def _csv_numbers(text: str) -> list[list[float]]:
    rows = []
    for line in text.strip().splitlines():
        try:
            rows.append([float(x) for x in line.split(",")])
        except ValueError:
            continue
    return rows


def gpu_memory() -> tuple[float, float] | None:
    """(used GB, total GB) over all NVIDIA GPUs, or None without one (a Mac shares its memory)."""
    out = _run(
        "nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits",
        timeout=3,
    )  # fmt: skip
    rows = [r for r in _csv_numbers(out) if len(r) == 2]
    if not rows:
        return None
    return sum(r[0] for r in rows) / 1024, sum(r[1] for r in rows) / 1024


def gpu_memory_by_pid() -> dict[int, float]:
    """GPU memory (GB) of each process using an NVIDIA GPU."""
    out = _run(
        "nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits",
        timeout=3,
    )  # fmt: skip
    usage: dict[int, float] = {}
    for row in _csv_numbers(out):
        if len(row) == 2:
            usage[int(row[0])] = usage.get(int(row[0]), 0) + row[1] / 1024
    return usage
