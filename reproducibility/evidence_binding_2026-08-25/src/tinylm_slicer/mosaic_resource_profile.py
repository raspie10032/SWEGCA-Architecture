from __future__ import annotations

import argparse
import ctypes
import gc
import json
import math
import os
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from tinylm_slicer.mosaic_phase1 import (
    Phase1Config,
    SharedDepthSequenceModel,
    profile_model,
)


MIB = 2**20


@dataclass(frozen=True)
class ResourceProfileConfig:
    batch_size: int = 1
    depths: tuple[int, ...] = (1, 2, 4, 8)
    warmups: int = 3
    repeats: int = 20
    max_combined_memory_mib: float = 2048.0

    def __post_init__(self) -> None:
        if (
            self.batch_size <= 0
            or not self.depths
            or any(depth <= 0 for depth in self.depths)
            or self.warmups < 0
            or self.repeats <= 0
            or self.max_combined_memory_mib <= 0
        ):
            raise ValueError("resource profile values are invalid")


def profile_resources(
    model_config: Phase1Config,
    resource_config: ResourceProfileConfig,
    *,
    device: str = "auto",
    checkpoint: Path | None = None,
) -> dict[str, object]:
    resolved_device = _resolve_device(device)
    before_model = process_memory()
    model = SharedDepthSequenceModel(model_config)
    if checkpoint is not None:
        payload = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=True,
        )
        if payload.get("variant") != "recurrent":
            raise ValueError("resource checkpoint must be a recurrent variant")
        if payload.get("config") != asdict(model_config):
            raise ValueError("resource checkpoint config does not match model config")
        model.load_state_dict(payload["state_dict"], strict=True)
        del payload
        gc.collect()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    runtime_weight_mib = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    ) / MIB
    after_cpu_model = process_memory()
    model = model.to(resolved_device).eval()

    if resolved_device.type == "cuda":
        torch.cuda.synchronize(resolved_device)
        torch.cuda.reset_peak_memory_stats(resolved_device)
    after_device_model = process_memory()

    start_nodes = torch.arange(
        resource_config.batch_size,
        device=resolved_device,
    ) % model_config.node_count
    operator_ids = torch.arange(
        resource_config.batch_size,
        device=resolved_device,
    ) % 2
    depth_reports = []
    with torch.inference_mode():
        for depth_value in resource_config.depths:
            depths = torch.full(
                (resource_config.batch_size,),
                depth_value,
                dtype=torch.long,
                device=resolved_device,
            )
            for _ in range(resource_config.warmups):
                model(
                    start_nodes,
                    operator_ids,
                    depths,
                    recurrent=True,
                )
            _synchronize(resolved_device)
            samples = []
            for _ in range(resource_config.repeats):
                started = time.perf_counter()
                model(
                    start_nodes,
                    operator_ids,
                    depths,
                    recurrent=True,
                )
                _synchronize(resolved_device)
                samples.append((time.perf_counter() - started) * 1000)
            depth_reports.append(
                {
                    "depth": depth_value,
                    "p50_latency_ms": round(statistics.median(samples), 3),
                    "p95_latency_ms": round(_percentile(samples, 0.95), 3),
                    "mean_latency_ms": round(statistics.fmean(samples), 3),
                    "samples": resource_config.repeats,
                }
            )

    after_inference = process_memory()
    cuda_memory = _cuda_memory(resolved_device)
    combined_peak_mib = (
        after_inference["peak_rss_mib"] + cuda_memory["max_reserved_mib"]
    )
    checks = {
        "finite_latency": all(
            math.isfinite(float(row["p95_latency_ms"]))
            and float(row["p95_latency_ms"]) > 0
            for row in depth_reports
        ),
        "combined_host_proxy_within_budget": (
            combined_peak_mib <= resource_config.max_combined_memory_mib
        ),
    }
    static_profile = profile_model(model_config)
    return {
        "schema_version": "mosaic-resource-profile-v0",
        "scope": (
            "desktop host proxy for process RSS, accelerator memory, and "
            "recurrent latency; not real 4GB mobile validation"
        ),
        "device": str(resolved_device),
        "checkpoint": str(checkpoint.resolve()) if checkpoint else None,
        "model_config": asdict(model_config),
        "resource_config": asdict(resource_config),
        "parameter_count": parameter_count,
        "runtime_weight_mib": round(runtime_weight_mib, 3),
        "projected_weight_mib": static_profile["raw_weight_mib"],
        "process_memory_mib": {
            "before_model": before_model,
            "after_cpu_model": after_cpu_model,
            "after_device_model": after_device_model,
            "after_inference": after_inference,
        },
        "cuda_memory_mib": cuda_memory,
        "combined_peak_host_proxy_mib": round(combined_peak_mib, 3),
        "depth_latency": depth_reports,
        "checks": checks,
        "host_checks_passed": all(checks.values()),
        "edge_verdict": (
            "host_budget_pass_not_device_validated"
            if all(checks.values())
            else "host_budget_failed"
        ),
    }


def process_memory() -> dict[str, float]:
    if os.name == "nt":
        return _windows_process_memory()
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        resident_pages = int(
            Path("/proc/self/statm").read_text(encoding="ascii").split()[1]
        )
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_bytes = peak if os.uname().sysname == "Darwin" else peak * 1024
        return {
            "rss_mib": round(resident_pages * page_size / MIB, 3),
            "peak_rss_mib": round(peak_bytes / MIB, 3),
        }
    except (AttributeError, OSError, ValueError):
        return {"rss_mib": 0.0, "peak_rss_mib": 0.0}


def _windows_process_memory() -> dict[str, float]:
    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.windll.kernel32
    psapi = ctypes.windll.psapi
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = (
        ctypes.c_void_p,
        ctypes.POINTER(ProcessMemoryCounters),
        ctypes.c_ulong,
    )
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    handle = kernel32.GetCurrentProcess()
    succeeded = psapi.GetProcessMemoryInfo(
        handle,
        ctypes.byref(counters),
        counters.cb,
    )
    if not succeeded:
        raise ctypes.WinError()
    return {
        "rss_mib": round(counters.WorkingSetSize / MIB, 3),
        "peak_rss_mib": round(counters.PeakWorkingSetSize / MIB, 3),
    }


def _cuda_memory(device: torch.device) -> dict[str, float]:
    if device.type != "cuda":
        return {
            "allocated_mib": 0.0,
            "reserved_mib": 0.0,
            "max_allocated_mib": 0.0,
            "max_reserved_mib": 0.0,
        }
    return {
        "allocated_mib": round(torch.cuda.memory_allocated(device) / MIB, 3),
        "reserved_mib": round(torch.cuda.memory_reserved(device) / MIB, 3),
        "max_allocated_mib": round(
            torch.cuda.max_memory_allocated(device) / MIB,
            3,
        ),
        "max_reserved_mib": round(
            torch.cuda.max_memory_reserved(device) / MIB,
            3,
        ),
    }


def _percentile(values: list[float], quantile: float) -> float:
    if not values or not 0 <= quantile <= 1:
        raise ValueError("percentile requires values and a quantile in [0, 1]")
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return resolved


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Profile MOSAIC Phase 1 memory and recurrent latency."
    )
    parser.add_argument("--preset", choices=("smoke", "target"), default="smoke")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--depths", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--max-memory-mib", type=float, default=2048.0)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    model_config = (
        Phase1Config.target() if args.preset == "target" else Phase1Config()
    )
    resource_config = ResourceProfileConfig(
        batch_size=args.batch_size,
        depths=tuple(args.depths),
        warmups=args.warmups,
        repeats=args.repeats,
        max_combined_memory_mib=args.max_memory_mib,
    )
    report = profile_resources(
        model_config,
        resource_config,
        device=args.device,
        checkpoint=args.checkpoint,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if report["host_checks_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
