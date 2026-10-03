"""Final SPEED-Bench metrics and completion validation."""

import json
import math
import statistics
from pathlib import Path

from .config import DOMAINS, digest


def _read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def aggregate(summaries, settings):
    if len(summaries) != settings["repeats"] or [r["seed"] for r in summaries] != settings["seeds"]:
        raise ValueError("All requested repetitions must finish before aggregation")
    block = {
        "measurement_mode": settings["measurement_mode"],
        "timing_valid": True,
        "num_gen_failed": 0,
        "num_judge_failed": 0,
        "repeats": summaries,
        "num_entries": summaries[0]["num_entries"],
        "num_completed": summaries[0]["num_entries"],
    }
    for key in ("total_generated_tokens", "generation_seconds"):
        block[key] = sum(r[key] for r in summaries)
    for key in ("output_tokens_per_second", "spec_acceptance_length", "ttft_seconds", "decode_step_seconds"):
        if all(key in r for r in summaries):
            values = [r[key] for r in summaries]
            block[key] = statistics.mean(values)
            block[key + "_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
    categories = summaries[0]["category_al"]
    block["category_al"] = {c: statistics.mean(r["category_al"][c] for r in summaries) for c in categories}
    return block


def _matches_aggregate(block, repeats, settings):
    expected = aggregate(repeats, settings)
    if block.keys() != expected.keys():
        return False
    for key, value in expected.items():
        saved = block[key]
        if key.endswith("_std"):
            if type(saved) not in (int, float) or not math.isclose(
                saved, value, rel_tol=1e-12, abs_tol=1e-15
            ):
                return False
        elif saved != value:
            return False
    return True


def read_report(root, config, failure=None):
    root = Path(root)
    settings = config["speed_bench"]
    count = settings.get("repeats", 0)
    directory = root / "eval-results/speed-bench"
    files = [directory / f"output-rs{i}.jsonl" for i in range(count)]
    completed = sum(p.is_file() and Path(str(p) + ".done").is_file() for p in files)
    engine = settings.get("engine", {})
    generation = settings.get("generation", {})
    spec = engine.get("speculative_config") or {}
    mode = settings.get("measurement_mode")
    conditions = {
        "model": settings.get("model"),
        "backend": "vllm",
        "attention_backend": engine.get("attention_backend"),
        "draft_model": spec.get("model"),
        "speculative_tokens": settings.get("speculative_tokens"),
        "draft_sample_method": spec.get("draft_sample_method"),
        "adaptive_verification": spec.get("enable_adaptive_verification", False),
        "max_tokens": generation.get("tokens_to_generate"),
        "ignore_eos": False,
        "concurrency": settings.get("concurrency"),
        "tp": engine.get("tensor_parallel_size", 1),
        "pp": engine.get("pipeline_parallel_size", 1),
        "expert_parallel": engine.get("enable_expert_parallel", False),
        "dp": 1,
        "replicas": 1,
        "temperature": generation.get("temperature"),
        "top_p": generation.get("top_p"),
        "top_k": generation.get("top_k"),
        "max_model_len": engine.get("max_model_len"),
        "chat_template": settings.get("chat_template"),
        "chat_template_kwargs": generation.get("chat_template_kwargs"),
    }
    comparison = {
        k: v
        for k, v in settings.items()
        if k
        not in {
            "speculative_tokens",
            "measurement_mode",
            "extra_server_args",
            "engine",
            "max_model_len",
            "gpu_memory_utilization",
        }
    }
    comparison["engine"] = {k: v for k, v in engine.items() if k != "speculative_config"}
    prompts = directory / "initial-prompts.json"
    comparison["initial_prompts_sha256"] = digest(prompts.read_text()) if prompts.is_file() else None
    report = {
        "run_id": digest(str(root.resolve())),
        "path": str(root),
        "status": "incomplete",
        "reason": "Waiting for all requested repetitions",
        "repeat_done": completed,
        "repeat_total": count,
        "measurement_mode": mode,
        "conditions": conditions,
        "variant": (f"{spec.get('method')} ({spec.get('draft_sample_method', 'default')})" if spec else "baseline"),
        "comparison_key": digest(comparison),
        "metrics": {},
    }
    execution = _read_json(root / "logs/run-status.json")
    if (
        failure
        or execution.get("state") in {"failed", "cancelled"}
        or execution.get("tasks", {}).get("speed-bench", {}).get("state") == "failed"
    ):
        report.update(status="failed", reason=failure or "Run failed or cancelled")
        return report
    manifest = _read_json(root / "logs/speed-bench-manifest.json")
    if manifest.get("settings_sha256") != digest(settings):
        report.update(status="invalid", reason="SPEED-Bench measurement settings do not match their manifest")
        return report
    key = "pass@1" if count == 1 else f"pass@1[avg-of-{count}]"
    path = directory / "metrics.json"
    block = _read_json(path).get("speed-bench", {}).get(key, {})
    if not count or completed != count or not block:
        return report
    repeats = block.get("repeats", [])
    try:
        if len(repeats) != count or [r["seed"] for r in repeats] != settings["seeds"]:
            raise ValueError("Missing or duplicate repetitions")
        for i, row in enumerate(repeats):
            if row != _read_json(directory / f"repeat-{i}/complete.json"):
                raise ValueError("Aggregate differs from the saved repetition")
            if files[i].stat().st_mtime_ns > path.stat().st_mtime_ns:
                raise ValueError("Metrics predate outputs")
            if (
                row["num_entries"] != 880
                or row["num_completed"] != 880
                or row["num_turns"] != 1138
                or row["num_gen_failed"] != 0
            ):
                raise ValueError("Incomplete Qualitative coverage")
            if set(row["category_al"]) != DOMAINS:
                raise ValueError("Incomplete domain coverage")
            lengths = [row["spec_acceptance_length"], *row["category_al"].values()]
            if any(not math.isfinite(v) or not 1 <= v <= settings["speculative_tokens"] + 1 for v in lengths):
                raise ValueError("Acceptance length is outside native decode limits")
            for name in (
                "generation_seconds",
                "output_tokens_per_second",
                "total_generated_tokens",
                "spec_acceptance_length",
            ):
                if not math.isfinite(row[name]) or row[name] <= 0:
                    raise ValueError("Invalid timing or acceptance length")
            if not math.isclose(
                row["output_tokens_per_second"], row["total_generated_tokens"] / row["generation_seconds"], rel_tol=1e-6
            ):
                raise ValueError("Throughput differs from tokens/time")
        for name in ("output_tokens_per_second", "spec_acceptance_length"):
            if not math.isclose(block[name], statistics.mean(r[name] for r in repeats), rel_tol=1e-9):
                raise ValueError("Invalid repetition average")
        if not _matches_aggregate(block, repeats, settings):
            raise ValueError("Final metrics differ from saved repetitions")
    except (KeyError, ValueError, TypeError, OSError, ZeroDivisionError) as exc:
        report.update(status="invalid", reason=str(exc))
        return report
    names = (
        "num_entries",
        "total_generated_tokens",
        "generation_seconds",
        "output_tokens_per_second",
        "spec_acceptance_length",
        "ttft_seconds",
        "decode_step_seconds",
        "output_tokens_per_second_std",
        "spec_acceptance_length_std",
    )
    report.update(
        status="complete",
        reason="",
        metrics={k: block[k] for k in names if k in block},
        category_al=block["category_al"],
    )
    return report
