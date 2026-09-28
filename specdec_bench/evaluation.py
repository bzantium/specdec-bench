"""Native SPEED-Bench execution, output capture and checkpoints."""

import asyncio
import dataclasses
import importlib.metadata
import importlib.util
import json
import time
from contextlib import contextmanager
from copy import deepcopy
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

from .config import digest
from .reporting import aggregate


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n")
    temporary.replace(path)


def metrics_for(rows, folder, gpu_count=1):
    from . import datasets, metrics

    class SpecBench(metrics.SpecBench):
        def __init__(self, requests):
            # Skip optional plotting dependencies; keep upstream metric calculations.
            metrics.AcceptanceRate.__init__(self)
            self.requests = requests

        def _create_visualizations(self, *args, **kwargs):
            pass

        def _pretty_print_results(self):
            pass

    al = SpecBench([datasets.base.Request(**r) for r in rows])
    timing = metrics.Timing(gpu_count)
    al.directory = timing.directory = str(folder)
    return al, timing


class Checkpoint:
    def __init__(self, rows, folder, max_step, progress=None):
        self.rows, self.folder, self.max_step = rows, folder, max_step
        self.progress = progress
        self.seen = set()
        self.tokens = [0] * len(rows)
        self.file = (folder / "raw-turns.jsonl").open("x")

    def process_step(self, output, request_id, turn_id):
        lengths = [len(ids) for ids in output["output_ids"][0]]
        key = (request_id, turn_id)
        if key in self.seen or not lengths or any(n < 1 or n > self.max_step for n in lengths):
            raise ValueError(f"Invalid/duplicate native steps: {key}")
        times = output["token_times"]
        if len(times) < 2 or any(b < a for a, b in pairwise(times)):
            raise ValueError(f"Invalid native timestamps: {key}")
        self.seen.add(key)
        self.tokens[request_id] += sum(lengths)
        self.file.write(
            json.dumps(
                {
                    "request_id": request_id,
                    "question_id": self.rows[request_id]["question_id"],
                    "category": self.rows[request_id]["category"],
                    "turn_id": turn_id,
                    "chunk_lengths": lengths,
                    "token_times": times,
                    "output_token_ids": [t for ids in output["output_ids"][0] for t in ids],
                }
            )
            + "\n"
        )
        self.file.flush()
        if self.progress and turn_id == len(self.rows[request_id]["turns"]) - 1:
            self.progress(self.rows[request_id], self.tokens[request_id])
        if len(self.seen) == 1 or len(self.seen) % 32 == 0:
            save(
                self.folder / "progress.json",
                {
                    "turns": len(self.seen),
                    "total_turns": sum(len(r["turns"]) for r in self.rows),
                },
            )

    def process_final(self, outputs):
        expected = {(i, j) for i, row in enumerate(self.rows) for j in range(len(row["turns"]))}
        if self.seen != expected:
            raise ValueError("Incomplete turn coverage")
        self.file.close()


async def evaluate(model, tokenizer, rows, settings, directory, on_progress=None):
    from . import datasets, runners
    from .loop import run_loop
    from .utils import postprocess_base

    generation = settings["generation"]
    max_step = settings["speculative_tokens"] + 1
    summaries = []
    for repeat, seed in enumerate(settings["seeds"]):
        folder = directory / f"repeat-{repeat}"
        folder.mkdir()
        model.set_sampling(seed, generation["tokens_to_generate"])
        gpu_count = settings.get("tp_size", 1) * settings.get("pp_size", 1)
        al, timing = metrics_for(rows, folder, gpu_count)

        def progress(row, tokens, repeat=repeat):
            if on_progress:
                on_progress(repeat, "committed", row=row, tokens=tokens)

        if on_progress:
            on_progress(repeat, "generating")
        checkpoint = Checkpoint(rows, folder, max_step, progress)
        runner = runners.SimpleRunner(model, [checkpoint, al, timing])
        success = False
        try:
            outputs = await run_loop(
                runner,
                SimpleNamespace(data=[datasets.base.Request(**r) for r in rows]),
                tokenizer,
                generation["tokens_to_generate"],
                postprocess_base,
                concurrency=settings["concurrency"],
                end_id=tokenizer.eos_token_id,
                show_progress=True,
                chat_template_args=generation["chat_template_kwargs"],
            )
            success = True
        finally:
            checkpoint.file.close()
            if on_progress:
                on_progress(repeat, "finished" if success else "failed")
        durations = [t for t in timing.timing]
        summary = {
            "seed": seed,
            "num_entries": len(rows),
            "num_completed": len(outputs),
            "num_turns": len(checkpoint.seen),
            "num_gen_failed": 0,
            "total_generated_tokens": sum(timing.total_tokens),
            "generation_seconds": max(t[-1] for t in durations) - min(t[0] for t in durations),
            "output_tokens_per_second": timing.out["Output TPS"],
            "spec_acceptance_length": al.out["Average_AL"],
            "ttft_seconds": float(timing.out["TTFT Time"]["mean"]),
            "category_al": al.out["Category_AL"],
        }
        step = timing.out.get("Request Generation Step Time")
        if step:
            summary["decode_step_seconds"] = float(step["mean"])
        output_file = directory / f"output-rs{repeat}.jsonl"
        with output_file.open("x") as f:
            for i, row in enumerate(rows):
                f.write(
                    json.dumps(
                        {
                            **row,
                            "generation": [m["content"] for m in outputs[i] if m["role"] == "assistant"],
                            "num_generated_tokens": checkpoint.tokens[i],
                            "spec_acceptance_length": al.out["Request_AL"][row["question_id"]],
                            "_speed_bench_mode": settings["measurement_mode"],
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        save(folder / "complete.json", summary)
        Path(str(output_file) + ".done").touch()
        summaries.append(summary)
    block = aggregate(summaries, settings)
    key = "pass@1" if len(summaries) == 1 else f"pass@1[avg-of-{len(summaries)}]"
    save(directory / "metrics.json", {"speed-bench": {key: block}})
    return block


def run_engine(settings, rows, directory, engine_args, on_progress=None):
    from vllm import SamplingParams
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.v1.engine.async_llm import AsyncLLM

    from . import datasets, models, runners
    from .loop import run_loop
    from .utils import encode_chat, get_tokenizer, postprocess_base

    if importlib.util.find_spec("kanana_models"):
        import kanana_models  # noqa: F401 - optional model registration
    engine_args = AsyncEngineArgs(**engine_args)
    tokenizer = get_tokenizer(engine_args.tokenizer or settings["model"], settings["trust_remote_code"])
    if settings["chat_template"]:
        tokenizer.chat_template = Path(settings["chat_template"]).read_text()
    if tokenizer.eos_token_id is None:
        raise ValueError("SPEED-Bench requires a tokenizer EOS token")
    prompts = []
    for row in rows:
        messages = [{"role": "system", "content": row["system_prompt"]}] if row["system_prompt"] else []
        messages.append({"role": "user", "content": row["turns"][0]})
        tokens = encode_chat(
            tokenizer,
            messages,
            chat_template_args=settings["generation"]["chat_template_kwargs"],
        )
        prompts.append(
            {
                "question_id": row["question_id"],
                "input_tokens": len(tokens),
                "sha256": digest(tokens),
            }
        )
    save(directory / "initial-prompts.json", prompts)
    capture = StepCapture()

    class Model(models.VLLMModel):
        def __init__(self):
            self.engine_args = engine_args
            self.model = AsyncLLM.from_engine_args(engine_args)
            self.sampling_kwargs = {k: settings["generation"][k] for k in ("temperature", "top_p", "top_k")}

        def set_sampling(self, seed, max_tokens):
            self.sampling_config = SamplingParams(
                detokenize=False,
                seed=seed,
                max_tokens=max_tokens,
                ignore_eos=False,
                **self.sampling_kwargs,
            )

        async def generate(self, prompt_ids, request_id, turn_id):
            key = f"{request_id}.{turn_id}"
            capture.begin(key)
            counts, times, tokens = await super().generate(prompt_ids, request_id, turn_id)
            return capture.finish(key, counts, times, tokens, settings["speculative_tokens"] + 1)

    with capture.installed():
        model = Model()
        try:
            save(directory / "serving-config.json", model.get_serving_config())
            save(
                directory / "runtime.json",
                {
                    "vllm": importlib.metadata.version("vllm"),
                    "torch": importlib.metadata.version("torch"),
                    "engine_args": dataclasses.asdict(engine_args),
                },
            )

            # Identical, excluded warmup for baseline and speculative runs.
            async def run():
                model.set_sampling(0, 256)
                warmup_rows = [rows[i * len(rows) // 32] for i in range(32)]
                folder = directory / "warmup"
                folder.mkdir()
                checkpoint = Checkpoint(warmup_rows, folder, settings["speculative_tokens"] + 1)
                original = model.model.generate

                async def slow_consumer(*args, **kwargs):
                    async for item in original(*args, **kwargs):
                        await asyncio.sleep(0.05)
                        yield item

                model.model.generate = slow_consumer
                try:
                    await run_loop(
                        runners.SimpleRunner(model, [checkpoint]),
                        SimpleNamespace(data=[datasets.base.Request(**r) for r in warmup_rows]),
                        tokenizer,
                        256,
                        postprocess_base,
                        concurrency=32,
                        end_id=tokenizer.eos_token_id,
                        chat_template_args=settings["generation"]["chat_template_kwargs"],
                    )
                finally:
                    model.model.generate = original
                    checkpoint.file.close()
                save(
                    folder / "complete.json",
                    {
                        "turns": len(checkpoint.seen),
                        "coalesced_requests": capture.coalesced_requests,
                    },
                )
                return await evaluate(model, tokenizer, rows, settings, directory, on_progress)

            return asyncio.run(run())
        finally:
            model.model.shutdown()


class StepCapture:
    def __init__(self):
        self.active = {}
        self.coalesced_requests = 0

    def begin(self, key):
        if key in self.active:
            raise ValueError(f"Duplicate request: {key}")
        self.active[key] = {"counts": [], "times": []}

    def observe(self, output):
        key = getattr(output, "request_id", None)
        if key not in self.active or not getattr(output, "outputs", None):
            return
        if len(output.outputs) != 1:
            raise ValueError("SPEED-Bench requires one completion per request")
        record = self.active[key]
        record["counts"].append(len(output.outputs[0].token_ids))
        record["times"].append(time.perf_counter())

    @contextmanager
    def installed(self, collector_class=None):
        if collector_class is None:
            from vllm.v1.engine.output_processor import RequestOutputCollector

            collector_class = RequestOutputCollector
        original = collector_class.put

        def put(collector, output):
            self.observe(output)
            return original(collector, output)

        collector_class.put = put
        try:
            yield self
        finally:
            collector_class.put = original
            self.active.clear()

    def finish(self, key, consumed_counts, consumed_times, tokens, max_step):
        record = self.active.pop(key)
        counts = record["counts"]
        if not counts or counts[-1] != len(tokens):
            raise ValueError(f"Missing native outputs for {key}; cannot measure AL")
        lengths = [b - a for a, b in zip([0] + counts, counts)]
        if any(n < 0 or n > max_step for n in lengths):
            raise ValueError(f"Invalid native decode step for {key}: max={max(lengths)}, limit={max_step}")
        if not consumed_times:
            raise ValueError(f"Missing request start time for {key}")
        self.coalesced_requests += len(consumed_counts) < len(counts)
        return counts, [consumed_times[0], *record["times"]], tokens


def initialize(settings, root):
    import torch

    settings["hardware"] = {
        "gpu_count": settings.get("tp_size", 1) * settings.get("pp_size", 1),
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }

    root = Path(root)
    directory = root / "eval-results/speed-bench"
    if directory.exists() or (root / "logs/config.log").exists():
        raise ValueError("SPEED-Bench requires a fresh output directory; partial timing cannot be resumed")
    directory.mkdir(parents=True)
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    config = {
        "model": settings["model"],
        "model_path": settings["model"],
        "backend": "vllm",
        "tasks": f"speed-bench:{settings['repeats']}",
        "output_dir": str(root),
        "speed_bench": settings,
    }
    save(logs / "config.log", config)
    save(logs / "speed-bench-manifest.json", {"settings_sha256": digest(settings)})
    return config


def run(settings, rows, root, on_progress=None):
    directory = Path(root) / "eval-results/speed-bench"
    return run_engine(settings, rows, directory, deepcopy(settings["engine"]), on_progress)
