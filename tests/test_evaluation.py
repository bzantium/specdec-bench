import asyncio
import json
from types import SimpleNamespace

import pytest

from specdec_bench.config import load_dataset
from specdec_bench.config import resolve as resolve_settings
from specdec_bench.evaluation import evaluate
from specdec_bench.reporting import aggregate


def context(**changes):
    params = {
        "model": "target",
        "tasks": "speed-bench:4",
        "backend": "vllm",
        "tp_size": None,
        "dp_size": None,
        "pp_size": 1,
        "config": None,
        "trust_remote_code": False,
        "chat_template": None,
    }
    params.update(changes)
    return params, set(changes)


def resolve(ctx):
    return resolve_settings(*ctx)


def test_measurement_defaults_and_rejection():
    settings = resolve(context())
    assert settings["generation"]["top_p"] == 1
    assert settings["generation"]["tokens_to_generate"] == 4096
    assert settings["seeds"] == [0, 1, 2, 3]
    for options in [
        {"tasks": "speed-bench,foo"},
        {"top_p": 0.95},
        {"tp_size": 2},
        {"temperature": 0.6},
        {"server_address": "http://host"},
        {"extra_client_args": "++max_concurrent_requests=64"},
    ]:
        with pytest.raises(ValueError):
            resolve(context(**options))


@pytest.mark.parametrize(
    "args",
    [
        "--enable-prefix-caching",
        "--stream-interval 2",
        "--max-num-seqs 64",
        "--seed 1",
        "--unknown-option 1",
        "--speculative-config '[]'",
        '--speculative-config \'{"num_speculative_tokens":3,"enable_adaptive_verification":true}\'',
    ],
)
def test_engine_invariants_reject_incompatible_settings(args):
    from specdec_bench.config import engine_options

    with pytest.raises(ValueError):
        engine_options(resolve(context(extra_server_args=args)))


def test_yaml_and_cli_engine_precedence(tmp_path):
    from specdec_bench.config import engine_options

    path = tmp_path / "config.yaml"
    path.write_text(
        "inference:\n  temperature: 0\nchat_template_kwargs:\n  thinking_mode: think\n"
        "gpu_memory_utilization: 0.8\nextra_server_args: --max-model-len 8192 --gpu-memory-utilization 0.7\n"
    )
    settings = resolve(context(config=str(path), temperature=1, max_model_len=131072))
    assert settings["generation"]["temperature"] == 1
    assert settings["generation"]["chat_template_kwargs"] == {"thinking_mode": "think"}
    assert engine_options(settings)["max_model_len"] == 131072
    assert engine_options(settings)["gpu_memory_utilization"] == 0.8


def test_empty_speculative_config_is_no_sd():
    from specdec_bench.config import engine_options

    options = engine_options(resolve(context(extra_server_args="--speculative-config '{}'")))
    assert "speculative_config" not in options


def test_dataset_conversion_preserves_order_and_system(tmp_path):
    path = tmp_path / "data.jsonl"
    rows = [
        {
            "question_id": "x",
            "category": "math",
            "messages": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "first"},
                {"role": "user", "content": "next"},
            ],
        }
    ]
    path.write_text(json.dumps(rows[0]) + "\n")
    result = load_dataset(path, full=False)
    assert result == [
        {
            "question_id": "x",
            "category": "math",
            "turns": ["first", "next"],
            "system_prompt": "s",
        }
    ]
    with pytest.raises(ValueError, match="880"):
        load_dataset(path)
    path.write_text(path.read_text() * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        load_dataset(path, full=False)


class Tokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, **kwargs):
        return messages[-1]["content"]

    def encode(self, text, **kwargs):
        return [int(text)]

    def decode(self, tokens):
        return "response"


class Model:
    def set_sampling(self, seed, max_tokens):
        self.seed = seed

    async def run(self, prompt_ids, max_length, end_id, request_id, turn_id):
        # One conversation: [4, 1], then [1]. Another: [1, 1].
        lengths = {1: [4, 1], 2: [1], 3: [1, 1]}[prompt_ids[0]]
        base = request_id * 10 + turn_id * 3
        return {
            "output_ids": [[[1] * n for n in lengths]],
            "token_times": [base + i for i in range(len(lengths) + 1)],
        }


def test_official_loop_macro_al_and_repeat_reset(tmp_path):
    rows = [
        {
            "question_id": "a",
            "category": "math",
            "system_prompt": None,
            "turns": ["1", "2"],
        },
        {
            "question_id": "b",
            "category": "coding",
            "system_prompt": None,
            "turns": ["3"],
        },
    ]
    settings = resolve(context(tasks="speed-bench:2"))
    settings.update(speculative_tokens=3, measurement_mode="speculative")
    block = asyncio.run(evaluate(Model(), Tokenizer(), rows, settings, tmp_path))
    # Conversation A is (4+1+1)/3=2; B is 1. Macro AL=1.5, not pooled 8/5.
    assert block["spec_acceptance_length"] == 1.5
    assert block["category_al"] == {"math": 2, "coding": 1}
    assert block["output_tokens_per_second"] == pytest.approx(8 / 12)
    assert block["total_generated_tokens"] == 16
    assert (tmp_path / "output-rs1.jsonl.done").exists()
    assert len((tmp_path / "repeat-1/raw-turns.jsonl").read_text().splitlines()) == 3
    assert [r["total_generated_tokens"] for r in block["repeats"]] == [8, 8]


def test_repeat_tps_is_arithmetic_mean_not_pooled_ratio():
    rows = [
        {
            "seed": i,
            "num_entries": 880,
            "total_generated_tokens": tokens,
            "generation_seconds": seconds,
            "output_tokens_per_second": tokens / seconds,
            "spec_acceptance_length": 2,
            "ttft_seconds": 0.1,
            "category_al": {"math": 2},
        }
        for i, (tokens, seconds) in enumerate([(100, 10), (400, 20)])
    ]
    settings = {
        "repeats": 2,
        "seeds": [0, 1],
        "measurement_mode": "speculative",
    }
    result = aggregate(rows, settings)
    assert result["output_tokens_per_second"] == 15
    assert result["generation_seconds"] == 30
    with pytest.raises(ValueError):
        aggregate(rows[:1], settings)


from specdec_bench.evaluation import StepCapture


class Collector:
    def __init__(self):
        self.latest = None

    def put(self, output):
        if isinstance(output, Exception):
            raise output
        self.latest = output


def output(count):
    return SimpleNamespace(request_id="req", outputs=[SimpleNamespace(token_ids=list(range(count)))])


@pytest.mark.parametrize("step", [1, 4, 8])
def test_native_steps_survive_coalescing_and_hook_restores(step):
    capture = StepCapture()
    original = Collector.put
    with capture.installed(Collector):
        capture.begin("req")
        collector = Collector()
        for count in [step, step * 2, step * 3]:
            collector.put(output(count))
        counts, times, tokens = capture.finish("req", [3 * step], [0, 1], list(range(3 * step)), step)
        assert counts == [step, step * 2, step * 3]
        assert len(times) == 4
        assert len(tokens) == 3 * step
        with pytest.raises(RuntimeError, match="engine failure"):
            collector.put(RuntimeError("engine failure"))
    assert Collector.put is original


def test_invalid_native_steps_fail_and_exception_restores():
    capture = StepCapture()
    original = Collector.put
    with pytest.raises(ValueError, match="native decode"), capture.installed(Collector):
        capture.begin("req")
        Collector().put(output(9))
        capture.finish("req", [9], [0, 1], list(range(9)), 8)
    assert Collector.put is original


def test_engine_cannot_mutate_saved_settings(tmp_path, monkeypatch):
    from specdec_bench import evaluation

    settings = {"engine": {"speculative_config": {"num_speculative_tokens": 3}}}

    def engine(settings, rows, directory, options, progress):
        options["speculative_config"]["target_model_config"] = object()

    monkeypatch.setattr(evaluation, "run_engine", engine)
    evaluation.run(settings, [], tmp_path)
    assert settings == {"engine": {"speculative_config": {"num_speculative_tokens": 3}}}
