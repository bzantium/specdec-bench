import json
from pathlib import Path
from types import SimpleNamespace

from specdec_bench import UPSTREAM_COMMIT, __version__, utils


def test_provenance_uses_standalone_checkout(tmp_path, monkeypatch):
    directories = []

    def git_sha(path):
        directories.append(path)
        return "standalone-commit"

    monkeypatch.delenv("SPECDEC_BENCH_SHA", raising=False)
    monkeypatch.setenv("MODELOPT_SHA", "unrelated-checkout")
    monkeypatch.setattr(utils, "_git_sha", git_sha)
    monkeypatch.setattr(utils, "_get_engine_version", lambda engine: "test")
    monkeypatch.setattr(utils, "_get_gpu_name", lambda: None)
    utils.dump_env(SimpleNamespace(engine="VLLM", model_dir=None), tmp_path)
    result = json.loads((tmp_path / "configuration.json").read_text())
    assert result["specdec_bench_sha"] == "standalone-commit"
    assert result["modelopt_sha"] == UPSTREAM_COMMIT
    assert result["specdec_bench_version"] == __version__
    assert directories == [Path(utils.__file__).resolve().parent]
