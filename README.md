# SpecDec Bench

Standalone evaluation code extracted from NVIDIA Model-Optimizer at
[`23355eda`](https://github.com/NVIDIA/Model-Optimizer/tree/23355eda90a25c290f9b1fdfb928ad54caae7d10/examples/specdec_bench).
Includes dataset preparation, conversation runners, engine adapters, acceptance
length and timing metrics. The upstream measurement code is preserved.
`specdec_bench.evaluation` adds native vLLM measurement with step capture, serial
repeats and checkpoints; a host harness supplies progress/reporting callbacks.
See [NOTICE](NOTICE) for provenance and local changes.

Use an existing vLLM, SGLang or TensorRT-LLM environment:

```bash
pip install -e '.[evaluation]'
python prepare_data.py --dataset speed --config qualitative
python run.py --help
```

The inference engine is installed separately. Prepared data is written under
`data/`; model weights, datasets and evaluation outputs are not included in this
repository. `run.py` and `prepare_data.py` are checkout entry points; the Python
package can also be imported by an evaluation harness.

Upstream usage examples:
[SpecDec Bench README](https://github.com/NVIDIA/Model-Optimizer/blob/23355eda90a25c290f9b1fdfb928ad54caae7d10/examples/specdec_bench/README.md).

Code is licensed under [Apache-2.0](LICENSE). SPEED-Bench data is governed by the
NVIDIA Evaluation Dataset License Agreement and its constituent datasets' terms;
see the [dataset repository](https://huggingface.co/datasets/nvidia/SPEED-Bench).
