# Testing

Maintainers adapted this guide from the omni testing documentation in
[`vllm-project/vllm-torchtpu`](https://github.com/vllm-project/vllm-torchtpu).

## Omni Multimodal and Diffusion Tests (`tests_omni/`)

Install `vllm-omni-torchtpu` with the `core-git` and `test` dependency groups
(see [installation.md](installation.md)), export `VLLM_PLUGINS`, and run
`pytest` from the repository root:

```bash
export VLLM_PLUGINS=torchtpu_omni,torchtpu_layers,vllm_omni_register_models

# PR lane on a single-host TPU
pytest tests_omni -sv --timeout=1200 -m 'not nightly'

# Nightly single-chip lane on TPU
pytest tests_omni -sv --timeout=1800 -m 'not multichip'

# Nightly multichip lane on TPU
pytest tests_omni -sv --timeout=1800 -m 'multichip'

# CPU checks (no TPU required)
pytest tests_omni/test_packaging.py
pytest tests_omni --collect-only -q
```

`tests_omni/conftest.py` also sets the `VLLM_PLUGINS` default automatically
before `pytest` imports `vllm`. Apart from `tests_omni/test_packaging.py`, the
tests in [`tests_omni/`](../tests_omni/) call `torch.accelerator.synchronize()`
during fixture setup and require TPU hardware to execute.

The FLUX and Qwen-Image tests compare outputs against golden files under
`$HF_HOME/golden`. Upstream CI stages those files from
`gs://tpu-commons-ci/golden/`.

## Pytest Markers Reference

`pyproject.toml` registers these markers under `[tool.pytest.ini_options]`:

| Marker | CI lane | Description |
| :--- | :--- | :--- |
| `@pytest.mark.nightly` | Nightly CI workflow (`NIGHTLY=1`) | Long-running or full-checkpoint tests excluded from per-PR gates. |
| `@pytest.mark.multichip` | Multichip TPU slice (`2x2x1`) | Tests that require more than two TPU chips. |
| `@pytest.mark.spawns_tpu_workers` | Single-host or multichip TPU | Tests that spawn distributed TPU workers and run before in-process tests. |
| `@pytest.mark.omni` | Omni TPU lanes | Tests that exercise `vllm-omni` multimodal and diffusion models. |
| `@pytest.mark.tpu_test` | Single-host TPU | Registered for parity with `vllm-torchtpu` marker definitions. |
