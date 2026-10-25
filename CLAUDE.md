# alphafold3-pytorch

Unofficial PyTorch reimplementation of DeepMind's AlphaFold 3 (protein/RNA/DNA/ligand complex structure prediction via a Pairformer trunk + atom-level diffusion). Paper module docstrings like `""" Algorithm 17 """` refer to `docs/alphafold3-supplementary.pdf`.

## Architecture

- `alphafold3_pytorch/alphafold3.py` (~8k lines): every model module plus the top-level `Alphafold3`. The ein-notation legend at the top (`b n n dp`, `m` = atoms, `s` = MSA, `t` = templates…) is what shape annotations use.
- `alphafold3_pytorch/inputs.py` (~4.8k lines): input dataclasses and conversions. Pipeline: `Alphafold3Input` / `PDBInput` → `MoleculeInput` → `AtomInput` → `BatchedAtomInput` → `alphafold3(**batched.model_forward_dict())`. Conversions are registered in `INPUT_TO_ATOM_TRANSFORM` (or `register_input_transform`).
- `attention.py` (windowed/local attention), `trainer.py` (Lightning Fabric `Trainer`), `configs.py` (pydantic configs built from YAML), `life.py` (residue/ligand RDKit templates), `plm.py`/`nlm.py` (optional language-model embeddings), `cli.py` (click inference CLI), `app.py` (gradio).
- `alphafold3_pytorch/data/` = parsing **code** (mmCIF, MSA, templates, weighted sampler). `alphafold3_pytorch/common/` = residue constants and `Biomolecule`. `alphafold3_pytorch/nim/` = optional Nim fast paths for MSA featurization.
- Top-level `data/` = committed **test fixtures** (not code). `scripts/` = one-off PDB filtering/clustering pipelines.

## Commands (always from the repo root: data paths are CWD-relative)

```bash
cp .env.sample .env                      # sets TYPECHECK=True; without it @typecheck is a no-op
uv venv --python 3.10 .venv && source .venv/bin/activate
uv pip install -e '.[test]'
uvx ruff@0.4.10 check alphafold3_pytorch/          # lint the way CI did; baseline is 1 known E731 at alphafold3.py:2743
pytest tests/test_af3.py -k pairformer -q          # target one area; full suite is ~9 min on CPU
IS_GITHUB_CI=True pytest tests/ -q                 # full suite, skipping the slow sampling paths (old CI setting)
pytest --num-shards 4 --shard-id 0 tests/          # pytest-shard is installed
```

Expected on a fresh install: 2 pre-existing failures. `test_af3.py::test_alphafold3_with_nlm_embeddings` fails because `multimolecule==0.0.5` (yanked) doesn't work with current transformers. `test_input.py::test_pdbinput_input` fails in its sampling half (the `PDBInput(filepath)` eval input fails conversion; hidden under `IS_GITHUB_CI`). The `test_nim_msa.py` tests and most of `test_data_parsing.py` skip without Nim or the full PDB download. That is normal.

## Conventions

- **Match the file's style. The repo has two:** model code (`alphafold3.py`, `attention.py`, `trainer.py`, `configs.py`, `cli.py`) uses `fn(dim = 64)` with spaces around kwarg `=`, single quotes, terse `# section` comments, and `exists(v)`/`default(v, d)` helpers instead of `is not None`. Data code (`data/`, `common/`, `utils/`, `life.py`, most of `inputs.py`) uses `fn(dim=64)`, double quotes, and `:param x:` docstrings.
- Tensor args are typed with shape strings: `x: Float['b n d']`, `mask: Bool['b n'] | None = None`, from `alphafold3_pytorch.tensor_typing` (not raw jaxtyping), and functions are decorated with `@typecheck`. Module `__init__`s are keyword-only (`*,`).
- New public modules get added to the imports and `__all__` in `alphafold3_pytorch/__init__.py`. New `Alphafold3.__init__` kwargs are saved automatically by `@save_args_and_kwargs` for `init_and_load`.
- Tests: plain pytest functions in `tests/test_af3.py` (models) or `tests/test_input.py` (inputs). Build a tiny model with random tensors, run forward, call `loss.backward()`, and assert shapes. Use `@pytest.mark.parametrize` for flags. Files start with `os.environ['TYPECHECK'] = 'True'` **before** importing the package, because type checking is decided at import time.

## Guardrails

- Don't run `ruff check .`, the bare `ruff check alphafold3_pytorch/`, or `ruff format`. The unpinned ruff (0.17+) enables far more rules: ~400 errors on untouched upstream, and `format` rewrites 29 files. Use the pinned `uvx` command above and only fix what you introduced.
- Don't edit, move, or regenerate anything under top-level `data/`. Tests hard-code PDB IDs (`721p`, `209d`, `7a4d`) and paths in it. `data/ccd_data/components_smiles.json` is generated from `components.cif`.
- `maybe_transform_to_atom_input` catches conversion errors and logs them instead of raising them. An `IndexError` in `collate_inputs_to_batched_atom_input` almost always means every input failed to convert, so read the `ERROR | ... Failed to convert input` log line above it.
- Tests write `./test-folder/`, `test-atom-input.pt`, and `test_reconstructed.cif` into the repo root. Don't commit them.
- Don't bump `version` in `pyproject.toml`. Releases are separate commits, and checkpoints store the package version.
- No CI exists (workflows were removed). Run the tests you touched yourself.
