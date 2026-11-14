# Environment Setup

## Python environments

The project has two main dependency profiles.

### Dataset preparation, generation, and judge pipeline

```bash
python -m venv .venv
# Activate the environment for your shell, then:
python -m pip install --upgrade pip
python -m pip install -r requirements-curation.txt
```

This profile includes the packages imported by the original chunking/generation/audit/judge workflow. The exact historical versions for every API/curation dependency were not captured, so this file is not claimed to recreate the original generation environment byte-for-byte.

### QLoRA training

Install a PyTorch build compatible with the target CUDA runtime and GPU first. The historical Kaggle runtime recorded PyTorch 2.11.0+cu130. Then install:

```bash
python -m pip install -r requirements-training.txt
```

The historical training environment recorded Transformers 5.20.2, PEFT 0.21.2, Accelerate 1.15.0, Datasets 4.8.5, and bitsandbytes 0.50.2.

## API credentials

Copy `.env.example` to `.env` and fill the values locally:

- `OPENROUTER_API_KEY`: used by generation scripts that call OpenRouter.
- `GROQ_API_KEY`: used by the cross-model judge script.

The root `.gitignore` excludes `.env`. Do not paste API keys into scripts, notebooks, commits, logs, or issues. The example file contains empty placeholders only.

## Data directory requirements

The scripts preserve historical project paths and expect source corpora under `data/frozen/` and intermediate outputs under `data/interim/`. The private/source IoT document corpus is not included in this public repository. Place a separately licensed copy in the expected location before running any generation, evidence-repair, or reconciliation pipeline.

## Reproduction caution

The source scripts are included, but a clean end-to-end rerun has not yet been demonstrated from this repository. The frozen corpus, prompt snapshots, historical API responses, and complete inference/evaluation code are not all present here. Treat new runs as new experiments and do not overwrite the frozen results.
