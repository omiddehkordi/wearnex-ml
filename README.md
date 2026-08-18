# WearNex-ML

Computer vision pipeline that classifies clothing images and generates
style recommendations, built to eventually sit behind the WearNex web
app, with an AI persona named **JeanClaude** surfaced to users.

## Pipeline overview

```
image (file / upload / URL)
  -> wearnex.data.io          load + validate bytes into a PIL image
  -> wearnex.data.preprocess  crop-to-subject, letterbox resize, tensor
  -> wearnex.models.classifier    category + color prediction
  -> wearnex.models.embeddings    style embedding vector
  -> wearnex.recommendation.engine  nearest-neighbor catalog search
  -> wearnex.api                  FastAPI service for the web app
```

## Project layout

```
src/wearnex/
  config.py            paths, categories, hyperparameters
  data/
    io.py               load images from disk / bytes / URL (implemented)
    preprocess.py        crop, resize/pad, tensor pipeline (implemented)
    dataset.py            PyTorch Dataset over a labeled image catalog
  models/
    backbone.py          pretrained ResNet50 trunk, used by the classifier
    classifier.py          category + color classification head
    embeddings.py          Marqo-FashionCLIP wrapper for style embeddings
  recommendation/
    engine.py             nearest-neighbor index over catalog embeddings
    compatibility.py        category-pairing rules ("what goes with this")
  training/
    train_classifier.py    supervised training loop for the classifier
    train_embeddings.py     optional fine-tuning loop for the embedding model
  inference/
    predict.py             single entry point: image -> classification + recs
  api/
    main.py                FastAPI app (/analyze, /recommend/*)
    schemas.py              request/response models
scripts/
  build_index.py           embeds a catalog directory into the recommendation index
data/
  raw/                     original, unprocessed images (gitignored)
  processed/                 cleaned/organized training catalogs (gitignored)
  embeddings/                 cached embedding exports (gitignored)
models_store/                trained model checkpoints (gitignored)
tests/                       unit tests for the implemented pieces
```

## Setup

```bash
uv venv --python 3.13 .venv
uv pip install -e ".[dev]" --python .venv/bin/python
uv lock  # regenerate uv.lock after changing dependencies
```

## What's implemented vs. scaffolded

- **Implemented and tested**: image loading (`data/io.py`), preprocessing
  (`data/preprocess.py`), the labeled dataset wrapper (`data/dataset.py`).
- **Embeddings — pretrained, usable out of the box**: `models/embeddings.py`
  wraps Marqo-FashionCLIP, so style embeddings and similarity search work
  without training anything. `training/train_embeddings.py` is only for
  optionally fine-tuning that pretrained space onto your own catalog/photos
  later (see the module docstring).
- **Structured but untrained**: the classifier architecture, its training
  loop, and the recommendation engine are complete, runnable code — but the
  classifier needs real labeled data
  (e.g. [DeepFashion](https://github.com/switchablenorms/DeepFashion2))
  before category/color predictions are meaningful. Point `--data-dir` at a
  folder laid out as `<category>/<image>.jpg` to train.

## Typical workflow

```bash
# 1. Train the classifier
python -m wearnex.training.train_classifier --data-dir data/processed/catalog

# 2. (optional) fine-tune the pretrained embedding model on your own catalog
python -m wearnex.training.train_embeddings --data-dir data/processed/catalog

# 3. Build the recommendation index over your catalog
python scripts/build_index.py --data-dir data/processed/catalog

# 4. Serve it
uvicorn wearnex.api.main:app --reload
```

## Tests

```bash
pytest
```
