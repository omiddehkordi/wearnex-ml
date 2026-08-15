# StyleGPT

Computer vision pipeline that classifies clothing images and generates
style recommendations, built to eventually sit behind a user-facing web
app.

## Pipeline overview

```
image (file / upload / URL)
  -> stylegpt.data.io          load + validate bytes into a PIL image
  -> stylegpt.data.preprocess  crop-to-subject, letterbox resize, tensor
  -> stylegpt.models.classifier    category + color prediction
  -> stylegpt.models.embeddings    style embedding vector
  -> stylegpt.recommendation.engine  nearest-neighbor catalog search
  -> stylegpt.api                  FastAPI service for the web app
```

## Project layout

```
src/stylegpt/
  config.py            paths, categories, hyperparameters
  data/
    io.py               load images from disk / bytes / URL (implemented)
    preprocess.py        crop, resize/pad, tensor pipeline (implemented)
    dataset.py            PyTorch Dataset over a labeled image catalog
  models/
    backbone.py          shared pretrained CNN trunk
    classifier.py          category + color classification head
    embeddings.py          style-embedding head for similarity search
  recommendation/
    engine.py             nearest-neighbor index over catalog embeddings
    compatibility.py        category-pairing rules ("what goes with this")
  training/
    train_classifier.py    supervised training loop for the classifier
    train_embeddings.py     triplet-loss training loop for embeddings
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
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## What's implemented vs. scaffolded

- **Implemented and tested**: image loading (`data/io.py`), preprocessing
  (`data/preprocess.py`), the labeled dataset wrapper (`data/dataset.py`).
- **Structured but untrained**: the classifier and embedding model
  architectures, training loops, and recommendation engine are complete,
  runnable code — but the models need real labeled data
  (e.g. [DeepFashion](https://github.com/switchablenorms/DeepFashion2))
  before predictions are meaningful. Point `--data-dir` at a folder laid
  out as `<category>/<image>.jpg` to train.

## Typical workflow

```bash
# 1. Train the classifier
python -m stylegpt.training.train_classifier --data-dir data/processed/catalog

# 2. Train the style-embedding model
python -m stylegpt.training.train_embeddings --data-dir data/processed/catalog

# 3. Build the recommendation index over your catalog
python scripts/build_index.py --data-dir data/processed/catalog

# 4. Serve it
uvicorn stylegpt.api.main:app --reload
```

## Tests

```bash
pytest
```
