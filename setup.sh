#!/bin/bash

# ─────────────────────────────────────────────
#  ML Framework - Project Tree Generator
#  Usage: bash setup_ml_framework.sh
#  (run from inside your project folder)
# ─────────────────────────────────────────────

echo "🚀 Creating ML framework in $(pwd)"

# ── Directories ───────────────────────────────
mkdir -p data/{raw,processed,models}
mkdir -p config
mkdir -p tests
mkdir -p src/{base,models,preprocessing,training,persistence,visualization}
mkdir -p utils

# ── src/base ──────────────────────────────────
touch src/base/base_model.py
touch src/base/base_preprocessor.py

# ── src/models ────────────────────────────────
touch src/models/__init__.py

# ── src/preprocessing ─────────────────────────
touch src/preprocessing/__init__.py
touch src/preprocessing/data_loader.py
touch src/preprocessing/normalizer.py
touch src/preprocessing/data_splitter.py

# ── src/training ──────────────────────────────
touch src/training/__init__.py
touch src/training/trainer.py
touch src/training/metrics.py
touch src/training/loss_functions.py

# ── src/persistence ───────────────────────────
touch src/persistence/__init__.py
touch src/persistence/model_io.py

# ── src/visualization ─────────────────────────
touch src/visualization/__init__.py
touch src/visualization/plotter.py

# ── utils ─────────────────────────────────────
touch utils/__init__.py
touch utils/math_utils.py

# ── config ────────────────────────────────────
touch config/config.yaml

# ── tests ─────────────────────────────────────
touch tests/test_models.py
touch tests/test_preprocessing.py
touch tests/test_metrics.py

# ── entry points ──────────────────────────────
touch train.py
touch predict.py
touch evaluate.py
touch README.md

echo ""
echo "✅ Done!"