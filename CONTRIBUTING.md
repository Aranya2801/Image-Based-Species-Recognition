# 🤝 Contributing to Image-Based Species Recognition

Thank you for your interest in contributing to this project!  
We welcome contributions from ecologists, ML researchers, developers, and citizen scientists alike.

---

## 📋 Table of Contents
- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Issue Templates](#issue-templates)

---

## 🌿 Code of Conduct

Be kind, respectful, and constructive. We aim to maintain an inclusive environment for all contributors regardless of background or expertise level.

---

## 🔧 How to Contribute

### 🐛 Report a Bug
1. Check existing [issues](https://github.com/Aranya2801/Image-Based-Species-Recognition/issues)
2. Open a new issue using the **Bug Report** template
3. Include: Python version, OS, CUDA version, error message, minimal reproduction

### 💡 Suggest a Feature
- Open an issue using the **Feature Request** template
- Describe the use case clearly — especially for daily-use improvements

### 🧬 Add a New Species Dataset
- Open a PR with a dataset integration script in `scripts/`
- Include dataset source, species count, and license info in `docs/DATASETS.md`

### 🤖 Improve Model Accuracy
- Train a new backbone / loss variant
- Open a PR with benchmark results on iNat-2021 Mini validation set

---

## ⚙️ Development Setup

```bash
# 1. Fork and clone
git clone https://github.com/YOUR_USERNAME/Image-Based-Species-Recognition.git
cd Image-Based-Species-Recognition

# 2. Create a feature branch
git checkout -b feat/your-feature-name

# 3. Set up environment
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

# 4. Verify setup
python scripts/verify_install.py

# 5. Make your changes ...

# 6. Run tests before committing
pytest tests/ -v

# 7. Push and open a PR
git push origin feat/your-feature-name
```

---

## 🎨 Code Style

We use:
- **Black** for formatting (line length: 100)
- **Ruff** for linting
- **isort** for import ordering
- **Type hints** everywhere (use `from __future__ import annotations`)

```bash
# Auto-format before committing
black src/ tests/
isort src/ tests/
ruff check src/ tests/ --fix
```

### Docstrings
Use Google-style docstrings:
```python
def predict(self, image: str, latitude: float = None) -> dict:
    """
    Predict species from an image.

    Args:
        image: Path to image file or PIL Image object.
        latitude: Optional GPS latitude for geo-prior boost.

    Returns:
        Dict with species, confidence, top_5, taxonomy, etc.

    Raises:
        ValueError: If image path does not exist.
    """
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run a specific test file
pytest tests/test_models.py -v

# Run a specific test
pytest tests/test_api.py::TestPredictEndpoint::test_predict_jpeg -v
```

### Test Requirements
- All new code must have tests in `tests/`
- Coverage should not drop below **90%**
- Tests must not require GPU (use `pretrained=False` and small models)

---

## 📥 Pull Request Process

1. **Title format:** `feat: add X` | `fix: correct Y` | `docs: update Z` | `test: add tests for W`
2. **Description:** What changed and why; link to the relevant issue
3. **Checklist:**
   - [ ] Tests pass (`pytest tests/`)
   - [ ] Code is formatted (`black`, `ruff`)
   - [ ] Docstrings added for new functions
   - [ ] `CHANGELOG.md` updated
   - [ ] No model weights or datasets committed

---

## 📝 Issue Templates

Use the provided templates in `.github/ISSUE_TEMPLATE/`:
- `bug_report.md` — For bugs and errors
- `feature_request.md` — For new features or improvements
- `dataset_request.md` — For requesting new dataset support

---

## 🏆 Recognition

All contributors are listed in `CHANGELOG.md` and will be acknowledged in release notes.  
Significant contributions may be highlighted in the README.

---

*Questions? Open a GitHub Discussion or issue.*  
*Thank you for making biodiversity AI better! 🌿*
