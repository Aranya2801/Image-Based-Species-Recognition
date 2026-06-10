# Changelog

All notable changes to this project will be documented in this file.  
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).  
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [2.1.0] — 2025-06-01

### Added
- 🆕 CLIP zero-shot species recognition (`src/models/clip_model.py`)
- 🆕 Monte Carlo Dropout uncertainty quantification (20-pass default)
- 🆕 Geographic prior embedding module (`GeoPriorEmbedding`)
- 🆕 Cross-attention multi-backbone fusion (`CrossAttentionFusion`)
- 🆕 Hierarchical classification head (species + genus + family)
- 🆕 Out-of-distribution (OOD) detection via entropy thresholding
- 🆕 ONNX export script with ONNX Runtime benchmark
- 🆕 Beautiful responsive web UI with Grad-CAM overlay
- 🆕 Docker multi-stage build with GPU support
- 🆕 MLflow + W&B experiment tracking
- 🆕 Comprehensive pytest test suite (94% coverage)
- 🆕 GitHub Actions CI/CD pipeline
- 🆕 Dataset downloader script (iNat, PlantNet, NABirds, IP102)

### Changed
- ♻️ Refactored backbone factory to support 8 timm architectures
- ♻️ Upgraded training loop: EMA weights + cosine warm restarts
- ♻️ API migrated to FastAPI 0.110 with async lifespan
- ♻️ Augmentation upgraded to Albumentations v1.3 pipeline

### Fixed
- 🐛 Memory leak in DataLoader when `pin_memory=True` with CPU
- 🐛 Incorrect label smoothing applied to hierarchical heads
- 🐛 Unicode species names now handled correctly in taxonomy lookup

---

## [2.0.0] — 2025-03-15

### Added
- 🆕 Full ensemble model (EfficientNetV2-L + ViT-Large + ConvNeXt-XL)
- 🆕 Geographic prior integration using GPS metadata
- 🆕 IUCN Red List conservation status lookup
- 🆕 Wikipedia API integration for species summaries
- 🆕 FastAPI REST API with OpenAPI documentation
- 🆕 Class-balanced weighted sampler for long-tail iNat distribution

### Changed
- ♻️ Replaced ResNet baseline with EfficientNetV2
- ♻️ Dataset class now supports iNaturalist COCO-JSON format

---

## [1.5.0] — 2024-12-01

### Added
- 🆕 Vision Transformer (ViT-Large/16) backbone support
- 🆕 Grad-CAM++ visualization
- 🆕 Mixed precision (FP16) training

### Changed
- ♻️ Upgraded to PyTorch 2.2

---

## [1.0.0] — 2024-09-01

### Added
- 🎉 Initial release
- EfficientNetV2-L baseline model
- iNaturalist 2021 dataset support
- Basic CLI inference script
- Simple Flask web interface

---

*Contributors: [@Aranya2801](https://github.com/Aranya2801)*
