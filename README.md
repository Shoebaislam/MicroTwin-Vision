# MicroTwin-Vision
**Status: scaffold prepared; dataset not downloaded; no experiments executed.**

Independent exploratory project for Fall 2027 doctoral applications. A lightweight comparison of interpretable image geometry and PCA-based pixel representation for predicting PDE-derived effective transport tensor entries from **synthetic** battery porous-electrode images. **Not real microscopy data.**

## Source and attribution
Weber, R.; Korneev, S.; Battiato, I. (2022). *Labeled Image Dataset of Generated Porous Electrode Microstructures and Calculated Transport Parameters for Neural Network Training.* Mendeley Data, v1. DOI: [10.17632/mgmxv5tjt2.1](https://doi.org/10.17632/mgmxv5tjt2.1). CC BY 4.0. Source includes 100,000 synthetic binary 64×64 TIFF images and PDE-closure transport labels generated with OpenFOAM.

## Quick start (Windows PowerShell)
```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
Download `images.zip` and `diffusion.zip` from https://data.mendeley.com/datasets/mgmxv5tjt2/1 into `data/` (dataset archives are **not** redistributed in this project).
```powershell
python inspect_dataset.py
# Check the label layout, tensor component meaning, pore phase convention, and ID matching first.
python run_experiment.py --sample-size 1000 --target-index 0 --pore-is-white
# OR --pore-is-black, after inspecting the dataset's phase coding.
```

## Model plan
- **Geometry**: area fraction of chosen pore phase; largest connected fraction; binary spanning indicator; four-neighbor interface density. Phase convention must be validated first.
- **Pixels**: standardized grayscale pixels reduced via PCA (train split only); Ridge regression.
- **Evaluation**: reproducible 80/20 random image-level split, MAE and R², additive pixel noise at sigma 0.05 and 0.10, same held-out labels.
- **Limitations**: random image split is not a cross-generator generalization study; no uncertainty calibration; no tuning or confidence intervals; largest connected fraction and spanning connectivity are simplistic 2D proxies; label file mapping and physics semantics must be verified. Noise is image-sensor-style perturbation, not physically realistic microstructure damage.

**Caution:** `--target-index 0` selects the first numeric element in a label file. Validate that this is the tensor component you intend to predict; if the stored format has headers, edit the parser after inspection. The geometric model assumes one verified binary pore-phase convention. Do not publish results until inspected.

## Results and research integrity
No measured results are available. Once executed, `results/metrics.csv` records MAE/R² from the real run. Never invent values or describe this dataset as experimental microscopy.

## Planned next work
Verify label orientation and physics, include directional connectivity, blocked train/test split by morphology regimes, confidence intervals, compare scalar/tensor targets, document engineering trade-offs and runtime.

## GitHub-ready walkthrough
This repository includes `.github/workflows/ci.yml`, `tests/`, a reproducible toy smoke-test generator, charts generated on execution, and a scientific interpretation guide in `DISCUSSION.md`.

```powershell
python -m pytest -q
python make_demo_data.py
python run_experiment.py --data-dir demo_data --sample-size 100 --pore-is-white
```
**All demo targets are arbitrary algebraic numbers, NOT OpenFOAM/PDE outputs, and none of the demo images is actual Mendeley data.** Results in `results/` are ignored by Git. Run against verified external Mendeley archives later to create scientific results, and review phase meaning, tensor indexing and pairing before interpreting them.

After a real dataset run, copy *validated* figures and tables to `docs/figures/` to commit them. See `DISCUSSION.md` for descriptions and failure modes. The experiment emits:
- `results/example_microstructures.png` — sample image inputs
- `results/mae_noise.png` — clean/noisy MAE comparison
- `results/r2_noise.png` — clean/noisy R² comparison
- `results/runtime.json` — measured wall-clock train+predict runtime per model
- `results/metrics.csv` — raw per-model evaluations

### Publish on GitHub
Create a blank GitHub repository named `MicroTwin-Vision` without creating a remote README, then run:
```powershell
git init
git add .
git commit -m "Add physics-aware microstructure regression benchmarks and CI"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/MicroTwin-Vision.git
git push -u origin main
```
Replace `YOUR-USERNAME` with your account name.
