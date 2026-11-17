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


git remote add origin https://github.com/YOUR-USERNAME/MicroTwin-Vision.git
git push -u origin main
```
Replace `YOUR-USERNAME` with your account name.
