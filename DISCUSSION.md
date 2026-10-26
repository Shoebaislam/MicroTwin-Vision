# Figures, engineering discussion and reporting checklist

**Research status:** Software scaffolding tested on toy images with nonphysical labels; Mendeley dataset experiment NOT yet executed. Do not put toy-data scores in a CV, application, or project claims.

## Graphs to include after genuine data validation
1. **Microstructure gallery:** Show representative synthetic binary 64×64 images, annotated with phase convention, porosity and image ID. Do not call these microscopy photos.
2. **Clean-model comparison:** MAE and R² for geometric descriptors versus standardized pixel PCA + Ridge, including units of the chosen tensor component and test-set count. Add a mean-baseline reference in a later revision.
3. **Noise robustness:** MAE (and R²) against added grayscale pixel-noise standard deviation σ=0, 0.05, 0.10. Label this as sensor-like numerical intensity corruption, not physically simulated degradation. A flat geometry curve can arise from the hard threshold at 127, not inherent physical robustness.
4. **Runtime comparison:** Measured wall clock training+inference duration and environment. Scale depends on hardware and sample count; it is not algorithmic complexity in isolation.
5. **Future scientific diagnostics:** parity scatter plot (predicted vs actual tensor component), residuals versus porosity, and subgroup errors for connectivity regimes; report only after verified external labels are loaded.

## Engineering interpretation to write after running Mendeley data
- **Physical meaning:** Porosity measures volume-area proxy, connectivity and spanning encode coarse potential transport paths, and interface density estimates particle/pore boundaries. These 2D approximations cannot fully specify tortuosity or direction-dependent transport.
- **Comparative models:** Geometry is lower-dimensional and interpretable; PCA can preserve broad image patterns but can miss important localized topology. Correlation is not proof of a mechanistic law.
- **Noise:** Geometry features are calculated after image thresholding. Their stability under modest additive noise can partly reflect threshold choice. Repeat under threshold shifts and morphological perturbations before making a robustness claim.
- **Data leakage:** Split by image before training PCA/scalers. Current random split does not guarantee independence across similar synthetic-generator settings. Consider morphology-group splits.
- **Physics caveat:** Regression does not solve transport PDEs or enforce physical constraints; it *uses* PDE-derived labels, which is a physics-aware surrogate baseline, **not** a physics-informed neural network.
- **Target semantics:** Resolve whether the chosen numeric label is a diagonal component, another tensor entry, or normalized property; verify image-label identifiers, dimensionality and convention before reporting results.

## What has actually been verified
`tests/test_pipeline.py` tests geometric descriptors, numeric label parsing, and identifiers. `make_demo_data.py` builds purely artificial images and toy algebraic labels. GitHub Actions runs these tests and executes the toy smoke test. Real scientific performance is **unknown**.
