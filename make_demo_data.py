"""Generate toy binary images and invented labels for software smoke tests ONLY.
These targets are algebraic functions of image features, NOT PDE calculations.
"""
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from io import BytesIO
import numpy as np
from PIL import Image


def generate(dest=Path('demo_data'), n=100, seed=42):
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    with ZipFile(dest/'images.zip','w',ZIP_DEFLATED) as zi, ZipFile(dest/'diffusion.zip','w',ZIP_DEFLATED) as zl:
        for i in range(n):
            # Smoothed random field yields connected artificial patterns.
            from scipy.ndimage import gaussian_filter
            field = gaussian_filter(rng.normal(size=(64,64)), sigma=float(rng.uniform(1.5,5)))
            fraction = float(rng.uniform(.25,.72))
            mask = (field > np.quantile(field, 1-fraction)).astype(np.uint8)*255
            stream = BytesIO(); Image.fromarray(mask).save(stream,format='TIFF')
            zi.writestr(f'image_{i:05d}.tif',stream.getvalue())
            # Fabricated toy formula to test code; NOT a scientific transport label.
            y = 0.2+1.3*mask.mean()/255 + 0.03*rng.normal()
            zl.writestr(f'diffusion_{i:05d}.txt', f'{y:.7f}\n')
    print('TOY DATA ONLY: not Mendeley images or PDE targets:',dest)

if __name__ == '__main__': generate()
