"""Paired synthetic-electrode baselines; outputs measured results only after execution.

Run inspect_dataset.py first. Target is the first row/column numeric value in a
label file, ASSUMING verified matrix layout. Override --target-index if needed.
"""
import argparse
import csv
import json
import time
import re
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from PIL import Image
from io import BytesIO
from scipy.ndimage import label
from skimage.measure import perimeter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

HERE = Path(__file__).resolve().parent

def key(name):
    numbers = re.findall(r'\d+', Path(name).stem)
    return str(int(numbers[-1])) if numbers else Path(name).stem.lower()

def target(b, i):
    # Require a plain numeric matrix, rather than silently misreading headers.
    text = b.decode('utf-8-sig', 'replace')
    rows = [ln.strip().replace(',', ' ').split() for ln in text.splitlines() if ln.strip()]
    try:
        vals = [float(x) for row in rows for x in row]
    except ValueError as exc:
        raise ValueError('Non-numeric label file; inspect its layout before training') from exc
    if len(vals) <= i:
        raise ValueError(f'Tensor index {i} unavailable: {len(vals)} numeric entries')
    return vals[i]

def features(im, pore_is_white):
    pore = (im > 127) if pore_is_white else (im <= 127)
    components, n = label(pore)
    counts = np.bincount(components.ravel())[1:]
    largest = counts.max() / pore.sum() if len(counts) and pore.sum() else 0.0
    spanning = 0.0
    for j in range(1, n + 1):
        mask = components == j
        if (mask[0].any() and mask[-1].any()) or (mask[:, 0].any() and mask[:, -1].any()):
            spanning = 1.0
            break
    interfaces = (np.count_nonzero(pore[:, 1:] != pore[:, :-1]) +
                  np.count_nonzero(pore[1:, :] != pore[:-1, :])) / pore.size
    return [pore.mean(), largest, spanning, interfaces]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data-dir', type=Path, default=HERE/'data', help='Archive directory (data/ for Mendeley, demo_data/ for toy smoke test)')
    p.add_argument('--sample-size', type=int, default=1000)
    p.add_argument('--target-index', type=int, default=0,
                   help='Zero-based numeric tensor-entry index; confirm against dataset documentation')
    p.add_argument('--pore-is-white', action='store_true',
                   help='Set ONLY after verifying pore phase convention')
    p.add_argument('--pore-is-black', action='store_true')
    args = p.parse_args()
    if args.pore_is_white == args.pore_is_black:
        p.error('Specify exactly one: --pore-is-white OR --pore-is-black (verify phase convention first)')
    a, b = args.data_dir/'images.zip', args.data_dir/'diffusion.zip'
    if not a.exists() or not b.exists():
        p.error('Download images.zip and diffusion.zip from Mendeley into data/ first.')
    X_geo, X_pixels, ys = [], [], []
    with ZipFile(a) as zi, ZipFile(b) as zl:
        imgs = {key(x): x for x in zi.namelist() if x.lower().endswith(('.tif', '.tiff'))}
        labels = {key(x): x for x in zl.namelist() if x.lower().endswith('.txt')}
        if len(imgs) != len(set(imgs)) or len(labels) != len(set(labels)):
            raise RuntimeError('Nonunique image/label identifiers')
        paired = sorted(set(imgs) & set(labels), key=lambda x: (not x.isdigit(), int(x) if x.isdigit() else x))
        if not paired:
            raise RuntimeError('No matching stems/IDs found; inspect_dataset.py and adapt key()')
        print('Matched pairs:', len(paired), 'Sample:', [(imgs[k], labels[k]) for k in paired[:3]])
        rng = np.random.default_rng(42)
        chosen = rng.choice(paired, size=min(args.sample_size, len(paired)), replace=False)
        for k in chosen:
            im = np.asarray(Image.open(BytesIO(zi.read(imgs[k]))).convert('L'))
            if im.shape != (64, 64):
                raise ValueError(f'Unexpected image dimension {im.shape} for {imgs[k]}')
            y = target(zl.read(labels[k]), args.target_index)
            if not np.isfinite(y):
                continue
            X_geo.append(features(im, args.pore_is_white))
            X_pixels.append(im.ravel() / 255.)
            ys.append(y)
    if len(ys) < 30:
        raise RuntimeError('Too few valid samples; inspect image-label pairing')
    geo, pix, y = np.asarray(X_geo), np.asarray(X_pixels), np.asarray(ys)
    ids_train, ids_test = train_test_split(np.arange(len(y)), test_size=.2, random_state=42)
    models = {
        'geometric_ridge': make_pipeline(StandardScaler(), Ridge(alpha=10.0)),
        'pixel_pca_ridge': make_pipeline(StandardScaler(), PCA(n_components=min(20, len(ids_train)-1), random_state=42), Ridge(alpha=10.0)),
    }
    out = HERE/'results'; out.mkdir(exist_ok=True)
    rows=[]
    runtimes={}
    for name, model in models.items():
        X = geo if name.startswith('geometric') else pix
        started=time.perf_counter()
        model.fit(X[ids_train], y[ids_train]); predicted=model.predict(X[ids_test])
        runtime=time.perf_counter()-started
        runtimes[name]=runtime
        rows.append([name,'clean',mean_absolute_error(y[ids_test],predicted),r2_score(y[ids_test],predicted),len(ids_test)])
        for sigma in [.05, .10]:
            if name.startswith('geometric'):
                # Perturb raw pixel intensities, then rederive engineering features.
                noisy = np.clip(pix[ids_test] + np.random.default_rng(42).normal(0,sigma,(len(ids_test),4096)),0,1)
                X_noisy=np.asarray([features((r.reshape(64,64)*255).astype('uint8'),args.pore_is_white) for r in noisy])
            else:
                X_noisy=np.clip(X[ids_test]+np.random.default_rng(42).normal(0,sigma,X[ids_test].shape),0,1)
            pred=model.predict(X_noisy)
            rows.append([name,f'pixel_noise_sigma_{sigma}',mean_absolute_error(y[ids_test],pred),r2_score(y[ids_test],pred),len(ids_test)])
    path = out/'metrics.csv'
    with path.open('w', newline='') as f:
        w=csv.writer(f); w.writerow(['model','condition','MAE','R2','test_samples']); w.writerows(rows)
    (out/'runtime.json').write_text(json.dumps({'training_and_inference_seconds':runtimes,'dataset_type': 'TOY_NOT_PDE' if args.data_dir.name=='demo_data' else 'UNVERIFIED_EXTERNAL_DATA', 'sample_count':len(y), 'seed':42}, indent=2))
    make_plots(out, rows, pix[ids_test], geo[ids_test], y[ids_test], args.data_dir.name == 'demo_data')
    print('Results written:',path)
    for row in rows: print(row)

def make_plots(out, rows, pixels, geo, y, is_toy=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Plots contain only measured run metrics; never prefilled numbers.
    models = sorted(set(r[0] for r in rows))
    conditions = ['clean', 'pixel_noise_sigma_0.05', 'pixel_noise_sigma_0.1']
    fig, ax = plt.subplots(figsize=(8,4))
    x=np.arange(len(conditions)); width=.35
    for i, model in enumerate(models):
        m={r[1]: r[2] for r in rows if r[0]==model}
        ax.bar(x+(i-.5)*width,[m[c] for c in conditions], width,label=model)
    ax.set_xticks(x, ['Clean','Noise σ=0.05','Noise σ=0.10'])
    ax.set_ylabel('MAE (target units)'); ax.set_title(('TOY SMOKE TEST ONLY — ' if is_toy else '') + 'MAE versus image noise')
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out/'mae_noise.png',dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(8,4))
    for model in models:
        m={r[1]: r[3] for r in rows if r[0]==model}
        ax.plot(conditions,[m[c] for c in conditions],marker='o',label=model)
    ax.set_xticks(range(3),['Clean','σ=0.05','σ=0.10']); ax.set_ylabel('R²'); ax.set_title(('TOY SMOKE TEST ONLY — ' if is_toy else '') + 'R² versus image noise')
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out/'r2_noise.png',dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1,3,figsize=(10,3))
    for j in range(3):
        axes[j].imshow(pixels[j].reshape(64,64), cmap='gray',vmin=0,vmax=1)
        axes[j].set_title(f'Sample {j+1}'); axes[j].axis('off')
    fig.suptitle(('TOY DATA — ' if is_toy else '') + 'Held-out synthetic image inputs (not microscopy)')
    fig.tight_layout(); fig.savefig(out/'example_microstructures.png',dpi=160); plt.close(fig)

if __name__ == '__main__': main()
