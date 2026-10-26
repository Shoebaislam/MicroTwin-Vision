"""Inspect downloaded Mendeley archives before choosing a tensor target."""
from pathlib import Path
from zipfile import ZipFile
ROOT = Path(__file__).resolve().parent
for fname in ['images.zip', 'diffusion.zip']:
    path = ROOT / 'data' / fname
    if not path.exists():
        print(f'MISSING {path} — download it from https://data.mendeley.com/datasets/mgmxv5tjt2/1')
        continue
    with ZipFile(path) as z:
        items = [x for x in z.namelist() if not x.endswith('/')]
        print(f'\n{fname}: {len(items)} entries; first names: {items[:6]}')
        if fname == 'diffusion.zip':
            for name in next((x for x in items if x.lower().endswith('.txt')), []),:
                if name:
                    with z.open(name) as f:
                        print('FIRST LABEL FILE:', name, '\n', f.read(1000).decode('utf-8', 'replace'))
