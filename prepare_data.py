"""Restore split Drive transfers, then verify all ten original GeoTIFFs."""
import hashlib
import json
import shutil
from pathlib import Path

root = Path(__file__).resolve().parent
folder = root / 'colab_test_data'
folder.mkdir(exist_ok=True)
manifest = json.loads((root / 'data_manifest.json').read_text())
for item in manifest:
    target = folder / item['name']
    if not target.exists():
        parts = sorted(folder.glob(item['name'] + '.part*'))
        if not parts:
            raise SystemExit(f'Missing {item["name"]}: download all files from the Drive folder first.')
        expected = (item['bytes'] + 19_999_999) // 20_000_000
        expected_names = [item['name'] + f'.part{i:02}' for i in range(1, expected + 1)]
        if [p.name for p in parts] != expected_names:
            raise SystemExit(f'Missing or extra transfer parts for {item["name"]}')
        pending = target.with_suffix('.assembling')
        with pending.open('wb') as output:
            for part in parts:
                with part.open('rb') as source:
                    shutil.copyfileobj(source, output)
        with pending.open('rb') as source:
            checksum = hashlib.file_digest(source, 'sha256').hexdigest()
        if pending.stat().st_size != item['bytes'] or checksum != item['sha256']:
            raise SystemExit(f'Invalid transfer parts for {item["name"]}; download them again.')
        pending.replace(target)
        print('Restored', target.name)

exec(compile((root / 'verify_data.py').read_text(), 'verify_data.py', 'exec'))
