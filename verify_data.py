"""Verify the exact release dataset without modifying it."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
for item in json.loads((root / 'data_manifest.json').read_text()):
    path = root / 'colab_test_data' / item['name']
    if not path.is_file() or path.stat().st_size != item['bytes']:
        raise SystemExit(f'Missing or incorrect size: {path.name}')
    with path.open('rb') as source:
        checksum = hashlib.file_digest(source, 'sha256').hexdigest()
    if checksum != item['sha256']:
        raise SystemExit(f'Checksum mismatch: {path.name}')
    print('OK', path.name)
print('All 10 input files match the release manifest.')
