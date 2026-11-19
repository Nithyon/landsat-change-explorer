"""Execute the local combined notebook and retain real text/image cell outputs.

This runner is for this repository's trusted notebook only. Jupyter Run All is
the normal interactive alternative. No network calls or package installs occur.
"""
import base64
import contextlib
import io
import json
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent


def execute():
    path = ROOT / 'Landsat_PBL_1_2_3.ipynb'
    notebook = json.loads(path.read_text(encoding='utf-8'))
    namespace = {'__name__': '__main__'}
    count = 0
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        count += 1
        outputs, stream = [], io.StringIO()

        def capture(*objects, **kwargs):
            for obj in objects:
                if hasattr(obj, 'data') and isinstance(obj.data, bytes):
                    outputs.append({'output_type': 'display_data', 'metadata': {},
                                    'data': {'image/png': base64.b64encode(obj.data).decode('ascii')}})
                else:
                    outputs.append({'output_type': 'display_data', 'metadata': {}, 'data': {'text/plain': str(obj)}})

        print(f'Executing code cell {count} (notebook cell {index})', flush=True)
        with patch('IPython.display.display', capture), contextlib.redirect_stdout(stream):
            # The first cell imports display into the persistent notebook namespace.
            if 'display' in namespace:
                namespace['display'] = capture
            exec(compile(''.join(cell['source']), f'{path.name}:cell{index}', 'exec'), namespace)
        if stream.getvalue():
            outputs.insert(0, {'output_type': 'stream', 'name': 'stdout', 'text': stream.getvalue()})
        cell['outputs'], cell['execution_count'] = outputs, count
    path.write_text(json.dumps(notebook, indent=1), encoding='utf-8')
    print(f'PASS: all {count} notebook code cells executed; outputs retained.', flush=True)


if __name__ == '__main__':
    execute()
