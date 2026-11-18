"""Validate upload names and stage selected GeoTIFFs only during analysis."""
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil

from landsat_analysis import Acquisition, AnalysisError, PATTERN, REQUIRED, discover


def upload_catalog(uploads):
    groups = {}
    seen = set()
    for upload in uploads:
        name = upload.name
        if '/' in name or '\\' in name or Path(name).name != name:
            raise AnalysisError('Upload individual GeoTIFF files with their original Landsat filenames.')
        match = PATTERN.fullmatch(name)
        if not match:
            raise AnalysisError(f'{name}: unsupported filename. Use original Landsat Collection 2 L2SP/L2SR SR filenames.')
        if name.casefold() in seen:
            raise AnalysisError(f'Duplicate uploaded filename: {name}. Remove the duplicate.')
        seen.add(name.casefold())
        identifier, band = match['id'].upper(), match['band'].upper()
        try:
            date = datetime.strptime(match['date'], '%Y%m%d').date().isoformat()
        except ValueError as exc:
            raise AnalysisError(f'Invalid acquisition date in {name}.') from exc
        acquisition = groups.setdefault(identifier, Acquisition(identifier, date, {}))
        if band in acquisition.files:
            raise AnalysisError(f'Duplicate {band} for {identifier}.')
        acquisition.files[band] = Path(name)
    return groups


def upload_signature(uploads):
    # Streamlit assigns a new file_id when file contents are replaced.
    return tuple(sorted((u.name, u.size, u.file_id) for u in uploads))


@contextmanager
def staged_pair(before_uploads, before_id, after_uploads, after_id):
    """Copy uploads to isolated temporary folders; always remove staged files."""
    with TemporaryDirectory(prefix='landsat_uploads_') as temp:
        acquisitions = []
        for label, uploads, identifier in (('before', before_uploads, before_id),
                                           ('after', after_uploads, after_id)):
            groups = upload_catalog(uploads)
            if identifier not in groups:
                raise AnalysisError(f'{label.title()}: select an uploaded acquisition.')
            missing = set(REQUIRED) - set(groups[identifier].files)
            if missing:
                raise AnalysisError(f'{label.title()}: missing {", ".join(sorted(missing))}.')
            folder = Path(temp) / label
            folder.mkdir()
            for upload in uploads:
                if PATTERN.fullmatch(upload.name)['id'].upper() != identifier:
                    continue
                position = upload.tell()
                try:
                    upload.seek(0)
                    with (folder / upload.name).open('wb') as target:
                        shutil.copyfileobj(upload, target, length=8 * 1024 * 1024)
                finally:
                    upload.seek(position)
            acquisitions.append(discover(str(folder))[identifier])
        yield tuple(acquisitions)
