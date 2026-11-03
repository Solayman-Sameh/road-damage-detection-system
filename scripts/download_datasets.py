"""Download the two selected original archives, resuming interrupted transfers.

Uses Python's standard library. Verifies publisher MD5, records SHA-256, and
checks every ZIP member's CRC without extracting duplicate image files.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'config' / 'dataset_sources.json'
RESERVE = 2 * 1024**3


def verify(path, spec):
    if path.stat().st_size != spec['bytes']:
        raise ValueError(f'Unexpected archive size: {path}')
    print(f"Checking publisher checksum and ZIP contents: {spec['id']}", flush=True)
    md5, sha = hashlib.md5(), hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            md5.update(block)
            sha.update(block)
    if md5.hexdigest() != spec['md5']:
        raise ValueError(f'Publisher checksum mismatch; file preserved: {path}')
    with zipfile.ZipFile(path) as archive:
        last = time.monotonic()
        entries = archive.infolist()
        for index, entry in enumerate(entries, 1):
            if not entry.is_dir():
                with archive.open(entry) as stream:
                    while stream.read(1024**2):
                        pass  # Reading to EOF checks the member's CRC.
            if time.monotonic() - last > 15:
                print(f"  {spec['id']}: ZIP check {index}/{len(entries)} members", flush=True)
                last = time.monotonic()
        image_count = sum(n.filename.lower().endswith(('.jpg','.jpeg','.png')) for n in entries)
        files = sum(not n.is_dir() for n in entries)
        unpacked = sum(n.file_size for n in entries)
    return {'bytes': path.stat().st_size, 'md5': md5.hexdigest(),
            'sha256': sha.hexdigest(), 'zip_crc': 'passed', 'files': files,
            'images': image_count, 'uncompressed_bytes': unpacked}


def download(spec, directory, retries, verify_only=False):
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / spec['filename']
    partial = directory / (spec['filename'] + '.part')
    if target.exists():
        checks = verify(target, spec)
    elif verify_only:
        raise FileNotFoundError(target)
    else:
        for attempt in range(1, retries+1):
            offset = partial.stat().st_size if partial.exists() else 0
            if offset > spec['bytes']:
                raise ValueError(f'Oversize partial archive preserved: {partial}')
            if offset == spec['bytes']:
                break
            if shutil.disk_usage(directory).free < spec['bytes']-offset+RESERVE:
                raise OSError('Insufficient space for the remaining download and 2 GiB reserve')
            # A distinct query avoids reusing cached small preview ranges.
            url = spec['url'] + ('&' if '?' in spec['url'] else '?') + f'download=1&transfer_offset={offset}'
            request = urllib.request.Request(url, headers={
                'User-Agent':'RoadDatasetDownload/1.0', 'Accept-Encoding':'identity',
                'Range':f'bytes={offset}-', 'Cache-Control':'no-cache'})
            try:
                with urllib.request.urlopen(request, timeout=45) as response:
                    if response.status == 206:
                        match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range',''))
                        if not match or int(match[1]) != offset or int(match[3]) != spec['bytes'] or not offset <= int(match[2]) < spec['bytes']:
                            raise ValueError('Unexpected byte range; partial archive preserved')
                        mode = 'ab' if offset else 'wb'
                    elif response.status == 200:
                        if offset:
                            raise ValueError('Server ignored resume request; partial archive preserved')
                        mode = 'wb'
                    else:
                        raise OSError(f'Unexpected HTTP status {response.status}')
                    total, last = offset, time.monotonic()
                    print(f"Downloading {spec['id']}: {total/1e6:.1f}/{spec['bytes']/1e6:.1f} MB", flush=True)
                    with partial.open(mode) as output:
                        while True:
                            block = response.read(1024**2)
                            if not block:
                                break
                            if total+len(block) > spec['bytes']:
                                raise ValueError('Response exceeds pinned archive size')
                            if shutil.disk_usage(directory).free < RESERVE+len(block):
                                raise OSError('Stopped to preserve free disk space')
                            output.write(block)
                            total += len(block)
                            if time.monotonic()-last >= 15:
                                print(f"  {spec['id']}: {total/1e6:.1f}/{spec['bytes']/1e6:.1f} MB ({100*total/spec['bytes']:.1f}%)", flush=True)
                                last = time.monotonic()
                if total == spec['bytes']:
                    break
                raise OSError(f'Connection ended at {total}/{spec["bytes"]} bytes')
            except urllib.error.HTTPError as error:
                if error.code not in {408,416,429,500,502,503,504}:
                    raise
                reason = str(error)
            except (OSError,http.client.HTTPException) as error:
                reason = str(error)
            if attempt == retries:
                raise OSError(f'{reason}; partial archive retained. Rerun to resume.')
            delay = min(2**attempt,30)
            print(f'Connection interrupted: {reason}. Retrying in {delay}s ({attempt}/{retries}).', flush=True)
            time.sleep(delay)
        checks = verify(partial,spec)
        partial.rename(target)
    record = {**spec, **checks, 'status':'verified', 'archive':str(target.resolve()),
              'verified_at':datetime.now(timezone.utc).isoformat(), 'extracted':False}
    (directory/'download_status.json').write_text(json.dumps(record,indent=2),encoding='utf8')
    print(f"READY: {spec['id']} — {checks['images']:,} images, MD5 and ZIP CRC passed", flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=None,help='Parent data directory; raw/ is created inside it')
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--retries',type=int,default=12)
    args = parser.parse_args()
    if args.retries < 1:
        parser.error('--retries must be positive')
    data = (args.data_dir or Path(os.environ.get('ROAD_DATA_DIR',str(ROOT/'data')))).expanduser().resolve()
    data.mkdir(parents=True,exist_ok=True)
    specs = json.loads(MANIFEST.read_text(encoding='utf8'))['datasets']
    needed = sum(max(0,s['bytes']-(data/'raw'/s['id']/(s['filename']+'.part')).stat().st_size)
                 if (data/'raw'/s['id']/(s['filename']+'.part')).exists() else
                 (0 if (data/'raw'/s['id']/s['filename']).exists() else s['bytes']) for s in specs)
    free = shutil.disk_usage(data).free
    print(f'Data directory: {data}\nFree space: {free/1e9:.2f} GB; remaining downloads: {needed/1e9:.2f} GB',flush=True)
    if not args.verify_only and free < needed+RESERVE:
        raise OSError('Not enough space for both downloads and 2 GiB reserve')
    records = [download(s,data/'raw'/s['id'],args.retries,args.verify_only) for s in specs]
    (data/'download_status.json').write_text(json.dumps({'datasets':records},indent=2),encoding='utf8')
    print('Both dataset archives are downloaded and verified. Archives have not been extracted.',flush=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Stopped; partial files retained. Rerun the same command to resume.',file=sys.stderr)
        raise SystemExit(130)
    except (OSError,ValueError,zipfile.BadZipFile) as error:
        print(f'ERROR: {error}',file=sys.stderr)
        raise SystemExit(1)
