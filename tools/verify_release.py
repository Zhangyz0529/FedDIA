from pathlib import Path
import hashlib,json
root=Path(__file__).resolve().parents[1]
entries=json.loads((root/'SHA256SUMS.json').read_text())
fail=[p for p,h in entries.items() if not (root/p).is_file() or hashlib.sha256((root/p).read_bytes()).hexdigest()!=h]
if fail: raise SystemExit('FAIL: '+str(fail))
print(f'PASS: {len(entries)} files match the publication-package hashes.')
