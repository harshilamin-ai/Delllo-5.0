from __future__ import annotations
import csv, json, re, shutil, datetime as dt, hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

def now_stamp() -> str: return dt.datetime.now().strftime('%d%m%y%H%M%S')
def read_json(path: Path) -> Dict[str, Any]:
    with path.open('r', encoding='utf-8') as f: return json.load(f)
def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f: json.dump(data, f, ensure_ascii=False, indent=2)
def read_csv(path: Path) -> List[Dict[str, Any]]:
    with path.open('r', encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))
def find_latest(pattern: str, folder: Path) -> Optional[Path]:
    files=sorted(folder.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return files[0] if files else None
def copy_inputs_to_output(input_dir: Path, output_run_dir: Path) -> Dict[str,str]:
    dest=output_run_dir/'INPUTS'; dest.mkdir(parents=True, exist_ok=True); copied={}
    for path in input_dir.glob('*'):
        if path.is_file():
            target=dest/path.name; shutil.copy(path,target); copied[path.name]=str(target)
    return copied
def parse_json_cell(value: Any, default: Any=None) -> Any:
    if default is None: default={}
    if isinstance(value,(dict,list)): return value
    if value is None or value=='': return default
    try: return json.loads(value)
    except Exception: return default
def normalise_text(value: Any) -> str: return re.sub(r'\s+',' ',str(value or '')).strip().lower()
def token_set(value: Any) -> set: return set(re.findall(r'[a-zA-Z0-9]{3,}', normalise_text(value)))
def safe_float(value: Any, default: float=0.0) -> float:
    try:
        if value is None or value=='': return default
        return float(value)
    except Exception: return default
def stable_id(prefix: str, text: str, width: int=8) -> str:
    return f"{prefix}_{hashlib.sha1(text.encode('utf-8')).hexdigest()[:width].upper()}"
def slug(value: str) -> str: return re.sub(r'[^A-Za-z0-9]+','_',value).strip('_')[:120]
