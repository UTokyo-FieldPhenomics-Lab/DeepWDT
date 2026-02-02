import os
import zipfile
import tempfile
from pathlib import Path
from collections import deque
from tqdm import tqdm

def lp(p: Path) -> str:
    p = p.resolve()
    return f"\\\\?\\{p}" if os.name == "nt" else str(p)

BAD_CHARS = '<>:"/\\|?*'

def safe_flat_name(zip_path: Path, member_name: str) -> str:
    # Keep only the file name (flatten) and sanitize
    base = Path(member_name).name
    for ch in BAD_CHARS:
        base = base.replace(ch, "_")
    # Optional: prefix with zip name to reduce collisions
    prefix = zip_path.stem
    return f"{prefix}__{base}"

def unique_path(out_dir: Path, filename: str) -> Path:
    p = out_dir / filename
    if not p.exists():
        return p
    stem, suf = Path(filename).stem, Path(filename).suffix
    i = 1
    while True:
        cand = out_dir / f"{stem}__{i}{suf}"
        if not cand.exists():
            return cand
        i += 1

def extract_tifs_from_zip(zip_path: Path, out_dir: Path, queue: deque):
    # Extract only TIFFs flat, and persist nested zips to process later
    with zipfile.ZipFile(lp(zip_path), "r") as z:
        for info in z.infolist():
            if info.is_dir():
                continue

            ext = Path(info.filename).suffix.lower()

            if ext in (".tif", ".tiff"):
                out_name = safe_flat_name(zip_path, info.filename)
                dest = unique_path(out_dir, out_name)
                with z.open(info, "r") as src, open(lp(dest), "wb") as dst:
                    dst.write(src.read())

            elif ext == ".zip":
                # persist nested zip to disk so we can process it
                with z.open(info, "r") as src:
                    nested_name = safe_flat_name(zip_path, info.filename)
                    nested_dest = unique_path(out_dir, nested_name)  # store in out_dir
                    with open(lp(nested_dest), "wb") as dst:
                        dst.write(src.read())
                queue.append(nested_dest)

def unzip_all_keep_tifs_flat(root, out_dir):
    root = Path(root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    q = deque(root.rglob("*.zip"))
    seen = set()

    pbar = tqdm(total=len(q), desc="ZIPs processed", unit="zip")

    while q:
        zp = Path(q.popleft()).resolve()
        if zp in seen:
            pbar.update(1)
            continue
        seen.add(zp)

        try:
            extract_tifs_from_zip(zp, out_dir, q)
        except zipfile.BadZipFile as e:
            print(f"\nBAD ZIP: {zp}\n  Reason: {e}")
        except Exception as e:
            print(f"\nFAILED ZIP: {zp}\n  Reason: {type(e).__name__}: {e}")

        # If we discovered more zips, update tqdm total
        if len(seen) < (pbar.total or 0) and len(q) > 0:
            pass  # harmless; tqdm total will be adjusted below

        if pbar.total < len(seen) + len(q):
            pbar.total = len(seen) + len(q)
            pbar.refresh()

        pbar.update(1)

    pbar.close()
    print(f"\nDone. TIFs are in: {out_dir}")

if __name__ == "__main__":
    unzip_all_keep_tifs_flat(
        root=r"C:\Users\CREST\Downloads\data",
        out_dir=r"C:\tifs_out",
    )
