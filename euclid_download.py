"""
Download Euclid DR1 files listed in a VOTable from the ESAC archive.

Usage:
    python euclid_download.py observations.vot --output ./raw/ --workers 4

The VOTable must contain a 'file_name' column with filenames like:
    EUC_VIS_SWL_DET-..._00.fits.gz
"""

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


def load_credentials(cred_file: str) -> tuple[str, str]:
    with open(cred_file) as f:
        lines = f.read().splitlines()
    if len(lines) < 2:
        sys.exit(f"ERROR: {cred_file} must have username on line 1, password on line 2.")
    return lines[0].strip(), lines[1].strip()


def login(username: str, password: str, cookie_file: str) -> None:
    cmd = [
        "curl", "-k", "-s",
        "-c", cookie_file,
        "-X", "POST",
        "-d", f"username={username}",
        "-d", f"password={password}",
        "-L", "https://easidr.esac.esa.int/sas-dd/login",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        sys.exit(f"ERROR: login failed.\n{result.stderr.decode()}")


def download_file(
    file_name: str,
    output_path: str,
    cookie_file: str,
    release: str,
) -> tuple[str, bool, str]:
    """Returns (file_name, success, message)."""
    if os.path.exists(output_path):
        return file_name, True, "skipped (exists)"

    url = (
        f"https://easidr.esac.esa.int/sas-dd/data"
        f"?file_name={file_name}&release={release}&retrieval_type=FILE"
    )
    tmp = output_path + ".part"
    cmd = ["curl", "-k", "-s", "-b", cookie_file, "-o", tmp, url]
    print(' '.join(cmd))
    result = subprocess.run(cmd, capture_output=True)

    if result.returncode != 0:
        return file_name, False, result.stderr.decode().strip()

    # Detect HTML error pages returned with HTTP 200
    if os.path.exists(tmp):
        with open(tmp, "rb") as f:
            head = f.read(512)
        if b"<html" in head.lower() or b"<!doctype" in head.lower():
            os.remove(tmp)
            return file_name, False, "server returned HTML (auth error or file not found)"
        os.rename(tmp, output_path)
        return file_name, True, f"ok ({os.path.getsize(output_path) / 1e6:.1f} MB)"

    return file_name, False, "no output written"


def read_votable(path: str, column: str) -> list[str]:
    try:
        from astropy.table import Table
        t = Table.read(path)
    except Exception as e:
        sys.exit(f"ERROR reading VOTable: {e}")

    if column not in t.colnames:
        sys.exit(
            f"ERROR: column '{column}' not found.\n"
            f"Available columns: {', '.join(t.colnames)}"
        )
    return [str(v) for v in t[column]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Euclid files from ESAC archive.")
    parser.add_argument("votable", help="Path to VOTable (.vot / .xml / .fits)")
    parser.add_argument("--output", default="./euclid_raw/", help="Output directory")
    parser.add_argument("--column", default="file_name", help="VOTable column with filenames")
    parser.add_argument("--cred", default="cred.txt", help="Credentials file (user\\npass)")
    parser.add_argument("--release", default="sedm", help="Data release (default: sedm)")
    parser.add_argument("--workers", type=int, default=4, help="Parallel downloads")
    parser.add_argument("--cookie", default="cookies.txt", help="Cookie jar file")
    args = parser.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Read file list
    file_names = read_votable(args.votable, args.column)
    print(f"Found {len(file_names)} entries in '{args.column}' column.")

    jobs: list[tuple[str, str]] = [
        (fn, str(out_dir / fn)) for fn in file_names if fn
    ]

    if not jobs:
        sys.exit("ERROR: no downloadable filenames found. Check --column value.")

    # Authenticate once
    print(f"Logging in to ESAC ({args.cred}) ...")
    username, password = load_credentials(args.cred)
    login(username, password, args.cookie)
    print("Login OK.\n")

    # Download
    done = skipped = failed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_file, fn, op, args.cookie, args.release): fn
            for fn, op in jobs
        }
        for future in as_completed(futures):
            file_name, success, msg = future.result()
            if success:
                if "skipped" in msg:
                    skipped += 1
                else:
                    done += 1
            else:
                failed += 1
            status = "OK" if success else "FAIL"
            print(f"  [{status}] {file_name}: {msg}")

    print(f"\nDone. {done} downloaded, {skipped} skipped, {failed} failed.")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
