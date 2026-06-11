#!/usr/bin/env python3
"""
Download and prepare the ESA Kelvins Collision Avoidance Challenge dataset.

Usage:
    python download_data.py                     # Download full dataset (~9GB)
    python download_data.py --sample 5000       # Download and extract 5000 event sample

The ESA Kelvins dataset contains ~160K CDM messages across ~12K events,
with labelled collision probabilities and maneuver decisions.

Dataset URL: https://kelvins.esa.int/collision-avoidance-challenge/data/
"""
import argparse
import os
import sys
import zipfile
import urllib.request
import shutil
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
ESA_URL = "https://zenodo.org/records/4463683/files/Collision%20Avoidance%20Challenge%20-%20Dataset.zip"
ESA_FILENAME = "esa_kelvins_dataset.zip"
ESA_CSV = "train_data.csv"
EXPECTED_CSV_LINES = 160000  # approximate


def download(url: str, dest: Path):
    """Download with progress bar."""
    print(f"Downloading {url}...")
    print(f"  Destination: {dest} ({dest.stat().st_size / 1e9:.1f} GB)" if dest.exists() else "  Starting download...")

    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size / 1e6
        total = total_size / 1e6 if total_size > 0 else 0
        if total > 0:
            pct = min(100, downloaded / total * 100)
            sys.stdout.write(f"\r  {downloaded:.0f}/{total:.0f} MB ({pct:.0f}%)")
            sys.stdout.flush()
        else:
            sys.stdout.write(f"\r  {downloaded:.0f} MB downloaded")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, dest, reporthook)
    print()
    print(f"  Download complete: {dest}")


def extract(zip_path: Path, extract_dir: Path, sample: int = 0):
    """Extract zip. If sample > 0, only keep first N events."""
    print(f"\nExtracting {zip_path}...")
    extract_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        # Find the nested train_data.zip
        nested_train_zip = None
        csv_in_zip = None
        for name in zf.namelist():
            if name.endswith('train_data.zip'):
                nested_train_zip = name
            if name.endswith('.csv'):
                csv_in_zip = name

        if nested_train_zip:
            # Zenodo dataset has nested train_data.zip
            print(f"  Found nested zip: {nested_train_zip}")
            tmp_dir = extract_dir / "tmp_nested"
            os.makedirs(tmp_dir, exist_ok=True)
            zf.extract(nested_train_zip, tmp_dir)
            inner_zip = tmp_dir / nested_train_zip
            with zipfile.ZipFile(inner_zip, 'r') as inner_zf:
                csv_name = [n for n in inner_zf.namelist() if n.endswith('.csv')]
                if not csv_name:
                    print("  ERROR: No CSV in nested zip")
                    return None
                csv_name = csv_name[0]
                print(f"  Found CSV in nested zip: {csv_name}")
                if sample > 0:
                    return _sample_csv(inner_zf, csv_name, extract_dir, sample)
                output_path = extract_dir / "esa_kelvins_train.csv"
                inner_zf.extract(csv_name, extract_dir)
                extracted = extract_dir / os.path.basename(csv_name)
                if extracted != output_path:
                    shutil.move(str(extracted), str(output_path))
                shutil.rmtree(tmp_dir)
                print(f"  Extracted to {output_path}")
                return output_path
        elif csv_in_zip:
            # Flat zip with CSV at top level
            print(f"  Found CSV: {csv_in_zip}")
            if sample > 0:
                with zipfile.ZipFile(zip_path, 'r') as zf2:
                    return _sample_csv(zf2, csv_in_zip, extract_dir, sample)
            output_path = extract_dir / "esa_kelvins_train.csv"
            zf.extract(csv_in_zip, extract_dir)
            extracted = extract_dir / os.path.basename(csv_in_zip)
            if extracted != output_path:
                shutil.move(str(extracted), str(output_path))
            print(f"  Extracted to {output_path}")
            return output_path
        else:
            print("  ERROR: No CSV or nested train_data.zip found in archive")
            return None


def _sample_csv(zf, csv_path: str, extract_dir: Path, sample: int):
    """Extract first N events from a CSV inside a zip."""
    import csv
    print(f"  Extracting sample of {sample} events...")
    tmp_dir = extract_dir / "tmp"
    os.makedirs(tmp_dir, exist_ok=True)
    zf.extract(csv_path, tmp_dir)
    full_csv = tmp_dir / os.path.basename(csv_path)
    events_seen = set()
    output_path = extract_dir / "esa_kelvins_train.csv"

    with open(full_csv) as fin, open(output_path, 'w', newline='') as fout:
        reader = csv.reader(fin)
        writer = csv.writer(fout)
        header = next(reader)
        writer.writerow(header)
        event_col = next(i for i, c in enumerate(header) if 'event' in c.lower() or 'id' in c.lower())

        for row in reader:
            eid = row[event_col]
            if eid in events_seen or len(events_seen) >= sample:
                if eid in events_seen:
                    writer.writerow(row)
            else:
                events_seen.add(eid)
                writer.writerow(row)

    shutil.rmtree(tmp_dir)
    print(f"  Sampled {len(events_seen)} events -> {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Download ESA Kelvins dataset")
    parser.add_argument("--sample", type=int, default=0,
                        help="Extract only N events (default: all ~12K)")
    parser.add_argument("--force", action="store_true",
                        help="Redownload even if file exists")
    args = parser.parse_args()

    DATA_DIR.mkdir(exist_ok=True)
    zip_path = DATA_DIR / ESA_FILENAME
    csv_path = DATA_DIR / "esa_kelvins_train.csv"

    if csv_path.exists() and not args.force:
        print(f"Dataset already exists at {csv_path}")
        print(f"  Size: {csv_path.stat().st_size / 1e6:.0f} MB")
        print("  Use --force to re-download")
        return

    if zip_path.exists() and not args.force:
        print(f"Zip already exists at {zip_path}")
    else:
        download(ESA_URL, zip_path)

    result = extract(zip_path, DATA_DIR, args.sample)
    if result:
        size_mb = result.stat().st_size / 1e6
        print(f"\nDataset ready: {result} ({size_mb:.0f} MB)")


if __name__ == '__main__':
    main()
