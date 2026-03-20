"""Download and cache all apStar spectra for the training set.

Run in a separate terminal while working elsewhere:
    cd /Users/junruiting/GitHub/ay-128/labs/02
    ../../.venv/bin/python download_spectra.py
"""
import os
import sys
sys.path.insert(0, "/Users/junruiting/GitHub/ay-128")
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from tqdm.auto import tqdm
from ugdatalab.models.sdss import SDSSData
from ugdatalab.spectra import _get_spectra

QUERY = """
SELECT s.*, a.*
FROM apogeeStar AS s
JOIN aspcapStar AS a ON s.apogee_id = a.apogee_id
WHERE s.field IN ('M15', 'N6791', 'K2_C4_168-21', '060+00')
"""

if __name__ == "__main__":
    sdss = SDSSData(QUERY)
    data = sdss.data
    print(f"Training set: {len(data)} stars\n")

    failed = []
    for row in tqdm(data, desc="Downloading apStar files"):
        try:
            _get_spectra(str(row["apogee_id"]), str(row["telescope"]), str(row["field"]))
        except Exception as e:
            failed.append((str(row["apogee_id"]), str(e)))

    print(f"\nDone. {len(data) - len(failed)}/{len(data)} succeeded.")
    if failed:
        print("Failed:")
        for apogee_id, err in failed:
            print(f"  {apogee_id}: {err}")
