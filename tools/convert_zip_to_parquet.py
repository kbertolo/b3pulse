#!/usr/bin/env python3
"""
convert_zip_to_parquet.py

Usage:
  python convert_zip_to_parquet.py /path/to/COTAHIST_A2025.ZIP ./data/parquet_cotas

What it does:
 - extracts files from the ZIP into a temporary folder
 - finds CSVs inside the zip
 - tries to detect delimiter (using csv.Sniffer) and encoding (utf-8, then latin1 or chardet if installed)
 - reads each CSV in chunks and writes a Parquet file per CSV using pyarrow backend
 - safe for large files (uses chunked read + pyarrow ParquetWriter)
"""
import sys
import zipfile
import tempfile
from pathlib import Path
import csv
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# optional: chardet for better encoding detection
try:
    import chardet
except Exception:
    chardet = None

CHUNK_SIZE = 200_000  # rows per chunk; reduce if memory is tight

def detect_encoding(sample_bytes: bytes):
    # try chardet first if available
    if chardet:
        res = chardet.detect(sample_bytes)
        return res['encoding'] or 'utf-8'
    # fallback simple attempts
    for enc in ("utf-8", "latin1", "cp1252"):
        try:
            sample_bytes.decode(enc)
            return enc
        except Exception:
            continue
    return "utf-8"

def detect_delimiter(sample_text: str):
    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample_text[:2000])
        return dialect.delimiter
    except Exception:
        # fallback common delimiters
        for d in [',', ';', '\t', '|']:
            if d in sample_text:
                return d
    return ','

def convert_csv_to_parquet_stream(csv_path: Path, parquet_path: Path, encoding=None, delimiter=None):
    """Read CSV in chunks and write a single Parquet using pyarrow ParquetWriter."""
    print(f"[convert] {csv_path} -> {parquet_path} (enc={encoding}, delim={delimiter})")
    # Open a reader to get a first small chunk to infer schema
    first_chunk = None
    reader = pd.read_csv(csv_path, sep=delimiter or ',', encoding=encoding or 'utf-8',
                         low_memory=False, nrows=CHUNK_SIZE)
    first_chunk = reader
    table = pa.Table.from_pandas(first_chunk, preserve_index=False)
    pq_writer = pq.ParquetWriter(str(parquet_path), table.schema, use_dictionary=True, compression='SNAPPY')

    # write first chunk
    pq_writer.write_table(table)
    # now iterate remaining chunks
    for chunk in pd.read_csv(csv_path, sep=delimiter or ',', encoding=encoding or 'utf-8',
                             low_memory=False, chunksize=CHUNK_SIZE, skiprows=CHUNK_SIZE, header=0):
        # pandas' chunks with skiprows above is a bit awkward. Instead, read in chunked mode properly:
        pass  # we'll re-open below more cleanly

    # reopen properly and stream
    with pd.read_csv(csv_path, sep=delimiter or ',', encoding=encoding or 'utf-8',
                     low_memory=False, chunksize=CHUNK_SIZE) as it:
        first = True
        for chunk in it:
            tbl = pa.Table.from_pandas(chunk, preserve_index=False)
            pq_writer.write_table(tbl)
    pq_writer.close()
    print(f"[done] wrote {parquet_path}")

def process_zip(zip_path: Path, out_dir: Path):
    if not zip_path.exists():
        print(f"ERROR: zip not found: {zip_path}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        print(f"Extracting {zip_path} to temporary dir {td}")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(td)

        # find CSV-like files (csv, txt) recursively
        files = list(td_path.rglob("*.csv")) + list(td_path.rglob("*.txt"))
        if not files:
            # also check for .ZIP entries that might be compressed inside (nested zips)
            print("No CSV files found inside zip. Listing all files:")
            for name in z.namelist():
                print(" -", name)
            return

        for f in files:
            print("Processing:", f)
            # read small sample for detection
            sample_bytes = f.read_bytes()[:8192]
            enc = None
            if chardet:
                enc = detect_encoding(sample_bytes)
            else:
                # try common encodings
                for e in ("utf-8", "latin1", "cp1252"):
                    try:
                        sample_bytes.decode(e)
                        enc = e
                        break
                    except Exception:
                        continue
                if not enc:
                    enc = "latin1"

            # read a text sample to detect delimiter
            try:
                sample_text = sample_bytes.decode(enc, errors='replace')
            except Exception:
                sample_text = sample_bytes.decode("latin1", errors='replace')

            delim = detect_delimiter(sample_text)
            # output parquet path
            out_file = out_dir / (f.stem + ".parquet")
            try:
                # use streaming conversion to avoid memory blow-up
                # We'll create writer by reading the first chunk separately (safer)
                # Read first small chunk to infer schema
                first = pd.read_csv(f, sep=delim, encoding=enc, engine='python', nrows=CHUNK_SIZE, low_memory=False)
                table = pa.Table.from_pandas(first, preserve_index=False)
                writer = pq.ParquetWriter(str(out_file), table.schema, compression='SNAPPY')
                writer.write_table(pa.Table.from_pandas(first, preserve_index=False))
                # iterate remaining chunks
                for chunk in pd.read_csv(f, sep=delim, encoding=enc, engine='python',
                                         chunksize=CHUNK_SIZE, low_memory=False, skiprows=CHUNK_SIZE, header=0):
                    writer.write_table(pa.Table.from_pandas(chunk, preserve_index=False))
                writer.close()
                print(f"Converted {f.name} -> {out_file}")
            except Exception as exc:
                print(f"Failed to convert {f} : {exc}")
                # fallback: try reading entire file in one go (if small)
                try:
                    df = pd.read_csv(f, sep=delim, encoding=enc, engine='python', low_memory=False)
                    df.to_parquet(out_file, index=False)
                    print(f"Fallback wrote {out_file}")
                except Exception as exc2:
                    print(f"Fallback also failed for {f}: {exc2}")

def main():
    if len(sys.argv) < 3:
        print("Usage: python convert_zip_to_parquet.py path/to/input.zip path/to/outdir")
        sys.exit(1)
    zip_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2])
    process_zip(zip_path, out_dir)

if __name__ == "__main__":
    main()
