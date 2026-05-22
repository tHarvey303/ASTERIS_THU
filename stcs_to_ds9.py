#!/usr/bin/env python3
"""
Convert a column of STC-S polygon strings into a DS9 region file.

Input STC-S format (one per row):
    Polygon J2000 RA1 Dec1 RA2 Dec2 RA3 Dec3 RA4 Dec4 ...

Usage:
    python stcs_to_ds9.py input.csv output.reg [--column stc_s]
    python stcs_to_ds9.py input.vot output.reg [--column stc_s]
"""

import argparse
import csv
import sys
from pathlib import Path


def parse_stcs_polygon(stcs_string):
    """
    Parse an STC-S Polygon string into a list of (ra, dec) vertex tuples.

    Expected format: 'Polygon <FRAME> ra1 dec1 ra2 dec2 ...'
    Returns None if the string can't be parsed as a polygon.
    """
    if stcs_string is None:
        return None
    s = str(stcs_string).strip()
    if not s:
        return None

    tokens = s.split()
    if len(tokens) < 4 or tokens[0].lower() != 'polygon':
        return None

    # tokens[1] is the frame (J2000, ICRS, etc.) — we map everything to fk5/J2000 for DS9
    coord_tokens = tokens[2:]
    if len(coord_tokens) % 2 != 0:
        return None

    try:
        coords = [float(t) for t in coord_tokens]
    except ValueError:
        return None

    vertices = list(zip(coords[0::2], coords[1::2]))
    if len(vertices) < 3:
        return None
    return vertices


def stcs_to_ds9_region(vertices):
    """Convert a list of (ra, dec) vertices into a DS9 polygon region line."""
    flat = ','.join(f'{ra:.8f},{dec:.8f}' for ra, dec in vertices)
    return f'polygon({flat})'


def read_column(path, column):
    """
    Yield values from the named column. Supports CSV and VOTable (.vot/.xml)
    inputs. Falls back to treating the file as one STC-S string per line if
    no column name is given and the file isn't CSV-like.
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {'.vot', '.xml', '.votable'}:
        try:
            from astropy.io.votable import parse_single_table
        except ImportError:
            sys.exit("VOTable input requires astropy. Install with: pip install astropy")
        table = parse_single_table(str(path)).to_table()
        if column not in table.colnames:
            sys.exit(f"Column '{column}' not found. Available: {table.colnames}")
        for value in table[column]:
            yield value

    elif suffix in {'.csv', '.tsv', '.txt'}:
        delimiter = '\t' if suffix == '.tsv' else ','
        with open(path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            if column not in reader.fieldnames:
                sys.exit(f"Column '{column}' not found. Available: {reader.fieldnames}")
            for row in reader:
                yield row[column]

    else:
        # Treat each non-empty line as a raw STC-S string
        with open(path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    yield line


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('input', help='Input file (CSV, TSV, VOTable, or plain text)')
    parser.add_argument('output', help='Output DS9 region file')
    parser.add_argument('--column', default='stc_s',
                        help="Name of the column containing STC-S strings (default: stc_s)")
    parser.add_argument('--color', default='green', help='DS9 region color (default: green)')
    args = parser.parse_args()

    n_total = 0
    n_written = 0
    n_skipped = 0

    with open(args.output, 'w', encoding='utf-8') as out:
        out.write('# Region file format: DS9 version 4.1\n')
        out.write(f'global color={args.color} dashlist=8 3 width=1 '
                  'font="helvetica 10 normal roman" select=1 highlite=1 '
                  'dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1\n')
        out.write('fk5\n')

        for value in read_column(args.input, args.column):
            n_total += 1
            verts = parse_stcs_polygon(value)
            if verts is None:
                n_skipped += 1
                continue
            out.write(stcs_to_ds9_region(verts) + '\n')
            n_written += 1

    print(f"Read {n_total} rows -> wrote {n_written} polygons "
          f"({n_skipped} skipped) to {args.output}")


if __name__ == '__main__':
    main()