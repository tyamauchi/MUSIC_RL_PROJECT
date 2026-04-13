"""
Convert a CSV/TSV/JSON that maps track IDs to metadata into the project's
`data/hetrec2011-lastfm/track_metadata.json` format.

Usage examples:
  # CSV with header id,name,artist,genre,duration_ms,popularity,release_year
  python utils/convert_to_track_metadata.py --input my_tracks.csv --id-col id --name-col name --artist-col artist --genre-col genre --out data/hetrec2011-lastfm/track_metadata.json

  # TSV
  python utils/convert_to_track_metadata.py --input my_tracks.tsv --sep '\t' ...

  # JSON array
  python utils/convert_to_track_metadata.py --input my_tracks.json --format json --id-col id --name-col name --artist-col artist

If some fields are missing in the input, defaults will be filled.
"""
import argparse
import csv
import json
import os
from collections import OrderedDict
import random

try:
    from track_metadata import TRACK_NAMES, GENRES
except Exception:
    TRACK_NAMES = [f"Track {i}" for i in range(1000)]
    GENRES = ["Unknown"]


def load_table(path, fmt='auto', sep=','):
    ext = os.path.splitext(path)[1].lower()
    if fmt == 'auto':
        if ext in ['.json']:
            fmt = 'json'
        elif ext in ['.tsv', '.txt']:
            fmt = 'csv'
            sep = '\t'
        else:
            fmt = 'csv'
    if fmt == 'json':
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # Expect list[dict] or dict
        if isinstance(data, dict):
            # turn dict of id->obj into list
            return list(data.values())
        return data
    else:
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f, delimiter=sep)
            return list(reader)


def to_track_db(rows, id_col='id', name_col='name', artist_col='artist', genre_col='genre', duration_col='duration_ms', popularity_col='popularity', year_col='release_year'):
    track_db = OrderedDict()
    for r in rows:
        # prefer string keys
        if id_col not in r:
            # try common alternatives
            candidates = ['track_id', 'trackid', 'tid']
            found = False
            for c in candidates:
                if c in r:
                    r[id_col] = r[c]
                    found = True
                    break
            if not found:
                continue
        tid_raw = r.get(id_col)
        try:
            tid = int(tid_raw)
        except Exception:
            # if it's like 't123' try to strip non-digits
            import re
            m = re.search(r"(\d+)", str(tid_raw))
            if m:
                tid = int(m.group(1))
            else:
                continue

        name = r.get(name_col) or r.get('title') or f'Unknown Track #{tid}'
        artist = r.get(artist_col) or r.get('artist_name') or 'Unknown Artist'
        genre = r.get(genre_col) or 'Unknown'

        try:
            duration_ms = int(r.get(duration_col)) if r.get(duration_col) else None
        except Exception:
            duration_ms = None
        if duration_ms is None:
            duration_ms = 180000

        try:
            popularity = int(r.get(popularity_col)) if r.get(popularity_col) else None
        except Exception:
            popularity = None
        if popularity is None:
            popularity = 50

        try:
            release_year = int(r.get(year_col)) if r.get(year_col) else None
        except Exception:
            release_year = None
        if release_year is None:
            release_year = 2000

        duration_str = f"{duration_ms // 60000}:{(duration_ms % 60000) // 1000:02d}"

        track_db[str(tid)] = {
            'id': tid,
            'name': name,
            'artist': artist,
            'genre': genre,
            'duration_ms': duration_ms,
            'duration_str': duration_str,
            'popularity': popularity,
            'release_year': release_year
        }
    return track_db


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', required=True, help='Input CSV/TSV/JSON mapping file')
    p.add_argument('--hetrec-dir', help='Path to hetrec2011-lastfm-2k directory to generate metadata from artists.dat')
    p.add_argument('--format', choices=['auto', 'csv', 'json'], default='auto')
    p.add_argument('--sep', default=',', help='CSV separator (default ,). Use "\\t" for TSV')
    p.add_argument('--id-col', default='id')
    p.add_argument('--name-col', default='name')
    p.add_argument('--artist-col', default='artist')
    p.add_argument('--genre-col', default='genre')
    p.add_argument('--duration-col', default='duration_ms')
    p.add_argument('--popularity-col', default='popularity')
    p.add_argument('--year-col', default='release_year')
    p.add_argument('--out', '-o', default='data/hetrec2011-lastfm/track_metadata.json')
    p.add_argument('--n-tracks', type=int, default=None, help='When generating from hetrec, number of tracks to create')
    p.add_argument('--seed', type=int, default=42, help='Random seed for generated metadata')
    args = p.parse_args()

    if args.hetrec_dir:
        # generate metadata from hetrec artists.dat
        hetrec_dir = args.hetrec_dir
        n = args.n_tracks or 256
        random.seed(args.seed)
        artists_file = os.path.join(hetrec_dir, 'artists.dat')
        artists = []
        if os.path.exists(artists_file):
            with open(artists_file, 'r', encoding='utf-8') as f:
                _ = f.readline()
                for line in f:
                    parts = line.rstrip('\n').split('\t')
                    if len(parts) >= 2:
                        name = parts[1].strip()
                        if name:
                            artists.append(name)
        if not artists:
            artists = ["Unknown Artist"]

        track_db = OrderedDict()
        for tid in range(n):
            name = random.choice(TRACK_NAMES)
            if random.random() > 0.7:
                name = f"{name} #{random.randint(1,10)}"
            artist = random.choice(artists)
            genre = random.choice(GENRES)
            duration_ms = random.randint(120000, 300000)
            popularity = random.randint(20, 100)
            release_year = random.randint(1960, 2020)
            duration_str = f"{duration_ms // 60000}:{(duration_ms % 60000) // 1000:02d}"
            track_db[str(tid)] = {
                'id': tid,
                'name': name,
                'artist': artist,
                'genre': genre,
                'duration_ms': duration_ms,
                'duration_str': duration_str,
                'popularity': popularity,
                'release_year': release_year
            }
    else:
        rows = load_table(args.input, fmt=args.format, sep=args.sep)
        track_db = to_track_db(rows, id_col=args.id_col, name_col=args.name_col, artist_col=args.artist_col, genre_col=args.genre_col, duration_col=args.duration_col, popularity_col=args.popularity_col, year_col=args.year_col)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(track_db, f, indent=2, ensure_ascii=False)

    print(f"✓ 変換完了: {args.out} ({len(track_db)} tracks)")


if __name__ == '__main__':
    main()
