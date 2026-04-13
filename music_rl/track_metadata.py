import json
import random
import os

# サンプルの曲名、アーティスト名、ジャンル
TRACK_NAMES = [
    "Midnight Dreams", "Summer Vibes", "Digital Love", "Neon Lights",
    "Lost in Time", "Electric Feel", "Cosmic Journey", "Urban Legends",
    "Silent Night", "Dancing Shadows", "Golden Hour", "Endless Roads",
    "Crystal Clear", "Velvet Sky", "Thunder Strike", "Ocean Waves",
    "Starlight", "Phoenix Rising", "Moonlight Sonata", "Wild Hearts",
    "Beyond Horizons", "Eternal Flame", "Frozen Moments", "Rainbow Bridge",
    "Midnight City", "Sunset Boulevard", "Whispered Secrets", "Broken Dreams",
    "Silver Lining", "Purple Rain", "Diamond Eyes", "Cherry Blossom",
    "Northern Lights", "Southern Cross", "Eastern Promise", "Western Wind",
    "Tokyo Nights", "Paris Dreams", "London Calling", "New York State",
    "California Love", "Texas Hold'em", "Miami Heat", "Chicago Blues",
    "Memphis Soul", "Nashville Country", "Seattle Grunge", "Detroit Rock",
    "ああ無情", "君の名は", "桜の花", "夏の思い出", "冬のソナタ"
]

ARTISTS = [
    "The Dreamers", "Neon Pulse", "Electric Storm", "Cosmic Travelers",
    "Urban Poets", "Silent Whispers", "Golden Age", "Crystal Vision",
    "Velvet Underground", "Phoenix Collective", "Moonlight Orchestra", "Wild Ones",
    "Horizon Seekers", "Eternal Band", "Frozen Echo", "Rainbow Warriors",
    "Diamond Dogs", "Silver Surfers", "Cherry Bomb", "Northern Soul",
    "Tokyo Drift", "Paris Nights", "London Grammar", "NYC Symphony",
    "サザンオールスターズ", "DREAMS COME TRUE", "Mr.Children", "B'z"
]

GENRES = [
    "Pop", "Rock", "Electronic", "Hip Hop", "Jazz", "Classical",
    "R&B", "Country", "Indie", "Alternative", "Dance", "Folk",
    "J-Pop", "K-Pop", "Soul", "Funk"
]


def generate_track_metadata_from_hetrec(hetrec_dir, n_tracks=256, seed=42, output_file='data/hetrec2011-lastfm/track_metadata.json'):
    """
    hetrec の raw ファイル（artists.dat など）を参照して簡易的な track_metadata.json を生成します。

    Args:
        hetrec_dir: hetrec 生データが置かれたディレクトリ
        n_tracks: 生成する曲数
        seed: 乱数シード
        output_file: 出力ファイルパス

    Returns:
        track_db: 生成した辞書
    """
    random.seed(seed)

    # artists.dat があればアーティスト名を抽出
    artists_file = os.path.join(hetrec_dir, 'artists.dat')
    artists_list = []
    if os.path.exists(artists_file):
        try:
            with open(artists_file, 'r', encoding='utf-8') as f:
                header = f.readline()
                for line in f:
                    parts = line.rstrip('\n').split('\t')
                    if len(parts) >= 2:
                        name = parts[1].strip()
                        if name:
                            artists_list.append(name)
        except Exception:
            artists_list = []

    if not artists_list:
        # fallback
        artists_list = ARTISTS

    track_db = {}
    for track_id in range(n_tracks):
        track_name = random.choice(TRACK_NAMES)
        # より hetrec に近づけるためアーティスト名は artists_list から選ぶ
        artist = random.choice(artists_list)
        genre = random.choice(GENRES)

        if random.random() > 0.7:
            track_name = f"{track_name} #{random.randint(1, 10)}"

        duration_ms = random.randint(120000, 300000)
        popularity = random.randint(20, 100)
        release_year = random.randint(1960, 2020)

        track_db[str(track_id)] = {
            'id': track_id,
            'name': track_name,
            'artist': artist,
            'genre': genre,
            'duration_ms': duration_ms,
            'duration_str': f"{duration_ms // 60000}:{(duration_ms % 60000) // 1000:02d}",
            'popularity': popularity,
            'release_year': release_year
        }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(track_db, f, indent=2, ensure_ascii=False)

    print(f"✓ hetrec から {n_tracks}曲のメタデータを生成: {output_file}")
    return track_db

def generate_track_database(n_tracks=256, seed=42, output_file='data/hetrec2011-lastfm/track_metadata.json'):
    """
    ダミーの曲データベースを生成
    
    Args:
        n_tracks: 生成する曲の数
        seed: 乱数シード
        output_file: 保存先ファイル
    
    Returns:
        track_db: 曲データベース（辞書）
    """
    random.seed(seed)
    
    track_db = {}
    
    for track_id in range(n_tracks):
        # ランダムに曲名とアーティストを選択
        track_name = random.choice(TRACK_NAMES)
        artist = random.choice(ARTISTS)
        genre = random.choice(GENRES)
        
        # ユニークにするためにIDを追加
        if random.random() > 0.7:  # 30%の確率で番号を追加
            track_name = f"{track_name} #{random.randint(1, 10)}"
        
        # 追加のメタデータ
        duration_ms = random.randint(120000, 300000)  # 2-5分
        popularity = random.randint(30, 100)
        release_year = random.randint(1990, 2024)
        
        track_db[str(track_id)] = {
            'id': track_id,
            'name': track_name,
            'artist': artist,
            'genre': genre,
            'duration_ms': duration_ms,
            'duration_str': f"{duration_ms // 60000}:{(duration_ms % 60000) // 1000:02d}",
            'popularity': popularity,
            'release_year': release_year
        }
    
    # ファイルに保存
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(track_db, f, indent=2, ensure_ascii=False)
    
    print(f"✓ {n_tracks}曲のメタデータを生成: {output_file}")
    return track_db


def load_track_metadata(metadata_file='data/hetrec2011-lastfm/track_metadata.json', n_tracks=None):
    """
    曲メタデータを読み込み
    
    Args:
        metadata_file: メタデータファイルのパス
    
    Returns:
        track_db: 曲データベース
    """
    if not os.path.exists(metadata_file):
        print(f"⚠ メタデータファイルが見つかりません。生成します: {metadata_file}")
        # try to generate from hetrec raw files if available
        hetrec_dir = os.path.join(os.path.dirname(metadata_file), 'hetrec2011-lastfm-2k')
        if os.path.exists(hetrec_dir):
            try:
                return generate_track_metadata_from_hetrec(hetrec_dir, n_tracks or 256, output_file=metadata_file)
            except Exception:
                # fallback to random generator
                return generate_track_database(n_tracks or 256, output_file=metadata_file)
        else:
            return generate_track_database(n_tracks or 256, output_file=metadata_file)
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        track_db = json.load(f)

    # If n_tracks is specified, trim or extend the loaded metadata to match
    if n_tracks is not None:
        try:
            existing = {int(k): v for k, v in track_db.items()}
        except Exception:
            # non-integer keys — keep as-is
            existing = {i: v for i, v in enumerate(track_db.values())}

        if len(existing) >= n_tracks:
            # trim to first n_tracks by sorted key order
            keys = sorted(existing.keys())[:n_tracks]
            new_db = {str(i): existing[k] for i, k in enumerate(keys)}
            # update id fields to be 0..n_tracks-1
            for i in range(len(new_db)):
                new_db[str(i)]['id'] = i
            # overwrite the metadata file with trimmed version
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(new_db, f, indent=2, ensure_ascii=False)
            return new_db
        else:
            # extend: keep existing entries and append generated ones
            new_db = {str(k): v for k, v in existing.items()}
            start = max(existing.keys()) + 1 if existing else 0
            need = n_tracks - len(existing)
            # generate additional entries
            gen = generate_track_database(n_tracks=need, seed=42, output_file=metadata_file + '.tmp')
            # append generated entries with new ids
            for i in range(need):
                gid = start + i
                entry = gen[str(i)].copy()
                entry['id'] = gid
                new_db[str(gid)] = entry

            # normalize keys to 0..n_tracks-1 ordering in final file
            final_db = {str(i): new_db[str(i)] for i in range(n_tracks)}
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(final_db, f, indent=2, ensure_ascii=False)
            return final_db

    return track_db


def get_track_info(track_id, track_db):
    """
    トラックIDから曲情報を取得
    
    Args:
        track_id: トラックID
        track_db: 曲データベース
    
    Returns:
        track_info: 曲情報の辞書
    """
    track_id_str = str(track_id)
    if track_id_str in track_db:
        return track_db[track_id_str]
    else:
        return {
            'id': track_id,
            'name': f'Unknown Track #{track_id}',
            'artist': 'Unknown Artist',
            'genre': 'Unknown',
            'duration_str': '0:00'
        }


if __name__ == '__main__':
    # テスト実行
    print("=" * 60)
    print("曲メタデータ生成ツール")
    print("=" * 60 + "\n")
    
    # データベース生成
    track_db = generate_track_database(n_tracks=256)
    
    # サンプル表示
    print("\n【サンプル曲情報】")
    for i in [0, 71, 100, 193, 255]:
        info = get_track_info(i, track_db)
        print(f"\nTrack #{i}:")
        print(f"  曲名: {info['name']}")
        print(f"  アーティスト: {info['artist']}")
        print(f"  ジャンル: {info['genre']}")
        print(f"  長さ: {info['duration_str']}")
        print(f"  人気度: {info['popularity']}")
        print(f"  リリース年: {info['release_year']}")