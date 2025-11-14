import pandas as pd

print("=" * 60)
print("tracks.csv のデータ構造を確認")
print("=" * 60)

try:
    # CSVファイルを読み込み
    df = pd.read_csv('data/tracks.csv')
    
    print(f"\n 基本情報:")
    print(f"  総レコード数: {len(df)}")
    print(f"  カラム名: {list(df.columns)}")
    
    print(f"\n 最初の5行:")
    print(df.head())
    
    print(f"\n 各カラムの欠損値:")
    print(df.isnull().sum())
    
    print(f"\n track_id の範囲:")
    print(f"  最小: {df['track_id'].min()}")
    print(f"  最大: {df['track_id'].max()}")
    
    # titleカラムが存在するか確認
    if 'title' in df.columns:
        print(f"\n 'title' カラムが存在します")
        print(f"  タイトルあり: {df['title'].notna().sum()} 件")
        print(f"  タイトルなし: {df['title'].isna().sum()} 件")
        
        print(f"\n タイトル例（最初の10件）:")
        for idx, row in df.head(10).iterrows():
            print(f"  ID {row['track_id']:4d}: {row['title']}")
    else:
        print(f"\n  'title' カラムが存在しません")
        print(f"  利用可能なカラム: {list(df.columns)}")
        
        # 代替カラムの候補を探す
        name_columns = [col for col in df.columns if 'name' in col.lower() or 'track' in col.lower()]
        if name_columns:
            print(f"\n💡 代替候補カラム: {name_columns}")
    
    print(f"\n データ構造の確認が完了しました")
    
except FileNotFoundError:
    print("\n エラー: data/tracks.csv が見つかりません")
    print("\n確認してください:")
    print("  1. ファイルが存在するか")
    print("  2. ファイルパスが正しいか")
    
except Exception as e:
    print(f"\n エラーが発生しました: {e}")
    print(f"  エラー詳細: {type(e).__name__}")