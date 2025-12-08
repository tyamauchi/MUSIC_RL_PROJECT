# Music RL Project

Double DQN + Dueling DQN による音楽推薦システム

## インストール

```bash
pip install -r requirements.txt
```

## 実行

```bash
python main.py
```

## 構造

- config/ - 設定
- models/ - ニューラルネットワーク
- agents/ - DQNエージェント
- environment/ - 環境
- memory/ - リプレイバッファ
- utils/ - ユーティリティ関数

## データセット 

- data/hetrec2011-lastfm-2k（main.py USE_REAL_LASTFM_DATA=Trueにすると学習データを生成する。)