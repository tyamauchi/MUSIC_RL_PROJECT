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

`main.py` の `AGENT_TYPE` で切替: `"dqn"` / `"ppo"` / `"graph_dqn"` / `"hetero_graph_dqn"`（異種グラフ HGT、比較用）。

## 構造

- config/ - 設定
- models/ - ニューラルネットワーク
- agents/ - DQN / PPO / Graph DQN / **Hetero Graph DQN** エージェント
- environment/ - 環境
- memory/ - リプレイバッファ
- utils/ - ユーティリティ関数

## データセット 

- data/hetrec2011-lastfm-2k（main.py USE_REAL_LASTFM_DATA=Trueにすると学習データを生成する。)