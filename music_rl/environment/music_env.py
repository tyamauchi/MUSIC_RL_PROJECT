"""
修正版: environment/music_env.py
Hidden stateを適切に管理するように修正
"""

import numpy as np
import torch
from config.config import DEVICE


class MusicEnvironment:
    """
    音楽推薦環境
    
    修正点:
    - LSTMのhidden stateを保存・管理
    - リセット時にhidden stateを初期化
    - 各ステップでhidden stateを引き継ぐ
    """
    
    def __init__(self, user_simulator, track_pool_size=256, session_length=20, state_dim=40):
        self.user_simulator = user_simulator
        self.track_pool_size = track_pool_size
        self.session_length = session_length
        self.state_dim = state_dim
        self.current_step = 0
        self.track_history = []
        self.response_history = []
        self.hidden = None  # ← 追加: hidden stateを保存
        
    def reset(self):
        """環境をリセット"""
        self.current_step = 0
        self.track_history = [0.0] * (self.state_dim // 2)
        self.response_history = [0.0] * (self.state_dim // 2)
        self.hidden = None  # ← 追加: hidden stateをリセット
        return self._get_state()

    def step(self, action):
        """
        1ステップ実行
        
        修正点:
        - user_simulatorにhidden stateを渡す
        - 返されたhidden stateを保存
        """
        state = self._get_state()
        state_tensor = torch.FloatTensor([state + [float(action)]]).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            # 修正: hidden stateを渡して、返されたhidden stateを保存
            response_prob, self.hidden = self.user_simulator(
                state_tensor, 
                self.hidden  # ← 修正: Noneまたは前回のhidden stateを渡す
            )
        response = float(response_prob.squeeze())
        
        # track_historyとresponse_historyを更新
        self.track_history[self.current_step] = float(action)
        self.response_history[self.current_step] = response
        self.current_step += 1
        
        # 報酬計算
        base_reward = response
        
        # 重複ペナルティ
        if float(action) in self.track_history[:self.current_step-1]:
            base_reward *= 0.2
        
        # セッション完了時のボーナス
        if self.current_step >= self.session_length:
            responses = np.array(self.response_history)
            avg_response = np.mean(responses)
            
            if avg_response > 0.9:
                base_reward += 2.0
            elif avg_response > 0.8:
                base_reward += 1.0
        
        done = self.current_step >= self.session_length
        return self._get_state(), base_reward, done

    def _get_state(self):
        """現在の状態を取得"""
        return self.track_history + self.response_history


# ===================================================================
# 使用例とテストコード
# ===================================================================

def test_hidden_state_management():
    """Hidden state管理のテスト"""
    print("="*60)
    print("Hidden State管理のテスト")
    print("="*60)
    
    from simulators.user_simulator import LSTMUserSimulator
    from config.config import STATE_DIM
    
    # LSTMシミュレータの作成
    lstm_sim = LSTMUserSimulator(STATE_DIM + 1).to(DEVICE)
    lstm_sim.eval()
    
    # 環境の作成
    env = MusicEnvironment(lstm_sim, track_pool_size=256, session_length=5, state_dim=STATE_DIM)
    
    print("\n[テスト1] 環境リセット後、hidden stateはNone")
    state = env.reset()
    print(f"  hidden is None: {env.hidden is None}")  # True
    
    print("\n[テスト2] 1ステップ実行後、hidden stateが設定される")
    next_state, reward, done = env.step(action=10)
    print(f"  hidden is None: {env.hidden is None}")  # False
    print(f"  hidden type: {type(env.hidden)}")
    print(f"  Response: {env.response_history[0]:.4f}")
    
    print("\n[テスト3] 2ステップ目ではhidden stateが引き継がれる")
    prev_hidden = env.hidden
    next_state, reward, done = env.step(action=20)
    curr_hidden = env.hidden
    print(f"  hidden changed: {prev_hidden != curr_hidden}")  # True (更新される)
    print(f"  Response: {env.response_history[1]:.4f}")
    
    print("\n[テスト4] リセット後、再びhidden stateがNoneになる")
    state = env.reset()
    print(f"  hidden is None: {env.hidden is None}")  # True
    
    print("\n[テスト5] 複数エピソードでの挙動確認")
    responses_ep1 = []
    responses_ep2 = []
    
    # エピソード1
    env.reset()
    for step in range(5):
        _, _, _ = env.step(action=step*10)
        responses_ep1.append(env.response_history[step])
    
    # エピソード2（同じアクション）
    env.reset()
    for step in range(5):
        _, _, _ = env.step(action=step*10)
        responses_ep2.append(env.response_history[step])
    
    print(f"  Episode 1 responses: {[f'{r:.4f}' for r in responses_ep1]}")
    print(f"  Episode 2 responses: {[f'{r:.4f}' for r in responses_ep2]}")
    print(f"  Responses identical: {np.allclose(responses_ep1, responses_ep2)}")
    
    print("\n"+"="*60)
    print("テスト完了")
    print("="*60)


def compare_old_vs_new_environment():
    """旧環境と新環境の比較"""
    print("\n"+"="*60)
    print("旧環境 vs 新環境の比較")
    print("="*60)
    
    # このテストを実行するには、旧環境のコードも必要
    # 実装は省略
    pass


if __name__ == "__main__":
    test_hidden_state_management()
    # compare_old_vs_new_environment()