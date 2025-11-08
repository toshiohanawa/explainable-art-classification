"""
Phase 1からPhase 5までを全データで実行する統合スクリプト
進捗状況と実行時間を記録
"""

import sys
import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path
import subprocess
import logging

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_manager import ConfigManager
from src.utils.logger import setup_logging


class PhaseRunner:
    """Phase実行管理クラス"""
    
    def __init__(self, config_path='config.yaml', generation_model='Stable-Diffusion'):
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_config()
        self.generation_model = generation_model
        self.project_root = project_root
        self.start_time = None
        self.phase_times = {}
        self.setup_logging()
        
    def setup_logging(self):
        """ログ設定"""
        setup_logging(config=self.config)
        self.logger = logging.getLogger(__name__)
    
    def print_header(self, phase_name, phase_number):
        """Phaseヘッダー表示"""
        print("\n" + "=" * 80)
        print(f"Phase {phase_number}: {phase_name}")
        print("=" * 80)
        print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("-" * 80)
    
    def print_footer(self, phase_name, phase_number, elapsed_time):
        """Phaseフッター表示"""
        print("-" * 80)
        print(f"完了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"実行時間: {self.format_time(elapsed_time)}")
        print("=" * 80)
        self.phase_times[f"Phase {phase_number}: {phase_name}"] = elapsed_time
    
    def format_time(self, seconds):
        """時間をフォーマット"""
        delta = timedelta(seconds=seconds)
        hours, remainder = divmod(delta.seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{delta.microseconds // 10000:02d}"
    
    def run_command(self, cmd, phase_name, phase_number):
        """コマンド実行"""
        self.print_header(phase_name, phase_number)
        start = time.time()
        
        try:
            # プロジェクトルートでコマンド実行
            result = subprocess.run(
                cmd,
                cwd=str(self.project_root),
                shell=True,
                check=True,
                capture_output=False,
                text=True
            )
            elapsed = time.time() - start
            self.print_footer(phase_name, phase_number, elapsed)
            return True
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - start
            print(f"\n❌ エラーが発生しました: {e}")
            print(f"実行時間: {self.format_time(elapsed)}")
            return False
        except KeyboardInterrupt:
            elapsed = time.time() - start
            print(f"\n⚠️  ユーザーにより中断されました")
            print(f"実行時間: {self.format_time(elapsed)}")
            return False
    
    def run_phase1_2(self):
        """Phase 1-2: データローダーと特徴量抽出"""
        cmd = (
            f"python scripts/train_wikiart_vlm.py "
            f"--mode extract "
            f"--generation-model {self.generation_model} "
            f"--config config.yaml"
        )
        return self.run_command(cmd, "データローダーと特徴量抽出", "1-2")
    
    def check_phase4_completed(self):
        """Phase 4の完了状況をチェック"""
        from pathlib import Path
        import pandas as pd
        
        data_dir = self.project_root / "data" / "gestalt" / "latest"
        if not data_dir.exists():
            return False
        gestalt_file = data_dir / f'gestalt_scores_{self.generation_model}.csv'
        if gestalt_file.exists():
            try:
                df = pd.read_csv(gestalt_file)
                if len(df) >= 17000:
                    self.logger.info(f"Phase 4の既存結果が見つかりました: {gestalt_file}")
                    self.logger.info(f"  行数: {len(df):,}")
                    return True
            except Exception as e:
                self.logger.warning(f"既存ファイルの読み込みエラー: {e}")
                return False
        return False
    
    def run_phase4(self):
        """Phase 4: ゲシュタルト原則スコアリング"""
        # 既存ファイルをチェック
        if self.check_phase4_completed():
            self.logger.info("Phase 4は既に完了しています。スキップします。")
            print("\n" + "=" * 80)
            print("Phase 4: ゲシュタルト原則スコアリング")
            print("=" * 80)
            print("既存の結果が見つかりました。スキップします。")
            print("=" * 80)
            return True
        
        cmd = (
            f"python scripts/score_gestalt_principles.py "
            f"--mode wikiart "
            f"--generation-model {self.generation_model} "
            f"--config config.yaml"
        )
        return self.run_command(cmd, "ゲシュタルト原則スコアリング", "4")
    
    def run_phase3_with_gestalt(self, use_existing_features=False):
        """Phase 3: モデル訓練とSHAP（ゲシュタルトスコア含む）"""
        mode = 'train' if use_existing_features else 'all'
        cmd = (
            f"python scripts/train_wikiart_vlm.py "
            f"--mode {mode} "
            f"--generation-model {self.generation_model} "
            f"--include-gestalt "
            f"--config config.yaml"
        )
        phase_name = "モデル訓練とSHAP（ゲシュタルト統合）"
        if use_existing_features:
            phase_name += "（既存特徴量使用）"
        return self.run_command(cmd, phase_name, "3")
    
    def run_phase3_without_gestalt(self, use_existing_features=False):
        """Phase 3: モデル訓練とSHAP（ゲシュタルトスコアなし）"""
        mode = 'train' if use_existing_features else 'all'
        cmd = (
            f"python scripts/train_wikiart_vlm.py "
            f"--mode {mode} "
            f"--generation-model {self.generation_model} "
            f"--config config.yaml"
        )
        phase_name = "モデル訓練とSHAP"
        if use_existing_features:
            phase_name += "（既存特徴量使用）"
        return self.run_command(cmd, phase_name, "3")
    
    def run_phase5(self):
        """Phase 5: 分析と可視化の拡充"""
        cmd = (
            f"python scripts/analyze_authenticity.py "
            f"--mode all "
            f"--generation-model {self.generation_model} "
            f"--config config.yaml"
        )
        return self.run_command(cmd, "分析と可視化の拡充", "5")
    
    def print_summary(self, total_time, success_count, total_phases):
        """実行サマリー表示"""
        print("\n" + "=" * 80)
        print("実行サマリー")
        print("=" * 80)
        print(f"総実行時間: {self.format_time(total_time)}")
        print(f"完了Phase数: {success_count} / {total_phases}")
        print(f"成功率: {success_count / total_phases * 100:.1f}%")
        print("\n各Phaseの実行時間:")
        for phase_name, elapsed in self.phase_times.items():
            print(f"  {phase_name}: {self.format_time(elapsed)}")
        print("=" * 80)


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(
        description='Phase 1からPhase 5までを全データで実行する統合スクリプト'
    )
    parser.add_argument(
        '--generation-model',
        choices=['Stable-Diffusion', 'FLUX', 'F-Lite'],
        default='Stable-Diffusion',
        help='使用する生成モデル（デフォルト: Stable-Diffusion）'
    )
    parser.add_argument(
        '--skip-gestalt',
        action='store_true',
        help='Phase 4（ゲシュタルト原則スコアリング）をスキップする'
    )
    parser.add_argument(
        '--skip-analysis',
        action='store_true',
        help='Phase 5（分析と可視化）をスキップする'
    )
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='設定ファイルのパス（デフォルト: config.yaml）'
    )
    
    args = parser.parse_args()
    
    # PhaseRunner初期化
    runner = PhaseRunner(
        config_path=args.config,
        generation_model=args.generation_model
    )
    
    # 全体の開始時刻
    runner.start_time = time.time()
    total_phases = 0
    success_count = 0
    
    print("\n" + "=" * 80)
    print("Phase 1-5 統合実行")
    print("=" * 80)
    print(f"生成モデル: {args.generation_model}")
    print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # Phase 1-2: データローダーと特徴量抽出
    total_phases += 1
    phase1_2_success = runner.run_phase1_2()
    if phase1_2_success:
        success_count += 1
    else:
        print("\n❌ Phase 1-2 でエラーが発生しました。続行しますか？（y/n）")
        response = input().strip().lower()
        if response != 'y':
            return
    
    # Phase 4: ゲシュタルト原則スコアリング（スキップ可能）
    phase4_success = False
    if not args.skip_gestalt:
        total_phases += 1
        phase4_success = runner.run_phase4()
        if phase4_success:
            success_count += 1
        else:
            print("\n❌ Phase 4 でエラーが発生しました。Phase 3（ゲシュタルトなし）を実行します。")
    
    # Phase 3: モデル訓練とSHAP
    total_phases += 1
    if not args.skip_gestalt and phase4_success:
        # Phase 4が成功した場合、ゲシュタルトスコアを含めて訓練
        # 既存の特徴量を使用する（Phase 1-2で抽出済み）
        # ただし、統合特徴量ファイルが存在しない場合、統合を行う必要がある
        # --include-gestaltフラグで統合特徴量を作成してから訓練
        if runner.run_phase3_with_gestalt(use_existing_features=False):
            success_count += 1
        else:
            print("\n❌ Phase 3（ゲシュタルト統合）でエラーが発生しました。")
    else:
        # Phase 4をスキップしたか失敗した場合、ゲシュタルトスコアなしで訓練
        # 既存の特徴量を使用する（Phase 1-2で抽出済み）
        if runner.run_phase3_without_gestalt(use_existing_features=True):
            success_count += 1
    
    # Phase 5: 分析と可視化の拡充（スキップ可能）
    if not args.skip_analysis:
        total_phases += 1
        if runner.run_phase5():
            success_count += 1
    else:
        print("\n⚠️  Phase 5（分析と可視化）をスキップしました。")
    
    # サマリー表示
    total_time = time.time() - runner.start_time
    runner.print_summary(total_time, success_count, total_phases)
    
    print("\n✅ すべてのPhaseが完了しました！")


if __name__ == '__main__':
    main()
