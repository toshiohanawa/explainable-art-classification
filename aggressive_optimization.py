# -*- coding: utf-8 -*-
"""
攻撃的パラメータ最適化スクリプト
より高いレート制限と短い休憩時間で最大性能を追求
"""

import time
import csv
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any

from src.data_collection.hybrid_collector import HybridCollector
from src.utils.config_manager import ConfigManager


class AggressiveOptimizer:
    """攻撃的パラメータ最適化クラス"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.config = self.config_manager.get_config()
        self.results = []
        
        # ログ設定
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('aggressive_optimization.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def test_aggressive_parameters(self, rate_limit: int, batch_break_min: float, batch_break_max: float, 
                                 session_refresh: int, test_items: int = 200) -> Dict[str, Any]:
        """
        攻撃的パラメータでテスト実行
        
        Args:
            rate_limit: レート制限（req/s）
            batch_break_min: バッチ休憩最小時間（秒）
            batch_break_max: バッチ休憩最大時間（秒）
            session_refresh: セッション再初期化間隔（件）
            test_items: テストするアイテム数
            
        Returns:
            テスト結果の辞書
        """
        self.logger.info(f"攻撃的テスト: レート={rate_limit} req/s, 休憩={batch_break_min}-{batch_break_max}s, セッション間隔={session_refresh}件")
        
        # 設定を一時的に変更
        original_rate = self.config['api']['rate_limit']
        self.config['api']['rate_limit'] = rate_limit
        
        # テスト用のチェックポイントファイルを削除
        checkpoint_file = Path('data/raw_data/checkpoint.json')
        if checkpoint_file.exists():
            checkpoint_file.unlink()
        
        # HybridCollectorを初期化
        collector = HybridCollector(self.config)
        collector.csv_url = "https://github.com/metmuseum/openaccess/raw/master/MetObjects.csv"
        collector.session_refresh_interval = session_refresh
        
        # バッチ休憩時間を動的に変更
        collector.batch_break_range = (batch_break_min, batch_break_max)
        
        # テスト用のObject IDリストを生成
        test_ids = list(range(1, test_items + 1))
        
        start_time = time.time()
        
        try:
            # セッション初期化
            if not collector.initialize_session():
                self.logger.error("セッション初期化に失敗")
                return {
                    'rate_limit': rate_limit,
                    'batch_break_min': batch_break_min,
                    'batch_break_max': batch_break_max,
                    'session_refresh': session_refresh,
                    'success_rate': 0.0,
                    'items_per_min': 0.0,
                    'total_time': 0.0,
                    'success_count': 0,
                    'failed_count': test_items,
                    'error': 'Session initialization failed'
                }
            
            # APIデータ収集（テスト用）
            collector.collect_api_data(test_ids, resume=False)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # 結果を計算
            success_count = len(collector.api_data)
            failed_count = len(collector.failed_ids)
            success_rate = success_count / test_items if test_items > 0 else 0.0
            items_per_min = (success_count / total_time) * 60 if total_time > 0 else 0.0
            
            result = {
                'rate_limit': rate_limit,
                'batch_break_min': batch_break_min,
                'batch_break_max': batch_break_max,
                'session_refresh': session_refresh,
                'success_rate': success_rate,
                'items_per_min': items_per_min,
                'total_time': total_time,
                'success_count': success_count,
                'failed_count': failed_count,
                'error': None
            }
            
            self.logger.info(f"テスト完了: 成功率 {success_rate:.2%}, 処理速度 {items_per_min:.1f}件/分")
            
        except Exception as e:
            import traceback
            self.logger.error(f"テスト中にエラー: {e}")
            self.logger.error(f"詳細: {traceback.format_exc()}")
            result = {
                'rate_limit': rate_limit,
                'batch_break_min': batch_break_min,
                'batch_break_max': batch_break_max,
                'session_refresh': session_refresh,
                'success_rate': 0.0,
                'items_per_min': 0.0,
                'total_time': 0.0,
                'success_count': 0,
                'failed_count': test_items,
                'error': str(e)
            }
        
        finally:
            # 設定を元に戻す
            self.config['api']['rate_limit'] = original_rate
        
        return result
    
    def run_aggressive_optimization(self):
        """攻撃的最適化を実行"""
        self.logger.info("=== 攻撃的パラメータ最適化を開始 ===")
        
        # より攻撃的なパラメータセット
        test_combinations = [
            # (rate_limit, batch_break_min, batch_break_max, session_refresh)
            (30, 1.0, 2.0, 100),  # 高レート、短休憩、長間隔
            (40, 0.5, 1.5, 150),  # 超高レート、超短休憩、超長間隔
            (50, 0.3, 1.0, 200),  # 極高レート、極短休憩、極長間隔
            (25, 1.5, 3.0, 80),   # 中高レート、中休憩、中間隔
            (35, 0.8, 2.5, 120),  # 高レート、短休憩、長間隔
        ]
        
        results = []
        
        for i, (rate, break_min, break_max, refresh) in enumerate(test_combinations):
            self.logger.info(f"=== テスト {i+1}/{len(test_combinations)} ===")
            result = self.test_aggressive_parameters(rate, break_min, break_max, refresh, test_items=200)
            results.append(result)
            
            # テスト間のクールダウン（短縮）
            if i < len(test_combinations) - 1:
                self.logger.info("次のテストまで15秒待機...")
                time.sleep(15)
        
        # 結果を保存
        self.save_results(results, 'aggressive_optimization_results.csv')
        
        # 最適パラメータを選択
        valid_results = [r for r in results if r['success_rate'] >= 0.85]  # 成功率85%以上
        if valid_results:
            best_result = max(valid_results, key=lambda x: x['items_per_min'])
            self.logger.info(f"最適パラメータ: レート={best_result['rate_limit']} req/s, "
                           f"休憩={best_result['batch_break_min']}-{best_result['batch_break_max']}s, "
                           f"セッション間隔={best_result['session_refresh']}件")
            self.logger.info(f"処理速度: {best_result['items_per_min']:.1f}件/分, 成功率: {best_result['success_rate']:.2%}")
        else:
            self.logger.warning("成功率85%以上のパラメータが見つかりませんでした。最も高い成功率を選択します。")
            best_result = max(results, key=lambda x: x['success_rate'])
            self.logger.info(f"最適パラメータ: レート={best_result['rate_limit']} req/s, "
                           f"成功率: {best_result['success_rate']:.2%}")
        
        return best_result, results
    
    def save_results(self, results: List[Dict[str, Any]], filename: str):
        """結果をCSVファイルに保存"""
        if not results:
            return
        
        output_file = Path(filename)
        
        # フィールド名を取得
        fieldnames = list(results[0].keys())
        
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        self.logger.info(f"結果を保存しました: {output_file}")
    
    def update_config_aggressive(self, best_result: Dict[str, Any]):
        """設定ファイルを攻撃的パラメータで更新"""
        self.logger.info("攻撃的パラメータで設定を更新中...")
        
        # config.yamlを更新
        config_file = Path('config.yaml')
        config_content = config_file.read_text(encoding='utf-8')
        
        # レート制限を更新
        config_content = config_content.replace(
            f"rate_limit: {self.config['api']['rate_limit']}",
            f"rate_limit: {best_result['rate_limit']}"
        )
        
        # ファイルに書き戻し
        config_file.write_text(config_content, encoding='utf-8')
        
        # hybrid_collector.pyを更新
        collector_file = Path('src/data_collection/hybrid_collector.py')
        collector_content = collector_file.read_text(encoding='utf-8')
        
        # セッション間隔を更新
        collector_content = collector_content.replace(
            'self.session_refresh_interval = 60',
            f'self.session_refresh_interval = {best_result["session_refresh"]}'
        )
        
        # バッチ休憩時間を更新
        collector_content = collector_content.replace(
            'batch_break = random.uniform(3, 5)',
            f'batch_break = random.uniform({best_result["batch_break_min"]}, {best_result["batch_break_max"]})'
        )
        
        collector_file.write_text(collector_content, encoding='utf-8')
        
        self.logger.info("攻撃的設定ファイルの更新が完了しました")


def main():
    """メイン関数"""
    optimizer = AggressiveOptimizer()
    
    try:
        best_result, results = optimizer.run_aggressive_optimization()
        
        print(f"\n=== 攻撃的最適化結果 ===")
        print(f"最適レート制限: {best_result['rate_limit']} req/s")
        print(f"最適バッチ休憩: {best_result['batch_break_min']}-{best_result['batch_break_max']}秒")
        print(f"最適セッション間隔: {best_result['session_refresh']}件")
        print(f"処理速度: {best_result['items_per_min']:.1f}件/分")
        print(f"成功率: {best_result['success_rate']:.2%}")
        print(f"詳細結果は aggressive_optimization_results.csv に保存されました")
        
        # 設定を更新
        optimizer.update_config_aggressive(best_result)
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
