# -*- coding: utf-8 -*-
"""
レート制限最適化スクリプト
HybridCollectorのパラメータを段階的に最適化し、処理速度を最大化する
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


class RateLimitOptimizer:
    """レート制限最適化クラス"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.config = self.config_manager.get_config()
        self.results = []
        
        # ログ設定
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('optimization.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def test_rate_limit(self, rate_limit: int, test_items: int = 100) -> Dict[str, Any]:
        """
        指定されたレート制限でテスト実行
        
        Args:
            rate_limit: テストするレート制限（req/s）
            test_items: テストするアイテム数
            
        Returns:
            テスト結果の辞書
        """
        self.logger.info(f"レート制限 {rate_limit} req/s でテスト開始（{test_items}件）")
        
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
        
        # テスト用のObject IDリストを生成（最初のtest_items件）
        test_ids = list(range(1, test_items + 1))
        
        start_time = time.time()
        
        try:
            # セッション初期化
            if not collector.initialize_session():
                self.logger.error("セッション初期化に失敗")
                return {
                    'rate_limit': rate_limit,
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
    
    def test_session_refresh_interval(self, rate_limit: int, refresh_interval: int, test_items: int = 100) -> Dict[str, Any]:
        """
        セッション再初期化間隔をテスト
        
        Args:
            rate_limit: レート制限
            refresh_interval: セッション再初期化間隔
            test_items: テストするアイテム数
            
        Returns:
            テスト結果の辞書
        """
        self.logger.info(f"セッション間隔 {refresh_interval}件 でテスト開始（レート: {rate_limit} req/s）")
        
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
        collector.session_refresh_interval = refresh_interval
        
        # テスト用のObject IDリストを生成
        test_ids = list(range(1, test_items + 1))
        
        start_time = time.time()
        
        try:
            # セッション初期化
            if not collector.initialize_session():
                self.logger.error("セッション初期化に失敗")
                return {
                    'rate_limit': rate_limit,
                    'refresh_interval': refresh_interval,
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
                'refresh_interval': refresh_interval,
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
                'refresh_interval': refresh_interval,
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
    
    def save_results(self, results: List[Dict[str, Any]], filename: str = 'optimization_results.csv'):
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
    
    def run_phase1_rate_optimization(self):
        """Phase 1: レート制限の最適化"""
        self.logger.info("=== Phase 1: レート制限の最適化を開始 ===")
        
        test_rates = [10, 15, 20]
        results = []
        
        for rate in test_rates:
            self.logger.info(f"レート制限 {rate} req/s をテスト中...")
            result = self.test_rate_limit(rate, test_items=100)
            results.append(result)
            
            # テスト間のクールダウン
            self.logger.info("次のテストまで30秒待機...")
            time.sleep(30)
        
        # 結果を保存
        self.save_results(results, 'phase1_rate_optimization.csv')
        
        # 最適レートを選択
        valid_results = [r for r in results if r['success_rate'] >= 0.9]
        if valid_results:
            best_result = max(valid_results, key=lambda x: x['items_per_min'])
            optimal_rate = best_result['rate_limit']
            self.logger.info(f"最適レート: {optimal_rate} req/s (処理速度: {best_result['items_per_min']:.1f}件/分)")
        else:
            self.logger.warning("成功率90%以上のレートが見つかりませんでした。最も高い成功率を選択します。")
            best_result = max(results, key=lambda x: x['success_rate'])
            optimal_rate = best_result['rate_limit']
            self.logger.info(f"最適レート: {optimal_rate} req/s (成功率: {best_result['success_rate']:.2%})")
        
        return optimal_rate, results
    
    def run_phase2_session_optimization(self, optimal_rate: int):
        """Phase 2: セッション再初期化間隔の最適化"""
        self.logger.info(f"=== Phase 2: セッション間隔の最適化を開始 (レート: {optimal_rate} req/s) ===")
        
        test_intervals = [30, 40, 60]
        results = []
        
        for interval in test_intervals:
            self.logger.info(f"セッション間隔 {interval}件 をテスト中...")
            result = self.test_session_refresh_interval(optimal_rate, interval, test_items=100)
            results.append(result)
            
            # テスト間のクールダウン
            self.logger.info("次のテストまで30秒待機...")
            time.sleep(30)
        
        # 結果を保存
        self.save_results(results, 'phase2_session_optimization.csv')
        
        # 最適間隔を選択
        valid_results = [r for r in results if r['success_rate'] >= 0.9]
        if valid_results:
            best_result = max(valid_results, key=lambda x: x['items_per_min'])
            optimal_interval = best_result['refresh_interval']
            self.logger.info(f"最適セッション間隔: {optimal_interval}件 (処理速度: {best_result['items_per_min']:.1f}件/分)")
        else:
            self.logger.warning("成功率90%以上の間隔が見つかりませんでした。最も高い成功率を選択します。")
            best_result = max(results, key=lambda x: x['success_rate'])
            optimal_interval = best_result['refresh_interval']
            self.logger.info(f"最適セッション間隔: {optimal_interval}件 (成功率: {best_result['success_rate']:.2%})")
        
        return optimal_interval, results
    
    def update_config(self, optimal_rate: int, optimal_interval: int):
        """設定ファイルを最適化されたパラメータで更新"""
        self.logger.info(f"設定を更新中: レート={optimal_rate} req/s, セッション間隔={optimal_interval}件")
        
        # config.yamlを更新
        config_file = Path('config.yaml')
        config_content = config_file.read_text(encoding='utf-8')
        
        # レート制限を更新
        config_content = config_content.replace(
            f"rate_limit: {self.config['api']['rate_limit']}",
            f"rate_limit: {optimal_rate}"
        )
        
        # ファイルに書き戻し
        config_file.write_text(config_content, encoding='utf-8')
        
        # hybrid_collector.pyのセッション間隔を更新
        collector_file = Path('src/data_collection/hybrid_collector.py')
        collector_content = collector_file.read_text(encoding='utf-8')
        
        collector_content = collector_content.replace(
            'self.session_refresh_interval = 50',
            f'self.session_refresh_interval = {optimal_interval}'
        )
        
        collector_file.write_text(collector_content, encoding='utf-8')
        
        self.logger.info("設定ファイルの更新が完了しました")
    
    def run_optimization(self):
        """最適化プロセス全体を実行"""
        self.logger.info("=== パラメータ最適化を開始 ===")
        
        try:
            # Phase 1: レート制限の最適化
            optimal_rate, rate_results = self.run_phase1_rate_optimization()
            
            # Phase 2: セッション間隔の最適化
            optimal_interval, session_results = self.run_phase2_session_optimization(optimal_rate)
            
            # 設定を更新
            self.update_config(optimal_rate, optimal_interval)
            
            # 最終結果をまとめて保存
            all_results = rate_results + session_results
            self.save_results(all_results, 'optimization_results.csv')
            
            self.logger.info("=== 最適化完了 ===")
            self.logger.info(f"最適パラメータ: レート={optimal_rate} req/s, セッション間隔={optimal_interval}件")
            
            return optimal_rate, optimal_interval, all_results
            
        except Exception as e:
            self.logger.error(f"最適化中にエラーが発生しました: {e}")
            raise


def main():
    """メイン関数"""
    optimizer = RateLimitOptimizer()
    
    try:
        optimal_rate, optimal_interval, results = optimizer.run_optimization()
        
        print(f"\n=== 最適化結果 ===")
        print(f"最適レート制限: {optimal_rate} req/s")
        print(f"最適セッション間隔: {optimal_interval}件")
        print(f"詳細結果は optimization_results.csv に保存されました")
        
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
