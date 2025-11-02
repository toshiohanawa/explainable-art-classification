"""
ステータスチェックユーティリティ
データ収集、画像ダウンロードなどの進捗状況を確認
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import pandas as pd


class StatusChecker:
    """ステータスチェッククラス"""
    
    def __init__(self, data_dir: str = "data"):
        """
        初期化
        
        Args:
            data_dir: データディレクトリのパス
        """
        self.data_dir = Path(data_dir)
        self.raw_data_dir = self.data_dir / "raw_data"
        self.filtered_data_dir = self.data_dir / "filtered_data"
        self.images_dir = self.filtered_data_dir / "paintings_images"
    
    def check_data_collection_status(self) -> Dict[str, Any]:
        """
        データ収集の進捗状況を確認
        
        Returns:
            進捗状況の辞書
        """
        checkpoint_file = self.raw_data_dir / 'checkpoint.json'
        
        if not checkpoint_file.exists():
            return {
                'status': 'not_started',
                'message': 'チェックポイントファイルが見つかりません'
            }
        
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint = json.load(f)
            
            processed = len(checkpoint.get('processed_ids', []))
            failed = len(checkpoint.get('failed_ids', []))
            success = checkpoint.get('api_data_count', 0)
            total = checkpoint.get('total_ids', 9005)  # デフォルト値
            
            progress_percent = (processed / total * 100) if total > 0 else 0
            success_rate = (success / (processed + failed) * 100) if (processed + failed) > 0 else 0
            
            is_complete = processed >= total
            
            return {
                'status': 'complete' if is_complete else 'in_progress',
                'processed': processed,
                'failed': failed,
                'success': success,
                'total': total,
                'remaining': total - processed,
                'progress_percent': progress_percent,
                'success_rate': success_rate,
                'is_complete': is_complete
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'チェックポイントファイルの読み込みエラー: {e}'
            }
    
    def check_download_status(self) -> Dict[str, Any]:
        """
        画像ダウンロードの進捗状況を確認
        
        Args:
            total_count: 総ダウンロード予定数（Noneの場合はデータから計算）
        
        Returns:
            ダウンロード進捗状況の辞書
        """
        if not self.images_dir.exists():
            return {
                'status': 'not_started',
                'message': '画像ディレクトリがまだ作成されていません'
            }
        
        # ダウンロード済み画像数
        downloaded_files = [f for f in self.images_dir.iterdir() if f.is_file()]
        downloaded_count = len(downloaded_files)
        
        # 統合データファイルから総数を取得
        total_count = None
        complete_dataset_file = self.filtered_data_dir / 'paintings_complete_dataset.csv'
        if complete_dataset_file.exists():
            try:
                df = pd.read_csv(complete_dataset_file, low_memory=False)
                total_count = df['primaryImageSmall'].notna().sum()
            except Exception:
                pass
        
        # 総数が取得できない場合はデフォルト値を使用
        if total_count is None:
            total_count = 5402  # デフォルト値
        
        progress_percent = (downloaded_count / total_count * 100) if total_count > 0 else 0
        remaining = total_count - downloaded_count
        
        # 予想完了時間（10 req/sで計算）
        estimated_minutes = remaining / 10 / 60 if remaining > 0 else 0
        
        # ファイルサイズの計算
        total_size = sum(f.stat().st_size for f in downloaded_files) if downloaded_files else 0
        total_size_mb = total_size / (1024 * 1024)
        
        # ファイル形式の分布
        extensions = {}
        for f in downloaded_files:
            ext = f.suffix.lower()
            extensions[ext] = extensions.get(ext, 0) + 1
        
        is_complete = downloaded_count >= total_count
        
        return {
            'status': 'complete' if is_complete else 'in_progress',
            'downloaded': downloaded_count,
            'total': total_count,
            'remaining': remaining,
            'progress_percent': progress_percent,
            'estimated_minutes': estimated_minutes,
            'total_size_mb': total_size_mb,
            'extensions': extensions,
            'is_complete': is_complete,
            'images_dir': str(self.images_dir.absolute())
        }
    
    def check_test_results(self, test_file: str = "test_paintings_200.csv") -> Dict[str, Any]:
        """
        テスト結果を確認
        
        Args:
            test_file: テスト結果ファイル名
        
        Returns:
            テスト結果の辞書
        """
        test_file_path = self.filtered_data_dir / test_file
        
        if not test_file_path.exists():
            return {
                'status': 'not_found',
                'message': f'テスト結果ファイルが見つかりません: {test_file_path}'
            }
        
        try:
            df = pd.read_csv(test_file_path, low_memory=False)
            
            # 基本統計
            total_rows = len(df)
            has_image = df['primaryImage'].notna().sum() if 'primaryImage' in df.columns else 0
            has_tags = df['tags'].notna().sum() if 'tags' in df.columns else 0
            has_measurements = df['measurements'].notna().sum() if 'measurements' in df.columns else 0
            has_constituents = df['constituents'].notna().sum() if 'constituents' in df.columns else 0
            
            # 文化圏の分布
            culture_counts = {}
            if 'Culture' in df.columns:
                culture_counts = df['Culture'].value_counts().head(10).to_dict()
            
            # Medium分布
            medium_counts = {}
            if 'Medium' in df.columns:
                medium_counts = df['Medium'].value_counts().head(10).to_dict()
            
            return {
                'status': 'success',
                'total_rows': total_rows,
                'has_image': has_image,
                'has_tags': has_tags,
                'has_measurements': has_measurements,
                'has_constituents': has_constituents,
                'culture_counts': culture_counts,
                'medium_counts': medium_counts
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': f'テスト結果ファイルの読み込みエラー: {e}'
            }
    
    def print_collection_status(self) -> None:
        """データ収集の進捗状況を表示"""
        status = self.check_data_collection_status()
        
        print("=== 全件データ収集完了確認 ===")
        
        if status['status'] == 'not_started':
            print("❌ チェックポイントファイルが見つかりません")
            return
        
        if status['status'] == 'error':
            print(f"❌ {status.get('message', 'エラーが発生しました')}")
            return
        
        print(f"処理済み: {status['processed']:,}件")
        print(f"失敗: {status['failed']:,}件")
        print(f"成功: {status['success']:,}件")
        print(f"進捗率: {status['progress_percent']:.1f}%")
        print(f"成功率: {status['success_rate']:.1f}%")
        
        if status['is_complete']:
            print("\n✅ 全件データ収集が完了しました！")
        else:
            print(f"\n⏳ 残り {status['remaining']:,}件 処理中...")
    
    def print_download_status(self, detailed: bool = False) -> None:
        """
        画像ダウンロードの進捗状況を表示
        
        Args:
            detailed: 詳細情報を表示するか
        """
        status = self.check_download_status()
        
        print("=== 画像ダウンロード進捗確認 ===")
        
        if status['status'] == 'not_started':
            print("❌ 画像ディレクトリがまだ作成されていません")
            return
        
        print(f"総数: {status['total']:,} 件")
        print(f"ダウンロード済み: {status['downloaded']:,} 件")
        print(f"残り: {status['remaining']:,} 件")
        print(f"進捗率: {status['progress_percent']:.1f}%")
        print(f"予想完了時間: {status['estimated_minutes']:.1f} 分")
        
        if detailed:
            print(f"\n総ファイルサイズ: {status['total_size_mb']:.1f} MB")
            print(f"画像保存先: {status['images_dir']}")
            
            if status['extensions']:
                print("\nファイル形式分布:")
                for ext, count in sorted(status['extensions'].items()):
                    print(f"  {ext}: {count:,} 件")
        
        if status['is_complete']:
            print("\n✅ 画像ダウンロードが完了しました！")
        else:
            print(f"\n⏳ 残り {status['remaining']:,} 件 ダウンロード中...")
    
    def print_test_results(self, test_file: str = "test_paintings_200.csv") -> None:
        """
        テスト結果を表示
        
        Args:
            test_file: テスト結果ファイル名
        """
        results = self.check_test_results(test_file)
        
        print("=== テスト結果確認 ===")
        
        if results['status'] == 'not_found':
            print(f"❌ {results.get('message', 'テスト結果ファイルが見つかりません')}")
            return
        
        if results['status'] == 'error':
            print(f"❌ {results.get('message', 'エラーが発生しました')}")
            return
        
        print(f"総行数: {results['total_rows']}")
        print(f"画像URL有り: {results['has_image']}")
        print(f"タグ有り: {results['has_tags']}")
        print(f"測定値有り: {results['has_measurements']}")
        print(f"constituents有り: {results['has_constituents']}")
        
        if results['culture_counts']:
            print("\n=== 文化圏分布 ===")
            for culture, count in results['culture_counts'].items():
                print(f"{culture}: {count}件")
        
        if results['medium_counts']:
            print("\n=== Medium分布（上位10件） ===")
            for medium, count in results['medium_counts'].items():
                print(f"{medium}: {count}件")

