"""
生成モデル別の性能比較スクリプト
Stable-Diffusion, FLUX, F-Liteの各モデルでの性能を比較
"""

import sys
import argparse
import pandas as pd
from pathlib import Path
import logging

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_collection.wikiart_vlm_loader import WikiArtVLMDataLoader
from src.feature_extraction.color_extractor import ColorFeatureExtractor
from src.model_training.random_forest_trainer import RandomForestTrainer
from src.explainability.shap_explainer import SHAPExplainer
from src.utils.config_manager import ConfigManager
from src.utils.timestamp_manager import TimestampManager
from src.utils.logger import setup_logging


def train_and_evaluate_model(generation_model: str, config: dict, max_samples: int = None, 
                           timestamp_manager: TimestampManager = None) -> dict:
    """
    指定された生成モデルでモデルを訓練し、評価結果を返す
    
    Args:
        generation_model: 生成モデル名 ('Stable-Diffusion', 'FLUX', 'F-Lite')
        config: 設定辞書
        max_samples: 最大サンプル数（テスト用）
        timestamp_manager: タイムスタンプ管理オブジェクト（Noneの場合は新規作成）
                         段階的実行で同じタイムスタンプを使用する場合に指定
    
    Returns:
        評価結果の辞書
    """
    logger = logging.getLogger(__name__)
    logger.info(f"\n{'='*60}")
    logger.info(f"生成モデル: {generation_model}")
    logger.info(f"{'='*60}")
    
    results = {
        'generation_model': generation_model,
        'status': 'success',
        'error': None
    }
    
    try:
        # 1. 特徴量抽出
        logger.info("\n[1/3] 特徴量抽出を開始...")
        # ✅ 同じTimestampManagerを使用（段階的実行対応）
        feature_extractor = ColorFeatureExtractor(config, timestamp_manager=timestamp_manager)
        feature_extractor.extract_features_from_wikiart_vlm(
            generation_model=generation_model,
            max_samples=max_samples
        )
        
        # 特徴量ファイルのパスを取得
        features_dir = feature_extractor.timestamp_manager.get_features_dir()
        features_file = features_dir / f"wikiart_vlm_features_{generation_model}.csv"
        
        if not features_file.exists():
            raise FileNotFoundError(f"特徴量ファイルが見つかりません: {features_file}")
        
        logger.info(f"特徴量抽出完了: {features_file}")
        
        # 2. モデル訓練
        logger.info("\n[2/3] モデル訓練を開始...")
        # ✅ 同じTimestampManagerを使用（段階的実行対応）
        trainer = RandomForestTrainer(
            config,
            task_type='authenticity',  # 本物 vs フェイク分類
            features_file=str(features_file),
            timestamp_manager=timestamp_manager
        )
        trainer.train_model()
        
        logger.info("モデル訓練完了")
        
        # 3. モデル評価（訓練済みモデルから結果を抽出）
        logger.info("\n[3/3] モデル評価を取得...")
        
        # 訓練結果ファイルから評価結果を読み込み
        results_dir = trainer.timestamp_manager.get_results_dir()
        results_file = trainer.timestamp_manager.get_results_file()
        
        if results_file.exists():
            with open(results_file, 'r', encoding='utf-8') as f:
                results_content = f.read()
            
            # 簡単なパース（実際の結果ファイルの形式に応じて調整）
            import re
            train_acc_match = re.search(r'訓練精度: ([\d.]+)', results_content)
            test_acc_match = re.search(r'テスト精度: ([\d.]+)', results_content)
            cv_match = re.search(r'交差検証スコア: ([\d.]+)', results_content)
            
            if train_acc_match:
                results['train_accuracy'] = float(train_acc_match.group(1))
            if test_acc_match:
                results['test_accuracy'] = float(test_acc_match.group(1))
            if cv_match:
                results['cv_score'] = float(cv_match.group(1))
        
        # データセット統計
        loader = WikiArtVLMDataLoader(config)
        stats = loader.get_dataset_stats(generation_model)
        results['total_images'] = stats['total_images']
        results['original_count'] = stats['original_count']
        results['generated_count'] = stats['generated_count']
        
        logger.info(f"評価完了: テスト精度={results.get('test_accuracy', 'N/A')}")
        
    except Exception as e:
        logger.error(f"エラーが発生しました ({generation_model}): {e}", exc_info=True)
        results['status'] = 'error'
        results['error'] = str(e)
    
    return results


def compare_models(generation_models: list, config: dict, max_samples: int = None,
                   timestamp_manager: TimestampManager = None) -> pd.DataFrame:
    """
    複数の生成モデルで訓練を行い、結果を比較
    
    Args:
        generation_models: 生成モデル名のリスト
        config: 設定辞書
        max_samples: 最大サンプル数（テスト用）
        timestamp_manager: タイムスタンプ管理オブジェクト（Noneの場合は新規作成）
                         段階的実行で同じタイムスタンプを使用する場合に指定
    
    Returns:
        比較結果のDataFrame
    """
    logger = logging.getLogger(__name__)
    
    all_results = []
    
    for model in generation_models:
        results = train_and_evaluate_model(model, config, max_samples, timestamp_manager)
        all_results.append(results)
    
    # 結果をDataFrameに変換
    comparison_df = pd.DataFrame(all_results)
    
    return comparison_df


def main():
    """メイン実行関数"""
    parser = argparse.ArgumentParser(
        description='生成モデル別の性能比較スクリプト'
    )
    parser.add_argument(
        '--generation-models',
        nargs='+',
        choices=['Stable-Diffusion', 'FLUX', 'F-Lite'],
        default=['Stable-Diffusion'],
        help='比較する生成モデル（複数指定可能）'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='特徴量抽出時の最大サンプル数（テスト用、Noneの場合は全データ）'
    )
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='設定ファイルのパス'
    )
    parser.add_argument(
        '--output',
        default=None,
        help='比較結果の出力CSVファイルパス（Noneの場合は自動生成）'
    )
    parser.add_argument(
        '--include-shap',
        action='store_true',
        help='SHAP説明も生成する（時間がかかります）'
    )
    
    args = parser.parse_args()
    
    # 設定読み込み
    config_manager = ConfigManager(args.config)
    config = config_manager.get_config()
    
    # ログ設定
    setup_logging(config=config)
    logger = logging.getLogger(__name__)
    
    # ✅ スクリプトの最初で一度だけTimestampManagerを作成（段階的実行対応）
    timestamp_manager = TimestampManager(config)
    timestamp = timestamp_manager.get_timestamp()
    
    logger.info("=" * 60)
    logger.info("生成モデル別の性能比較")
    logger.info("=" * 60)
    logger.info(f"タイムスタンプ: {timestamp}")
    logger.info(f"出力ディレクトリ: {timestamp_manager.get_output_dir()}")
    logger.info(f"比較モデル: {', '.join(args.generation_models)}")
    if args.max_samples:
        logger.info(f"最大サンプル数: {args.max_samples} (テスト用)")
    
    try:
        # モデル比較
        comparison_df = compare_models(
            args.generation_models,
            config,
            max_samples=args.max_samples,
            timestamp_manager=timestamp_manager
        )
        
        # 結果を表示
        logger.info("\n" + "=" * 60)
        logger.info("比較結果")
        logger.info("=" * 60)
        print("\n" + comparison_df.to_string(index=False))
        
        # CSVに保存
        if args.output:
            output_file = Path(args.output)
        else:
            # ✅ 同じTimestampManagerを使用（段階的実行対応）
            output_file = timestamp_manager.get_results_dir() / 'generation_models_comparison.csv'
        
        comparison_df.to_csv(output_file, index=False)
        logger.info(f"\n比較結果を保存: {output_file}")
        
        # SHAP説明を生成（オプション）
        if args.include_shap:
            logger.info("\n" + "=" * 60)
            logger.info("SHAP説明を生成中...")
            logger.info("=" * 60)
            
            for model in args.generation_models:
                try:
                    logger.info(f"\n{model}のSHAP説明を生成...")
                    # ✅ 同じTimestampManagerを使用（段階的実行対応）
                    explainer = SHAPExplainer(config, task_type='authenticity',
                                            timestamp_manager=timestamp_manager)
                    explainer.generate_explanations()
                    logger.info(f"{model}のSHAP説明完了")
                except Exception as e:
                    logger.warning(f"{model}のSHAP説明生成でエラー: {e}")
        
        logger.info("\n" + "=" * 60)
        logger.info("すべての処理が完了しました")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"エラーが発生しました: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

