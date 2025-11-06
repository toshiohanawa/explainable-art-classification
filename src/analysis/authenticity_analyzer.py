"""
真贋分析クラス
誤分類作品の詳細分析、生成モデル間の比較、教育的価値の可視化を行う
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import json

from ..utils.config_manager import ConfigManager
from ..utils.timestamp_manager import TimestampManager


class AuthenticityAnalyzer:
    """真贋分析クラス"""
    
    def __init__(self, config: Dict[str, Any], timestamp_manager: Optional[TimestampManager] = None):
        """
        初期化
        
        Args:
            config: 設定辞書
            timestamp_manager: タイムスタンプ管理オブジェクト（Noneの場合は新規作成）
        """
        from ..utils.timestamp_manager import TimestampManager
        
        self.config = config
        self.data_config = config['data']
        self.viz_config = config['visualization']
        
        self.logger = logging.getLogger(__name__)
        
        # タイムスタンプ管理（共有または新規作成）
        if timestamp_manager is not None:
            self.timestamp_manager = timestamp_manager
        else:
            self.timestamp_manager = TimestampManager(config)
        
        self.output_dir = self.timestamp_manager.get_output_dir()
        self.results_dir = self.timestamp_manager.get_results_dir()
        self.viz_dir = self.timestamp_manager.get_visualizations_dir()
        self.features_dir = self.timestamp_manager.get_features_dir()
        
        # ディレクトリ作成
        self.timestamp_manager.create_directories()
        
        # スタイル設定
        plt.style.use('seaborn-v0_8')
        sns.set_palette(self.viz_config.get('color_palette', 'husl'))
    
    def analyze_misclassifications(self, generation_model: str = 'Stable-Diffusion') -> pd.DataFrame:
        """
        誤分類作品の詳細分析
        
        Args:
            generation_model: 生成モデル名
            
        Returns:
            誤分類分析結果のDataFrame
        """
        self.logger.info(f"誤分類作品の詳細分析を開始 (モデル: {generation_model})")
        
        # 特徴量ファイルを読み込み
        features_file = self.features_dir / f"wikiart_vlm_features_with_gestalt_{generation_model}.csv"
        if not features_file.exists():
            features_file = self.features_dir / f"wikiart_vlm_features_{generation_model}.csv"
        
        if not features_file.exists():
            raise FileNotFoundError(f"特徴量ファイルが見つかりません: {features_file}")
        
        features_df = pd.read_csv(features_file)
        self.logger.info(f"特徴量データ読み込み: {len(features_df)}件")
        
        # 誤分類ファイルを読み込み
        misclass_file = self.results_dir / 'misclassifications.csv'
        if not misclass_file.exists():
            self.logger.warning("誤分類ファイルが見つかりません。先にモデル訓練を実行してください。")
            return pd.DataFrame()
        
        misclass_df = pd.read_csv(misclass_file)
        self.logger.info(f"誤分類データ読み込み: {len(misclass_df)}件")
        
        # 誤分類作品の詳細情報を取得
        misclass_details = []
        
        for _, row in misclass_df.iterrows():
            test_idx = row.get('test_index', -1)
            actual_label = row.get('actual_label', 'Unknown')
            predicted_label = row.get('predicted_label', 'Unknown')
            
            # 特徴量データから詳細情報を取得
            if 'image_id' in misclass_df.columns and 'image_id' in features_df.columns:
                image_id = row.get('image_id', '')
                # image_idを文字列として比較（型の違いに対応）
                feature_row = features_df[features_df['image_id'].astype(str) == str(image_id)]
                if len(feature_row) == 0:
                    # 数値としても試す
                    try:
                        if pd.api.types.is_numeric_dtype(features_df['image_id']):
                            feature_row = features_df[features_df['image_id'].astype(int) == int(image_id)]
                    except (ValueError, TypeError):
                        pass
                
                if len(feature_row) > 0:
                    feature_row = feature_row.iloc[0]
                    
                    detail = {
                        'image_id': image_id,
                        'actual_label': actual_label,
                        'predicted_label': predicted_label,
                        'predicted_probability': row.get('predicted_probability', 0.0),
                        'source': feature_row.get('source', 'Unknown'),
                        'generation_model': feature_row.get('generation_model', generation_model),
                    }
                    
                    # 色彩特徴量
                    color_features = ['mean_hue', 'mean_saturation', 'mean_value', 'contrast', 'color_diversity']
                    for feat in color_features:
                        if feat in feature_row.index:
                            detail[f'color_{feat}'] = feature_row[feat]
                    
                    # ゲシュタルト原則スコア
                    gestalt_features = ['simplicity_score', 'proximity_score', 'similarity_score',
                                       'continuity_score', 'closure_score', 'figure_ground_score']
                    for feat in gestalt_features:
                        if feat in feature_row.index:
                            detail[f'gestalt_{feat}'] = feature_row[feat]
                    
                    misclass_details.append(detail)
        
        if len(misclass_details) == 0:
            self.logger.warning("誤分類作品の詳細情報が見つかりませんでした")
            return pd.DataFrame()
        
        analysis_df = pd.DataFrame(misclass_details)
        
        # 誤分類タイプ別の分析
        self._analyze_false_positives(analysis_df, features_df)
        self._analyze_false_negatives(analysis_df, features_df)
        
        # 可視化
        self._visualize_misclassification_details(analysis_df)
        
        # 結果を保存
        analysis_file = self.results_dir / f'misclassification_detailed_analysis_{generation_model}.csv'
        analysis_df.to_csv(analysis_file, index=False)
        self.logger.info(f"詳細分析結果を保存: {analysis_file}")
        
        return analysis_df
    
    def _analyze_false_positives(self, analysis_df: pd.DataFrame, features_df: pd.DataFrame) -> None:
        """
        偽陽性（本物をフェイクと誤分類）の分析
        
        Args:
            analysis_df: 誤分類分析結果
            features_df: 特徴量データ
        """
        fp_df = analysis_df[analysis_df['actual_label'] == 'Authentic']
        
        if len(fp_df) == 0:
            self.logger.info("偽陽性（本物をフェイクと誤分類）: 0件")
            return
        
        self.logger.info(f"\n偽陽性（本物をフェイクと誤分類）: {len(fp_df)}件")
        
        # 色彩特徴量の分析
        color_cols = [col for col in fp_df.columns if col.startswith('color_')]
        if color_cols:
            fp_color_stats = fp_df[color_cols].describe()
            self.logger.info("色彩特徴量の統計:")
            self.logger.info(f"\n{fp_color_stats}")
        
        # ゲシュタルト原則スコアの分析
        gestalt_cols = [col for col in fp_df.columns if col.startswith('gestalt_')]
        if gestalt_cols:
            fp_gestalt_stats = fp_df[gestalt_cols].describe()
            self.logger.info("ゲシュタルト原則スコアの統計:")
            self.logger.info(f"\n{fp_gestalt_stats}")
    
    def _analyze_false_negatives(self, analysis_df: pd.DataFrame, features_df: pd.DataFrame) -> None:
        """
        偽陰性（フェイクを本物と誤分類）の分析
        
        Args:
            analysis_df: 誤分類分析結果
            features_df: 特徴量データ
        """
        fn_df = analysis_df[analysis_df['actual_label'] == 'Fake']
        
        if len(fn_df) == 0:
            self.logger.info("偽陰性（フェイクを本物と誤分類）: 0件")
            return
        
        self.logger.info(f"\n偽陰性（フェイクを本物と誤分類）: {len(fn_df)}件")
        
        # 生成モデル別の分析
        if 'generation_model' in fn_df.columns:
            model_counts = fn_df['generation_model'].value_counts()
            self.logger.info("生成モデル別の誤分類数:")
            for model, count in model_counts.items():
                self.logger.info(f"  - {model}: {count}件")
        
        # 色彩特徴量の分析
        color_cols = [col for col in fn_df.columns if col.startswith('color_')]
        if color_cols:
            fn_color_stats = fn_df[color_cols].describe()
            self.logger.info("色彩特徴量の統計:")
            self.logger.info(f"\n{fn_color_stats}")
        
        # ゲシュタルト原則スコアの分析
        gestalt_cols = [col for col in fn_df.columns if col.startswith('gestalt_')]
        if gestalt_cols:
            fn_gestalt_stats = fn_df[gestalt_cols].describe()
            self.logger.info("ゲシュタルト原則スコアの統計:")
            self.logger.info(f"\n{fn_gestalt_stats}")
    
    def _visualize_misclassification_details(self, analysis_df: pd.DataFrame) -> None:
        """
        誤分類詳細の可視化
        
        Args:
            analysis_df: 誤分類分析結果
        """
        if len(analysis_df) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. 誤分類タイプ別の特徴量比較（色彩）
        ax1 = axes[0, 0]
        color_features = ['color_mean_hue', 'color_mean_saturation', 'color_mean_value']
        available_color_features = [f for f in color_features if f in analysis_df.columns]
        
        if available_color_features:
            fp_data = analysis_df[analysis_df['actual_label'] == 'Authentic'][available_color_features].mean()
            fn_data = analysis_df[analysis_df['actual_label'] == 'Fake'][available_color_features].mean()
            
            x = np.arange(len(available_color_features))
            width = 0.35
            
            ax1.bar(x - width/2, fp_data.values, width, label='False Positive (Authentic→Fake)', color='#ff6b6b')
            ax1.bar(x + width/2, fn_data.values, width, label='False Negative (Fake→Authentic)', color='#4ecdc4')
            
            ax1.set_xlabel('Color Features', fontsize=12)
            ax1.set_ylabel('Mean Value', fontsize=12)
            ax1.set_title('Average Color Features by Misclassification Type', fontsize=14)
            ax1.set_xticks(x)
            ax1.set_xticklabels([f.replace('color_', '') for f in available_color_features], rotation=45, ha='right')
            ax1.legend()
            ax1.grid(axis='y', alpha=0.3)
        
        # 2. 誤分類タイプ別のゲシュタルト原則スコア比較
        ax2 = axes[0, 1]
        gestalt_features = ['gestalt_simplicity_score', 'gestalt_proximity_score', 'gestalt_similarity_score',
                           'gestalt_continuity_score', 'gestalt_closure_score', 'gestalt_figure_ground_score']
        available_gestalt_features = [f for f in gestalt_features if f in analysis_df.columns]
        
        if available_gestalt_features:
            fp_gestalt = analysis_df[analysis_df['actual_label'] == 'Authentic'][available_gestalt_features].mean()
            fn_gestalt = analysis_df[analysis_df['actual_label'] == 'Fake'][available_gestalt_features].mean()
            
            x = np.arange(len(available_gestalt_features))
            width = 0.35
            
            ax2.bar(x - width/2, fp_gestalt.values, width, label='False Positive (Authentic→Fake)', color='#ff6b6b')
            ax2.bar(x + width/2, fn_gestalt.values, width, label='False Negative (Fake→Authentic)', color='#4ecdc4')
            
            ax2.set_xlabel('Gestalt Principles', fontsize=12)
            ax2.set_ylabel('Mean Score', fontsize=12)
            ax2.set_title('Average Gestalt Scores by Misclassification Type', fontsize=14)
            ax2.set_xticks(x)
            ax2.set_xticklabels([f.replace('gestalt_', '').replace('_score', '') for f in available_gestalt_features],
                              rotation=45, ha='right')
            ax2.legend()
            ax2.grid(axis='y', alpha=0.3)
        
        # 3. 予測確率の分布
        ax3 = axes[1, 0]
        if 'predicted_probability' in analysis_df.columns:
            fp_probs = analysis_df[analysis_df['actual_label'] == 'Authentic']['predicted_probability']
            fn_probs = analysis_df[analysis_df['actual_label'] == 'Fake']['predicted_probability']
            
            ax3.hist(fp_probs, bins=20, alpha=0.6, label='False Positive (Authentic→Fake)', color='#ff6b6b')
            ax3.hist(fn_probs, bins=20, alpha=0.6, label='False Negative (Fake→Authentic)', color='#4ecdc4')
            ax3.set_xlabel('Predicted Probability', fontsize=12)
            ax3.set_ylabel('Frequency', fontsize=12)
            ax3.set_title('Prediction Probability Distribution by Misclassification Type', fontsize=14)
            ax3.legend()
            ax3.grid(axis='y', alpha=0.3)
        
        # 4. 生成モデル別の誤分類数（偽陰性のみ）
        ax4 = axes[1, 1]
        if 'generation_model' in analysis_df.columns:
            fn_by_model = analysis_df[analysis_df['actual_label'] == 'Fake'].groupby('generation_model').size()
            if len(fn_by_model) > 0:
                ax4.bar(fn_by_model.index, fn_by_model.values, color='#4ecdc4')
                ax4.set_xlabel('Generation Model', fontsize=12)
                ax4.set_ylabel('False Negative Count', fontsize=12)
                ax4.set_title('False Negatives (Fake→Authentic) by Generation Model', fontsize=14)
                ax4.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # 保存
        viz_file = self.viz_dir / 'misclassification_detailed_analysis.png'
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"誤分類詳細分析を保存: {viz_file}")
    
    def compare_generation_models(self, generation_models: List[str] = None) -> pd.DataFrame:
        """
        生成モデル間の詳細比較分析
        
        Args:
            generation_models: 比較する生成モデルのリスト（Noneの場合はすべて）
            
        Returns:
            比較結果のDataFrame
        """
        if generation_models is None:
            generation_models = ['Stable-Diffusion', 'FLUX', 'F-Lite']
        
        self.logger.info(f"生成モデル間の詳細比較分析を開始: {generation_models}")
        
        comparison_results = []
        
        for model in generation_models:
            # 特徴量ファイルを読み込み
            features_file = self.features_dir / f"wikiart_vlm_features_with_gestalt_{model}.csv"
            if not features_file.exists():
                features_file = self.features_dir / f"wikiart_vlm_features_{model}.csv"
            
            if not features_file.exists():
                self.logger.warning(f"特徴量ファイルが見つかりません: {features_file}")
                continue
            
            features_df = pd.read_csv(features_file)
            
            # モデルファイルを読み込み（存在チェックのみ、実際には使用しない）
            model_file = self.output_dir / 'models' / f'random_forest_model_{model}.pkl'
            if not model_file.exists():
                # デフォルトのモデルファイルを探す
                model_file = self.output_dir / 'models' / 'random_forest_model.pkl'
            
            # モデルファイルがなくても続行（特徴量ファイルから分析可能）
            
            # 訓練結果を読み込み（存在チェックのみ）
            results_file = self.output_dir / 'results' / f'training_results_{model}.txt'
            if not results_file.exists():
                results_file = self.output_dir / 'results' / 'training_results.txt'
            
            result = {
                'generation_model': model,
                'total_samples': len(features_df),
                'authentic_count': len(features_df[features_df['label'] == 0]) if 'label' in features_df.columns else 0,
                'fake_count': len(features_df[features_df['label'] == 1]) if 'label' in features_df.columns else 0,
            }
            
            # 訓練結果から精度を取得
            if results_file.exists():
                with open(results_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    import re
                    test_acc_match = re.search(r'テスト精度:\s*([\d.]+)', content)
                    if test_acc_match:
                        result['test_accuracy'] = float(test_acc_match.group(1))
            
            # 特徴量統計
            if 'mean_hue' in features_df.columns:
                fake_df = features_df[features_df['label'] == 1] if 'label' in features_df.columns else features_df
                result['avg_hue'] = fake_df['mean_hue'].mean()
                result['avg_saturation'] = fake_df['mean_saturation'].mean()
                result['avg_value'] = fake_df['mean_value'].mean()
            
            # ゲシュタルト原則スコア統計
            gestalt_cols = [col for col in features_df.columns if col.endswith('_score') and 'gestalt' not in col]
            gestalt_cols = [col for col in gestalt_cols if not col.startswith('_')]
            if len(gestalt_cols) > 0:
                fake_df = features_df[features_df['label'] == 1] if 'label' in features_df.columns else features_df
                for col in gestalt_cols[:6]:  # 最初の6つのゲシュタルトスコア
                    result[f'avg_{col}'] = fake_df[col].mean()
            
            comparison_results.append(result)
        
        if len(comparison_results) == 0:
            self.logger.warning("比較結果が見つかりませんでした")
            return pd.DataFrame()
        
        comparison_df = pd.DataFrame(comparison_results)
        
        # 可視化
        self._visualize_generation_model_comparison(comparison_df)
        
        # 結果を保存
        comparison_file = self.results_dir / 'generation_models_detailed_comparison.csv'
        comparison_df.to_csv(comparison_file, index=False)
        self.logger.info(f"詳細比較結果を保存: {comparison_file}")
        
        return comparison_df
    
    def _visualize_generation_model_comparison(self, comparison_df: pd.DataFrame) -> None:
        """
        生成モデル間の比較を可視化
        
        Args:
            comparison_df: 比較結果のDataFrame
        """
        if len(comparison_df) == 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. テスト精度の比較
        ax1 = axes[0, 0]
        if 'test_accuracy' in comparison_df.columns:
            ax1.bar(comparison_df['generation_model'], comparison_df['test_accuracy'], color='#3498db')
            ax1.set_ylabel('Test Accuracy', fontsize=12)
            ax1.set_title('Test Accuracy by Generation Model', fontsize=14)
            ax1.set_ylim([0, 1])
            ax1.grid(axis='y', alpha=0.3)
            for i, v in enumerate(comparison_df['test_accuracy']):
                ax1.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
        
        # 2. 色彩特徴量の比較（フェイク画像）
        ax2 = axes[0, 1]
        color_features = ['avg_hue', 'avg_saturation', 'avg_value']
        available_features = [f for f in color_features if f in comparison_df.columns]
        
        if available_features:
            x = np.arange(len(comparison_df))
            width = 0.25
            
            for i, feat in enumerate(available_features):
                offset = (i - len(available_features)/2) * width + width/2
                ax2.bar(x + offset, comparison_df[feat], width, label=feat.replace('avg_', ''))
            
            ax2.set_xlabel('Generation Model', fontsize=12)
            ax2.set_ylabel('Average Value', fontsize=12)
            ax2.set_title('Average Color Features (Fake Images) by Generation Model', fontsize=14)
            ax2.set_xticks(x)
            ax2.set_xticklabels(comparison_df['generation_model'], rotation=45, ha='right')
            ax2.legend()
            ax2.grid(axis='y', alpha=0.3)
        
        # 3. ゲシュタルト原則スコアの比較（フェイク画像）
        ax3 = axes[1, 0]
        gestalt_cols = [col for col in comparison_df.columns if col.startswith('avg_') and '_score' in col]
        if len(gestalt_cols) > 0:
            x = np.arange(len(comparison_df))
            width = 0.8 / len(gestalt_cols)
            
            for i, col in enumerate(gestalt_cols[:6]):  # 最初の6つ
                offset = (i - len(gestalt_cols[:6])/2) * width + width/2
                ax3.bar(x + offset, comparison_df[col], width, label=col.replace('avg_', '').replace('_score', ''))
            
            ax3.set_xlabel('Generation Model', fontsize=12)
            ax3.set_ylabel('Average Score', fontsize=12)
            ax3.set_title('Average Gestalt Scores (Fake Images) by Generation Model', fontsize=14)
            ax3.set_xticks(x)
            ax3.set_xticklabels(comparison_df['generation_model'], rotation=45, ha='right')
            ax3.legend(loc='upper left', fontsize=8)
            ax3.grid(axis='y', alpha=0.3)
        
        # 4. サンプル数の比較
        ax4 = axes[1, 1]
        if 'total_samples' in comparison_df.columns:
            ax4.bar(comparison_df['generation_model'], comparison_df['total_samples'], color='#2ecc71')
            ax4.set_ylabel('Total Samples', fontsize=12)
            ax4.set_title('Total Samples by Generation Model', fontsize=14)
            ax4.grid(axis='y', alpha=0.3)
            for i, v in enumerate(comparison_df['total_samples']):
                ax4.text(i, v + max(comparison_df['total_samples']) * 0.01, f'{int(v)}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # 保存
        viz_file = self.viz_dir / 'generation_models_detailed_comparison.png'
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"生成モデル比較を保存: {viz_file}")
    
    def visualize_authenticity_factors(self, generation_model: str = 'Stable-Diffusion') -> None:
        """
        「本物らしさはどこに宿るか？」の可視化
        
        Args:
            generation_model: 生成モデル名
        """
        self.logger.info(f"「本物らしさ」の可視化を開始 (モデル: {generation_model})")
        
        # 特徴量ファイルを読み込み
        features_file = self.features_dir / f"wikiart_vlm_features_with_gestalt_{generation_model}.csv"
        if not features_file.exists():
            features_file = self.features_dir / f"wikiart_vlm_features_{generation_model}.csv"
        
        if not features_file.exists():
            raise FileNotFoundError(f"特徴量ファイルが見つかりません: {features_file}")
        
        features_df = pd.read_csv(features_file)
        
        # 本物とフェイクを分離
        authentic_df = features_df[features_df['label'] == 0] if 'label' in features_df.columns else pd.DataFrame()
        fake_df = features_df[features_df['label'] == 1] if 'label' in features_df.columns else pd.DataFrame()
        
        if len(authentic_df) == 0 or len(fake_df) == 0:
            self.logger.warning("本物またはフェイクのデータがありません")
            return
        
        # SHAP重要度ファイルを読み込み
        shap_importance_file = self.output_dir / 'shap_explanations' / 'feature_importance_shap.csv'
        shap_importance = None
        if shap_importance_file.exists():
            shap_importance_df = pd.read_csv(shap_importance_file)
            # カラム名を確認して適切に処理
            if 'importance' in shap_importance_df.columns:
                if 'feature' in shap_importance_df.columns:
                    shap_importance = shap_importance_df.set_index('feature')['importance'].to_dict()
                else:
                    # featureカラムがない場合は最初のカラムをfeatureとして扱う
                    shap_importance = shap_importance_df.set_index(shap_importance_df.columns[0])['importance'].to_dict()
            elif len(shap_importance_df.columns) >= 2:
                # 最初の2列をfeature, importanceとして扱う
                shap_importance = shap_importance_df.set_index(shap_importance_df.columns[0])[shap_importance_df.columns[1]].to_dict()
        
        # 可視化
        self._create_authenticity_factor_visualization(authentic_df, fake_df, shap_importance)
    
    def _create_authenticity_factor_visualization(self, authentic_df: pd.DataFrame, 
                                                  fake_df: pd.DataFrame,
                                                  shap_importance: Optional[Dict[str, float]] = None) -> None:
        """
        「本物らしさ」の可視化を作成
        
        Args:
            authentic_df: 本物画像の特徴量
            fake_df: フェイク画像の特徴量
            shap_importance: SHAP特徴量重要度
        """
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. 色彩特徴量の比較（本物 vs フェイク）
        ax1 = fig.add_subplot(gs[0, 0])
        color_features = ['mean_hue', 'mean_saturation', 'mean_value', 'contrast', 'color_diversity']
        available_color = [f for f in color_features if f in authentic_df.columns]
        
        if available_color:
            authentic_means = [authentic_df[col].mean() for col in available_color]
            fake_means = [fake_df[col].mean() for col in available_color]
            
            x = np.arange(len(available_color))
            width = 0.35
            
            ax1.bar(x - width/2, authentic_means, width, label='Authentic', color='#27ae60')
            ax1.bar(x + width/2, fake_means, width, label='Fake', color='#e74c3c')
            
            ax1.set_xlabel('Color Features', fontsize=11)
            ax1.set_ylabel('Mean Value', fontsize=11)
            ax1.set_title('Color Features: Authentic vs Fake', fontsize=13)
            ax1.set_xticks(x)
            ax1.set_xticklabels(available_color, rotation=45, ha='right')
            ax1.legend()
            ax1.grid(axis='y', alpha=0.3)
        
        # 2. ゲシュタルト原則スコアの比較（本物 vs フェイク）
        ax2 = fig.add_subplot(gs[0, 1])
        gestalt_features = ['simplicity_score', 'proximity_score', 'similarity_score',
                           'continuity_score', 'closure_score', 'figure_ground_score']
        available_gestalt = [f for f in gestalt_features if f in authentic_df.columns]
        
        if available_gestalt:
            authentic_gestalt = [authentic_df[col].mean() for col in available_gestalt]
            fake_gestalt = [fake_df[col].mean() for col in available_gestalt]
            
            x = np.arange(len(available_gestalt))
            width = 0.35
            
            ax2.bar(x - width/2, authentic_gestalt, width, label='Authentic', color='#27ae60')
            ax2.bar(x + width/2, fake_gestalt, width, label='Fake', color='#e74c3c')
            
            ax2.set_xlabel('Gestalt Principles', fontsize=11)
            ax2.set_ylabel('Mean Score', fontsize=11)
            ax2.set_title('Gestalt Scores: Authentic vs Fake', fontsize=13)
            ax2.set_xticks(x)
            ax2.set_xticklabels([f.replace('_score', '') for f in available_gestalt], rotation=45, ha='right')
            ax2.legend()
            ax2.grid(axis='y', alpha=0.3)
        
        # 3. SHAP特徴量重要度（トップ10）
        ax3 = fig.add_subplot(gs[0, 2])
        if shap_importance:
            sorted_features = sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)[:10]
            feature_names = [f[0] for f in sorted_features]
            importance_values = [f[1] for f in sorted_features]
            
            ax3.barh(range(len(feature_names)), importance_values, color='#3498db')
            ax3.set_yticks(range(len(feature_names)))
            ax3.set_yticklabels(feature_names, fontsize=9)
            ax3.set_xlabel('SHAP Importance', fontsize=11)
            ax3.set_title('Top 10 Features by SHAP Importance', fontsize=13)
            ax3.grid(axis='x', alpha=0.3)
        
        # 4. 特徴量の分散比較
        ax4 = fig.add_subplot(gs[1, :])
        if available_color:
            authentic_stds = [authentic_df[col].std() for col in available_color]
            fake_stds = [fake_df[col].std() for col in available_color]
            
            x = np.arange(len(available_color))
            width = 0.35
            
            ax4.bar(x - width/2, authentic_stds, width, label='Authentic', color='#27ae60', alpha=0.7)
            ax4.bar(x + width/2, fake_stds, width, label='Fake', color='#e74c3c', alpha=0.7)
            
            ax4.set_xlabel('Color Features', fontsize=12)
            ax4.set_ylabel('Standard Deviation', fontsize=12)
            ax4.set_title('Color Feature Variability: Authentic vs Fake', fontsize=14)
            ax4.set_xticks(x)
            ax4.set_xticklabels(available_color, rotation=45, ha='right')
            ax4.legend()
            ax4.grid(axis='y', alpha=0.3)
        
        # 5. 「本物らしさ」の要因分析（テキスト）
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        analysis_text = self._generate_authenticity_analysis_text(authentic_df, fake_df, shap_importance)
        ax5.text(0.05, 0.95, analysis_text, transform=ax5.transAxes,
                fontsize=11, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle('Where Does "Authenticity" Reside?\nFeature Analysis: Authentic vs Fake Artworks', 
                     fontsize=16, fontweight='bold')
        
        # 保存
        viz_file = self.viz_dir / 'authenticity_factors_analysis.png'
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"「本物らしさ」の可視化を保存: {viz_file}")
    
    def _generate_authenticity_analysis_text(self, authentic_df: pd.DataFrame,
                                            fake_df: pd.DataFrame,
                                            shap_importance: Optional[Dict[str, float]] = None) -> str:
        """
        「本物らしさ」の分析テキストを生成
        
        Args:
            authentic_df: 本物画像の特徴量
            fake_df: フェイク画像の特徴量
            shap_importance: SHAP特徴量重要度
            
        Returns:
            分析テキスト
        """
        text = "Key Findings: What Makes Art 'Authentic'?\n"
        text += "=" * 70 + "\n\n"
        
        # 色彩特徴量の違い
        if 'mean_hue' in authentic_df.columns:
            hue_diff = authentic_df['mean_hue'].mean() - fake_df['mean_hue'].mean()
            sat_diff = authentic_df['mean_saturation'].mean() - fake_df['mean_saturation'].mean()
            val_diff = authentic_df['mean_value'].mean() - fake_df['mean_value'].mean()
            
            text += "1. COLOR CHARACTERISTICS:\n"
            text += f"   - Hue Difference: {hue_diff:.3f} (Authentic {'warmer' if hue_diff > 0 else 'cooler'})\n"
            text += f"   - Saturation Difference: {sat_diff:.3f} (Authentic {'more' if sat_diff > 0 else 'less'} saturated)\n"
            text += f"   - Value Difference: {val_diff:.3f} (Authentic {'brighter' if val_diff > 0 else 'darker'})\n\n"
        
        # ゲシュタルト原則の違い
        gestalt_features = ['simplicity_score', 'proximity_score', 'similarity_score',
                           'continuity_score', 'closure_score', 'figure_ground_score']
        available_gestalt = [f for f in gestalt_features if f in authentic_df.columns]
        
        if available_gestalt:
            text += "2. GESTALT PRINCIPLES:\n"
            for feat in available_gestalt:
                auth_mean = authentic_df[feat].mean()
                fake_mean = fake_df[feat].mean()
                diff = auth_mean - fake_mean
                feat_name = feat.replace('_score', '').replace('_', ' ').title()
                text += f"   - {feat_name}: {diff:+.2f} (Authentic {'higher' if diff > 0 else 'lower'})\n"
            text += "\n"
        
        # SHAP重要度
        if shap_importance:
            text += "3. MOST IMPORTANT FEATURES (by SHAP):\n"
            sorted_features = sorted(shap_importance.items(), key=lambda x: x[1], reverse=True)[:5]
            for i, (feat, importance) in enumerate(sorted_features, 1):
                text += f"   {i}. {feat}: {importance:.4f}\n"
            text += "\n"
        
        text += "4. EDUCATIONAL INSIGHTS:\n"
        text += "   - Authentic artworks show distinct patterns in color and composition\n"
        text += "   - AI-generated images may lack the subtle variations found in real art\n"
        text += "   - Gestalt principles help quantify the 'human touch' in art\n"
        text += "   - Understanding these differences raises questions about:\n"
        text += "     * What makes art 'authentic'?\n"
        text += "     * Can AI truly replicate artistic expression?\n"
        text += "     * How do we define 'real' vs 'fake' in the digital age?\n"
        
        return text
    
    def generate_educational_report(self, generation_model: str = 'Stable-Diffusion') -> None:
        """
        学生向けの教育的レポートを生成
        
        Args:
            generation_model: 生成モデル名
        """
        self.logger.info(f"教育的レポートを生成中 (モデル: {generation_model})")
        
        # 誤分類分析
        misclass_df = self.analyze_misclassifications(generation_model)
        
        # 生成モデル比較
        comparison_df = self.compare_generation_models([generation_model])
        
        # 「本物らしさ」の可視化
        self.visualize_authenticity_factors(generation_model)
        
        # レポートを生成
        report = self._create_educational_report(misclass_df, comparison_df, generation_model)
        
        # 保存
        report_file = self.results_dir / f'educational_report_{generation_model}.md'
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        self.logger.info(f"教育的レポートを保存: {report_file}")
    
    def _create_educational_report(self, misclass_df: pd.DataFrame, 
                                   comparison_df: pd.DataFrame,
                                   generation_model: str) -> str:
        """
        教育的レポートを作成
        
        Args:
            misclass_df: 誤分類分析結果
            comparison_df: 生成モデル比較結果
            generation_model: 生成モデル名
            
        Returns:
            レポートのMarkdownテキスト
        """
        report = f"""# Educational Report: Authentic vs Fake Art Classification

## Project Overview

This project explores the question: **"Where does 'authenticity' reside in art?"** using explainable AI to analyze the differences between authentic artworks and AI-generated images.

### Research Question

Can machine learning distinguish between authentic art and AI-generated images? What features make art "authentic"?

## Key Findings

### 1. Classification Performance

The model was trained to distinguish between:
- **Authentic (0)**: Original artworks from WikiArt_VLM dataset
- **Fake (1)**: AI-generated images using {generation_model}

### 2. Misclassification Analysis

"""
        
        if len(misclass_df) > 0:
            fp_count = len(misclass_df[misclass_df['actual_label'] == 'Authentic'])
            fn_count = len(misclass_df[misclass_df['actual_label'] == 'Fake'])
            
            report += f"""
- **False Positives (Authentic→Fake)**: {fp_count} cases
  - Authentic artworks incorrectly identified as fake
  - These artworks may have features that resemble AI-generated images

- **False Negatives (Fake→Authentic)**: {fn_count} cases
  - AI-generated images incorrectly identified as authentic
  - These generated images may be particularly convincing
"""
        else:
            report += "\nNo misclassifications found in the test set.\n"
        
        report += """
### 3. What Makes Art "Authentic"?

Based on the analysis, several factors contribute to authenticity:

#### Color Characteristics
- Authentic artworks show distinct color patterns
- Variations in hue, saturation, and value reflect artistic choices

#### Gestalt Principles
- **Simplicity**: How simple or complex is the composition?
- **Proximity**: How are elements grouped?
- **Similarity**: How similar are repeated elements?
- **Continuity**: How do lines and paths flow?
- **Closure**: How complete are the forms?
- **Figure-Ground Separation**: How clearly is the subject separated from background?

These principles help quantify the "human touch" in art.

## Discussion Topics for Students

### 1. Philosophical Questions
- What does "authenticity" mean in art?
- Can AI-generated art be considered "real" art?
- Is there a fundamental difference between human and AI creativity?

### 2. Technical Questions
- Which features are most important for distinguishing authentic from fake?
- How do different AI generation models compare in creating "authentic-looking" art?
- What role do Gestalt principles play in our perception of art?

### 3. Ethical Questions
- What are the implications of AI-generated art for artists?
- How should we attribute AI-generated artworks?
- What is the value of "authenticity" in the digital age?

## Conclusion

This project demonstrates that machine learning can identify patterns that distinguish authentic art from AI-generated images. However, the analysis also raises deeper questions about what makes art "authentic" and how we define creativity in the age of AI.

The findings suggest that authentic artworks have distinct characteristics in:
- Color composition and variation
- Application of Gestalt principles
- Subtle variations that reflect human artistic process

## References

- WikiArt_VLM Dataset
- SHAP (SHapley Additive exPlanations) for explainability
- Gestalt Principles in visual perception

---
*Generated by Explainable Art Classification Project*
"""
        
        return report

