"""
SHAP説明クラス
モデルの予測をSHAPで解釈し、可視化する
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.preprocessing import StandardScaler
import pickle

from ..utils.config_manager import ConfigManager
from ..utils.timestamp_manager import TimestampManager


class SHAPExplainer:
    """SHAP説明クラス"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初期化
        
        Args:
            config: 設定辞書
        """
        self.config = config
        self.data_config = config['data']
        self.shap_config = config['shap']
        self.viz_config = config['visualization']
        
        self.logger = logging.getLogger(__name__)
        
        # 生データは固定ディレクトリから読み込み、分析結果はタイムスタンプ付きディレクトリに保存
        self.base_output_dir = Path(self.data_config['output_dir'])
        
        # タイムスタンプ管理（分析結果用）
        self.timestamp_manager = TimestampManager(config)
        self.output_dir = self.timestamp_manager.get_output_dir()
        self.shap_dir = self.timestamp_manager.get_shap_explanations_dir()
        self.viz_dir = self.timestamp_manager.get_visualizations_dir()
        
        # ディレクトリ作成
        self.timestamp_manager.create_directories()
        
        # モデルとデータ
        self.model = None
        self.label_encoder = None
        self.feature_columns = None
        self.scaler = None
        self.explainer = None
    
    def load_model_and_data(self) -> Tuple[pd.DataFrame, np.ndarray]:
        """モデルとデータを読み込み"""
        # 既存のモデルファイルを探す（タイムスタンプ付きディレクトリから）
        base_dir = self.base_output_dir
        model_file = None
        scaler_file = None
        features_file = None
        
        # 既存のanalysis_*ディレクトリから最新のものを探す
        analysis_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith('analysis_')]
        if analysis_dirs:
            # 最新のディレクトリを選択
            latest_dir = max(analysis_dirs, key=lambda x: x.stat().st_mtime)
            model_file = latest_dir / 'models' / 'random_forest_model.pkl'
            scaler_file = latest_dir / 'models' / 'scaler.pkl'
            features_file = latest_dir / 'features' / self.data_config['features_file']
            
            # タイムスタンプ付きディレクトリにファイルがない場合は、直接のディレクトリを試す
            if not model_file.exists():
                model_file = base_dir / 'models' / 'random_forest_model.pkl'
                scaler_file = base_dir / 'models' / 'scaler.pkl'
                features_file = base_dir / 'features' / self.data_config['features_file']
        else:
            # フォールバック: 直接のmodelsディレクトリ
            model_file = base_dir / 'models' / 'random_forest_model.pkl'
            scaler_file = base_dir / 'models' / 'scaler.pkl'
            features_file = base_dir / 'features' / self.data_config['features_file']
        
        if not model_file.exists():
            raise FileNotFoundError(f"モデルファイルが見つかりません: {model_file}")
        
        with open(model_file, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.feature_columns = model_data['feature_columns']
        
        # スケーラーを読み込み
        if scaler_file.exists():
            with open(scaler_file, 'rb') as f:
                self.scaler = pickle.load(f)
        
        # 特徴量データを読み込み
        if not features_file.with_suffix('.csv').exists():
            raise FileNotFoundError(f"特徴量ファイルが見つかりません: {features_file}")
        
        df = pd.read_csv(features_file.with_suffix('.csv'))
        
        # 特徴量を抽出
        X = df[self.feature_columns].values
        
        self.logger.info(f"モデルとデータ読み込み完了:")
        self.logger.info(f"  - サンプル数: {len(df)}")
        self.logger.info(f"  - 特徴量数: {len(self.feature_columns)}")
        
        return df, X
    
    def create_explainer(self, X: np.ndarray) -> None:
        """SHAP説明器を作成"""
        self.logger.info("SHAP説明器を作成中...")
        
        # 背景データを準備（ランダムサンプリング）
        background_size = min(self.shap_config['background_samples'], len(X))
        background_indices = np.random.choice(len(X), background_size, replace=False)
        background_data = X[background_indices]
        
        # TreeExplainerを作成
        self.explainer = shap.TreeExplainer(self.model, background_data)
        
        self.logger.info(f"SHAP説明器作成完了 (背景データ: {background_size}件)")
    
    def calculate_shap_values(self, X: np.ndarray) -> np.ndarray:
        """SHAP値を計算"""
        self.logger.info("SHAP値を計算中...")
        
        # 説明用サンプルを準備
        max_samples = min(self.shap_config['max_samples'], len(X))
        sample_indices = np.random.choice(len(X), max_samples, replace=False)
        sample_data = X[sample_indices]
        
        # SHAP値を計算
        shap_values = self.explainer.shap_values(sample_data)
        
        self.logger.info(f"SHAP値の初期形状: {type(shap_values)}, {shap_values.shape if hasattr(shap_values, 'shape') else 'N/A'}")
        
        # 二値分類の場合の処理
        if isinstance(shap_values, list):
            # リストの場合は正のクラス（印象派）のSHAP値を使用
            self.logger.info(f"リスト形式のSHAP値: {len(shap_values)}個のクラス")
            shap_values = shap_values[1]  # 正のクラス（印象派）
        elif len(shap_values.shape) == 3:
            # 3次元配列の場合は正のクラスのSHAP値を使用
            self.logger.info(f"3次元配列のSHAP値: {shap_values.shape}")
            shap_values = shap_values[:, :, 1]  # 正のクラス（印象派）の列
        elif len(shap_values.shape) == 2 and shap_values.shape[1] == 2:
            # 2次元配列で2列の場合は正のクラス（印象派）のSHAP値を使用
            self.logger.info(f"2次元配列のSHAP値: {shap_values.shape}")
            shap_values = shap_values[:, 1]  # 正のクラス（印象派）の列
        
        self.logger.info(f"SHAP値計算完了 (サンプル数: {max_samples}, 形状: {shap_values.shape})")
        
        return shap_values, sample_data, sample_indices
    
    def generate_explanations(self) -> None:
        """SHAP説明を生成"""
        self.logger.info("SHAP説明生成を開始...")
        
        # モデルとデータを読み込み
        df, X = self.load_model_and_data()
        
        # 説明器を作成
        self.create_explainer(X)
        
        # SHAP値を計算
        shap_values, sample_data, sample_indices = self.calculate_shap_values(X)
        
        # 可視化を生成
        self._create_summary_plot(shap_values, sample_data)
        self._create_feature_importance_plot(shap_values)
        self._create_dependence_plots(shap_values, sample_data)
        
        # 結果を保存
        self._save_shap_values(shap_values, sample_indices)
        
        self.logger.info("SHAP説明生成完了")
    
    def _create_summary_plot(self, shap_values: np.ndarray, sample_data: np.ndarray) -> None:
        """SHAP summary plotを作成"""
        plt.figure(figsize=(12, 8))
        
        # SHAP summary plot
        shap.summary_plot(shap_values, sample_data, 
                         feature_names=self.feature_columns,
                         show=False, max_display=20)
        
        plt.title('SHAP Summary Plot - 特徴量の寄与度', fontsize=16)
        plt.tight_layout()
        
        # 保存
        summary_file = self.viz_dir / 'shap_summary_plot.png'
        plt.savefig(summary_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Summary plotを保存: {summary_file}")
    
    def _create_waterfall_plots(self, shap_values: np.ndarray, sample_data: np.ndarray, 
                               sample_indices: np.ndarray, df: pd.DataFrame) -> None:
        """特徴量寄与度プロットを作成（上位5サンプル）"""
        # 予測確率が高いサンプルを選択
        predictions = self.model.predict_proba(sample_data)
        prob_impressionist = predictions[:, 1]  # 印象派の確率
        
        # 上位5サンプルを選択
        top_indices = np.argsort(prob_impressionist)[-5:]
        
        for i, idx in enumerate(top_indices):
            try:
                # 単一サンプルのSHAP値を取得
                single_shap_values = shap_values[idx:idx+1]  # 2次元配列として保持
                single_sample_data = sample_data[idx:idx+1]
                
                # SHAP Explanationオブジェクトを作成
                explanation = shap.Explanation(
                    values=single_shap_values,
                    base_values=self.explainer.expected_value[1],
                    data=single_sample_data,
                    feature_names=self.feature_columns
                )
                
                # Waterfall plotを作成
                plt.figure(figsize=(12, 8))
                shap.plots.waterfall(explanation[0], show=False, max_display=15)
                
                plt.title(f'SHAP Waterfall Plot - {df.iloc[sample_indices[idx]]["title"][:50]}...\n'
                         f'Artist: {df.iloc[sample_indices[idx]]["artist"]} | '
                         f'Predicted Probability: {prob_impressionist[idx]:.3f}',
                         fontsize=12)
                
                # 保存
                waterfall_file = self.shap_dir / f'waterfall_plot_{i+1}.png'
                plt.savefig(waterfall_file, dpi=300, bbox_inches='tight')
                plt.close()
                
            except Exception as e:
                self.logger.warning(f"Waterfall plot {i+1} の作成に失敗: {e}")
                # フォールバック: カスタムバープロット
                self._create_custom_waterfall_plot(shap_values, sample_data, sample_indices, df, idx, i)
        
        self.logger.info(f"Waterfall plotsを保存: {self.shap_dir}/waterfall_plot_*.png")
    
    def _create_custom_waterfall_plot(self, shap_values: np.ndarray, sample_data: np.ndarray, 
                                    sample_indices: np.ndarray, df: pd.DataFrame, 
                                    idx: int, plot_num: int) -> None:
        """カスタムWaterfall plot（フォールバック用）"""
        plt.figure(figsize=(12, 8))
        
        # 特徴量寄与度を可視化
        feature_contributions = shap_values[idx]
        sorted_indices = np.argsort(np.abs(feature_contributions))[::-1]
        
        # 上位15個の特徴量を表示
        top_features = sorted_indices[:15]
        top_contributions = feature_contributions[top_features]
        
        colors = ['red' if float(x) < 0 else 'blue' for x in top_contributions]
        bars = plt.barh(range(len(top_features)), top_contributions, color=colors)
        
        plt.yticks(range(len(top_features)), [self.feature_columns[j] for j in top_features])
        plt.xlabel('SHAP Value')
        plt.title(f'Feature Contributions - {df.iloc[sample_indices[idx]]["title"][:50]}...\n'
                 f'Artist: {df.iloc[sample_indices[idx]]["artist"]} | '
                 f'Predicted Probability: {self.model.predict_proba(sample_data[idx:idx+1])[0][1]:.3f}',
                 fontsize=12)
        plt.gca().invert_yaxis()
        
        # 保存
        waterfall_file = self.shap_dir / f'custom_waterfall_plot_{plot_num+1}.png'
        plt.savefig(waterfall_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_feature_importance_plot(self, shap_values: np.ndarray) -> None:
        """特徴量重要度プロットを作成"""
        # 特徴量の平均絶対SHAP値を計算
        mean_shap_values = np.mean(np.abs(shap_values), axis=0)
        
        # 重要度でソート
        sorted_indices = np.argsort(mean_shap_values)[::-1]
        
        # 上位15個の特徴量を表示
        top_n = min(15, len(self.feature_columns))
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(range(top_n), mean_shap_values[sorted_indices[:top_n]])
        plt.yticks(range(top_n), [self.feature_columns[i] for i in sorted_indices[:top_n]])
        plt.xlabel('平均絶対SHAP値')
        plt.title('SHAP特徴量重要度 (平均絶対値)', fontsize=16)
        plt.gca().invert_yaxis()
        
        # バーの色をグラデーションに
        for i, bar in enumerate(bars):
            bar.set_color(plt.cm.viridis(i / top_n))
        
        plt.tight_layout()
        
        # 保存
        importance_file = self.viz_dir / 'shap_feature_importance.png'
        plt.savefig(importance_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"特徴量重要度プロットを保存: {importance_file}")
    
    def _create_dependence_plots(self, shap_values: np.ndarray, sample_data: np.ndarray) -> None:
        """Dependence plotを作成（上位5特徴量）"""
        # 特徴量重要度を計算
        mean_shap_values = np.mean(np.abs(shap_values), axis=0)
        top_features = np.argsort(mean_shap_values)[-5:]
        
        for i, feature_idx in enumerate(top_features):
            plt.figure(figsize=(8, 6))
            
            # Dependence plot
            shap.dependence_plot(
                feature_idx,
                shap_values,
                sample_data,
                feature_names=self.feature_columns,
                show=False
            )
            
            plt.title(f'Dependence Plot - {self.feature_columns[feature_idx]}', fontsize=14)
            plt.tight_layout()
            
            # 保存
            dep_file = self.viz_dir / f'dependence_plot_{i+1}.png'
            plt.savefig(dep_file, dpi=300, bbox_inches='tight')
            plt.close()
        
        self.logger.info(f"Dependence plotsを保存: {self.viz_dir}/dependence_plot_*.png")
    
    def _save_shap_values(self, shap_values: np.ndarray, sample_indices: np.ndarray) -> None:
        """SHAP値を保存"""
        # SHAP値をDataFrameに変換
        shap_df = pd.DataFrame(shap_values, columns=self.feature_columns)
        shap_df['sample_index'] = sample_indices
        
        # 保存
        shap_file = self.shap_dir / 'shap_values.csv'
        shap_df.to_csv(shap_file, index=False)
        
        # 特徴量重要度も保存
        mean_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'mean_abs_shap': np.mean(np.abs(shap_values), axis=0)
        }).sort_values('mean_abs_shap', ascending=False)
        
        importance_file = self.shap_dir / 'feature_importance_shap.csv'
        mean_importance.to_csv(importance_file, index=False)
        
        self.logger.info(f"SHAP値を保存: {shap_file}")
        self.logger.info(f"特徴量重要度を保存: {importance_file}")
    
    def explain_single_prediction(self, X: np.ndarray, feature_names: List[str] = None) -> Dict[str, Any]:
        """
        単一予測の説明を生成
        
        Args:
            X: 特徴量配列（1サンプル）
            feature_names: 特徴量名のリスト
            
        Returns:
            説明結果の辞書
        """
        if self.explainer is None:
            raise ValueError("説明器が初期化されていません")
        
        if feature_names is None:
            feature_names = self.feature_columns
        
        # SHAP値を計算
        shap_values = self.explainer.shap_values(X.reshape(1, -1))
        
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # 正のクラス
        
        # 予測確率を取得
        prediction = self.model.predict_proba(X.reshape(1, -1))
        prob_impressionist = prediction[0][1]
        
        # 特徴量の寄与度を計算
        feature_contributions = []
        for i, (feature, shap_val) in enumerate(zip(feature_names, shap_values[0])):
            feature_contributions.append({
                'feature': feature,
                'shap_value': shap_val,
                'abs_shap_value': abs(shap_val)
            })
        
        # 重要度でソート
        feature_contributions.sort(key=lambda x: x['abs_shap_value'], reverse=True)
        
        return {
            'prediction_probability': prob_impressionist,
            'predicted_class': 'Impressionist' if prob_impressionist > 0.5 else 'Non-Impressionist',
            'base_value': self.explainer.expected_value[1],
            'feature_contributions': feature_contributions[:10]  # 上位10個
        }
