"""
ランダムフォレスト訓練クラス
分類器の訓練、評価、保存を行う
"""

import pandas as pd
import numpy as np
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, List
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils.config_manager import ConfigManager
from ..utils.timestamp_manager import TimestampManager


class RandomForestTrainer:
    """ランダムフォレスト訓練クラス"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初期化
        
        Args:
            config: 設定辞書
        """
        self.config = config
        self.data_config = config['data']
        self.classification_config = config['classification']
        self.rf_config = self.classification_config['random_forest']
        
        self.logger = logging.getLogger(__name__)
        
        # 生データは固定ディレクトリから読み込み、分析結果はタイムスタンプ付きディレクトリに保存
        self.base_output_dir = Path(self.data_config['output_dir'])
        
        # タイムスタンプ管理（分析結果用）
        self.timestamp_manager = TimestampManager(config)
        self.output_dir = self.timestamp_manager.get_output_dir()
        self.models_dir = self.timestamp_manager.get_models_dir()
        self.results_dir = self.timestamp_manager.get_results_dir()
        self.viz_dir = self.timestamp_manager.get_visualizations_dir()
        
        # ディレクトリ作成
        self.timestamp_manager.create_directories()
        
        # ファイルパス
        self.model_file = self.timestamp_manager.get_model_file()
        self.results_file = self.timestamp_manager.get_results_file()
        
        # モデルとエンコーダー
        self.model = None
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
    
    def load_features(self) -> pd.DataFrame:
        """特徴量データを読み込み"""
        features_file = self.timestamp_manager.get_features_file()
        
        if not features_file.with_suffix('.csv').exists():
            raise FileNotFoundError(f"特徴量ファイルが見つかりません: {features_file}")
        
        df = pd.read_csv(features_file.with_suffix('.csv'))
        self.logger.info(f"特徴量データ読み込み完了: {len(df)}件")
        return df
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        訓練データを準備
        
        Args:
            df: 特徴量データフレーム
            
        Returns:
            X: 特徴量配列
            y: ラベル配列
            feature_names: 特徴量名のリスト
        """
        # 数値特徴量のみを抽出
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        feature_columns = [col for col in numeric_columns if col not in ['object_id']]
        
        # 特徴量とラベルを分離
        X = df[feature_columns].values
        self.feature_columns = feature_columns
        
        # ラベルの準備（アーティスト名ベースの分類）
        # 印象派の主要アーティストを定義
        impressionist_artists = [
            'Claude Monet', 'Pierre-Auguste Renoir', 'Edgar Degas', 'Camille Pissarro',
            'Berthe Morisot', 'Alfred Sisley', 'Paul Cézanne', 'Vincent van Gogh',
            'Paul Gauguin', 'Henri de Toulouse-Lautrec', 'Mary Cassatt', 'Gustave Caillebotte'
        ]
        
        # アーティスト名から印象派を判定
        df['is_impressionist'] = df['artist'].fillna('').str.contains(
            '|'.join(impressionist_artists), 
            case=False, 
            na=False
        )
        
        # 作品タイトルからも印象派を判定（補完）
        impressionist_keywords = ['impression', 'impressionist', 'monet', 'renoir', 'degas', 'pissarro']
        df.loc[df['title'].fillna('').str.contains('|'.join(impressionist_keywords), case=False, na=False), 'is_impressionist'] = True
        
        # ラベルをエンコード
        y = self.label_encoder.fit_transform(df['is_impressionist'])
        
        self.logger.info(f"データ準備完了:")
        self.logger.info(f"  - 特徴量数: {len(feature_columns)}")
        self.logger.info(f"  - サンプル数: {len(X)}")
        self.logger.info(f"  - クラス分布: {np.bincount(y)}")
        
        return X, y, feature_columns
    
    def train_model(self) -> None:
        """モデルを訓練"""
        self.logger.info("モデル訓練を開始...")
        
        # データを読み込み
        df = self.load_features()
        
        # データを準備
        X, y, feature_names = self.prepare_data(df)
        
        # 訓練・テスト分割
        test_size = self.classification_config['test_size']
        random_state = self.classification_config['random_state']
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # ランダムフォレストモデルの作成
        self.model = RandomForestClassifier(
            n_estimators=self.rf_config['n_estimators'],
            max_depth=self.rf_config['max_depth'],
            min_samples_split=self.rf_config['min_samples_split'],
            min_samples_leaf=self.rf_config['min_samples_leaf'],
            random_state=self.rf_config['random_state'],
            n_jobs=-1
        )
        
        # ハイパーパラメータチューニング
        self._tune_hyperparameters(X_train, y_train)
        
        # 最終モデルを訓練
        self.model.fit(X_train, y_train)
        
        # 評価
        self._evaluate_model(X_train, X_test, y_train, y_test, feature_names)
        
        # モデルを保存
        self._save_model()
        
        self.logger.info("モデル訓練完了")
    
    def _tune_hyperparameters(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """ハイパーパラメータをチューニング"""
        self.logger.info("ハイパーパラメータチューニングを開始...")
        
        # グリッドサーチのパラメータ
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        # グリッドサーチ実行
        grid_search = GridSearchCV(
            self.model,
            param_grid,
            cv=self.classification_config['cv_folds'],
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # 最適パラメータを設定
        self.model = grid_search.best_estimator_
        
        self.logger.info(f"最適パラメータ: {grid_search.best_params_}")
        self.logger.info(f"最適スコア: {grid_search.best_score_:.4f}")
    
    def _evaluate_model(self, X_train: np.ndarray, X_test: np.ndarray, 
                       y_train: np.ndarray, y_test: np.ndarray, 
                       feature_names: List[str]) -> None:
        """モデルを評価"""
        self.logger.info("モデル評価を開始...")
        
        # 予測
        y_train_pred = self.model.predict(X_train)
        y_test_pred = self.model.predict(X_test)
        
        # 訓練データの評価
        train_accuracy = accuracy_score(y_train, y_train_pred)
        
        # テストデータの評価
        test_accuracy = accuracy_score(y_test, y_test_pred)
        
        # 交差検証
        cv_scores = cross_val_score(self.model, X_train, y_train, 
                                   cv=self.classification_config['cv_folds'])
        
        # 結果をログに出力
        self.logger.info(f"訓練精度: {train_accuracy:.4f}")
        self.logger.info(f"テスト精度: {test_accuracy:.4f}")
        self.logger.info(f"交差検証スコア: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        # 詳細な分類レポート
        class_names = [str(cls) for cls in self.label_encoder.classes_]
        report = classification_report(y_test, y_test_pred, target_names=class_names)
        self.logger.info(f"分類レポート:\n{report}")
        
        # 混同行列を可視化
        self._plot_confusion_matrix(y_test, y_test_pred, class_names)
        
        # 特徴量重要度を可視化
        self._plot_feature_importance(feature_names)
        
        # 結果をファイルに保存
        self._save_results(train_accuracy, test_accuracy, cv_scores, report)
    
    def _plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                              class_names: List[str]) -> None:
        """混同行列を可視化"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('混同行列')
        plt.xlabel('予測ラベル')
        plt.ylabel('実際のラベル')
        
        # 保存
        cm_file = self.viz_dir / 'confusion_matrix.png'
        plt.savefig(cm_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"混同行列を保存: {cm_file}")
    
    def _plot_feature_importance(self, feature_names: List[str]) -> None:
        """特徴量重要度を可視化"""
        importance = self.model.feature_importances_
        
        # 重要度でソート
        indices = np.argsort(importance)[::-1]
        
        # 上位20個の特徴量を表示
        top_n = min(20, len(feature_names))
        
        plt.figure(figsize=(10, 8))
        plt.title(f'特徴量重要度 (上位{top_n}個)')
        plt.bar(range(top_n), importance[indices[:top_n]])
        plt.xticks(range(top_n), [feature_names[i] for i in indices[:top_n]], 
                  rotation=45, ha='right')
        plt.tight_layout()
        
        # 保存
        importance_file = self.viz_dir / 'feature_importance.png'
        plt.savefig(importance_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"特徴量重要度を保存: {importance_file}")
    
    def _save_results(self, train_accuracy: float, test_accuracy: float, 
                     cv_scores: np.ndarray, report: str) -> None:
        """結果をファイルに保存"""
        with open(self.results_file, 'w', encoding='utf-8') as f:
            f.write("ランダムフォレスト訓練結果\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"訓練精度: {train_accuracy:.4f}\n")
            f.write(f"テスト精度: {test_accuracy:.4f}\n")
            f.write(f"交差検証スコア: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})\n\n")
            f.write("分類レポート:\n")
            f.write(report)
    
    def _save_model(self) -> None:
        """モデルを保存"""
        model_data = {
            'model': self.model,
            'label_encoder': self.label_encoder,
            'feature_columns': self.feature_columns
        }
        
        with open(self.model_file, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"モデルを保存: {self.model_file}")
    
    def load_model(self) -> None:
        """保存されたモデルを読み込み"""
        if not self.model_file.exists():
            raise FileNotFoundError(f"モデルファイルが見つかりません: {self.model_file}")
        
        with open(self.model_file, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.label_encoder = model_data['label_encoder']
        self.feature_columns = model_data['feature_columns']
        
        self.logger.info("モデル読み込み完了")
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        予測を実行
        
        Args:
            X: 特徴量配列
            
        Returns:
            predictions: 予測ラベル
            probabilities: 予測確率
        """
        if self.model is None:
            raise ValueError("モデルが訓練されていません")
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        return predictions, probabilities
