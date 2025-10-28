"""
結果可視化クラス
分類結果とSHAP説明の可視化を行う
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from ..utils.config_manager import ConfigManager


class ResultVisualizer:
    """結果可視化クラス"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初期化
        
        Args:
            config: 設定辞書
        """
        self.config = config
        self.data_config = config['data']
        self.viz_config = config['visualization']
        
        self.logger = logging.getLogger(__name__)
        
        # 出力ディレクトリ
        self.output_dir = Path(self.data_config['output_dir'])
        self.viz_dir = self.output_dir / 'visualizations'
        self.viz_dir.mkdir(exist_ok=True)
        
        # スタイル設定
        plt.style.use('seaborn-v0_8')
        sns.set_palette(self.viz_config['color_palette'])
    
    def create_comprehensive_report(self) -> None:
        """包括的なレポートを作成"""
        self.logger.info("包括的レポートを作成中...")
        
        # データを読み込み
        df = self._load_data()
        
        # 各種可視化を作成
        self._create_data_overview(df)
        self._create_classification_results(df)
        self._create_feature_analysis(df)
        self._create_shap_analysis()
        self._create_misclassification_analysis(df)
        
        # HTMLレポートを生成
        self._create_html_report()
        
        self.logger.info("包括的レポート作成完了")
    
    def _load_data(self) -> pd.DataFrame:
        """データを読み込み"""
        # 特徴量データ
        features_file = self.output_dir / self.data_config['features_file']
        if not features_file.with_suffix('.csv').exists():
            raise FileNotFoundError(f"特徴量ファイルが見つかりません: {features_file}")
        
        df = pd.read_csv(features_file.with_suffix('.csv'))
        
        # ラベルを追加
        df['is_impressionist'] = df['period'].str.contains(
            'impressionist|Impressionist|1860|1870|1880|1890', 
            case=False, 
            na=False
        )
        
        return df
    
    def _create_data_overview(self, df: pd.DataFrame) -> None:
        """データ概要の可視化"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 期間別分布
        period_counts = df['period'].value_counts().head(10)
        axes[0, 0].bar(range(len(period_counts)), period_counts.values)
        axes[0, 0].set_xticks(range(len(period_counts)))
        axes[0, 0].set_xticklabels(period_counts.index, rotation=45, ha='right')
        axes[0, 0].set_title('期間別作品数 (上位10)')
        axes[0, 0].set_ylabel('作品数')
        
        # 2. 印象派 vs 非印象派
        impressionist_counts = df['is_impressionist'].value_counts()
        axes[0, 1].pie(impressionist_counts.values, 
                      labels=['非印象派', '印象派'],
                      autopct='%1.1f%%',
                      startangle=90)
        axes[0, 1].set_title('印象派 vs 非印象派の分布')
        
        # 3. 部門別分布
        dept_counts = df['department'].value_counts().head(8)
        axes[1, 0].barh(range(len(dept_counts)), dept_counts.values)
        axes[1, 0].set_yticks(range(len(dept_counts)))
        axes[1, 0].set_yticklabels(dept_counts.index)
        axes[1, 0].set_title('部門別作品数 (上位8)')
        axes[1, 0].set_xlabel('作品数')
        
        # 4. 文化圏別分布
        culture_counts = df['culture'].value_counts().head(8)
        axes[1, 1].bar(range(len(culture_counts)), culture_counts.values)
        axes[1, 1].set_xticks(range(len(culture_counts)))
        axes[1, 1].set_xticklabels(culture_counts.index, rotation=45, ha='right')
        axes[1, 1].set_title('文化圏別作品数 (上位8)')
        axes[1, 1].set_ylabel('作品数')
        
        plt.tight_layout()
        
        # 保存
        overview_file = self.viz_dir / 'data_overview.png'
        plt.savefig(overview_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"データ概要を保存: {overview_file}")
    
    def _create_classification_results(self, df: pd.DataFrame) -> None:
        """分類結果の可視化"""
        # モデル結果を読み込み
        results_file = self.output_dir / 'training_results.txt'
        if not results_file.exists():
            self.logger.warning("訓練結果ファイルが見つかりません")
            return
        
        # 混同行列を読み込み
        cm_file = self.output_dir / 'confusion_matrix.png'
        if cm_file.exists():
            # 混同行列をコピー
            import shutil
            viz_cm_file = self.viz_dir / 'confusion_matrix.png'
            shutil.copy2(cm_file, viz_cm_file)
        
        # 特徴量重要度を読み込み
        importance_file = self.output_dir / 'feature_importance.png'
        if importance_file.exists():
            viz_importance_file = self.viz_dir / 'feature_importance.png'
            shutil.copy2(importance_file, viz_importance_file)
        
        self.logger.info("分類結果の可視化を保存")
    
    def _create_feature_analysis(self, df: pd.DataFrame) -> None:
        """特徴量分析の可視化"""
        # 数値特徴量を抽出
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        feature_columns = [col for col in numeric_columns if col not in ['object_id']]
        
        # 印象派 vs 非印象派で特徴量を比較
        impressionist_data = df[df['is_impressionist'] == True][feature_columns]
        non_impressionist_data = df[df['is_impressionist'] == False][feature_columns]
        
        # 上位10個の特徴量を選択（分散が大きいもの）
        feature_vars = df[feature_columns].var().sort_values(ascending=False)
        top_features = feature_vars.head(10).index
        
        # 箱ひげ図を作成
        fig, axes = plt.subplots(2, 5, figsize=(20, 10))
        axes = axes.flatten()
        
        for i, feature in enumerate(top_features):
            data_to_plot = [impressionist_data[feature].dropna(), 
                           non_impressionist_data[feature].dropna()]
            
            axes[i].boxplot(data_to_plot, labels=['印象派', '非印象派'])
            axes[i].set_title(feature, fontsize=10)
            axes[i].tick_params(axis='x', rotation=45)
        
        plt.suptitle('印象派 vs 非印象派 特徴量比較', fontsize=16)
        plt.tight_layout()
        
        # 保存
        feature_analysis_file = self.viz_dir / 'feature_analysis.png'
        plt.savefig(feature_analysis_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"特徴量分析を保存: {feature_analysis_file}")
    
    def _create_shap_analysis(self) -> None:
        """SHAP分析の可視化"""
        shap_dir = self.output_dir / 'shap_results'
        
        if not shap_dir.exists():
            self.logger.warning("SHAP結果ディレクトリが見つかりません")
            return
        
        # SHAPファイルをコピー
        import shutil
        
        shap_files = [
            'shap_summary_plot.png',
            'shap_feature_importance.png'
        ]
        
        for file_name in shap_files:
            src_file = shap_dir / file_name
            if src_file.exists():
                dst_file = self.viz_dir / file_name
                shutil.copy2(src_file, dst_file)
        
        # Waterfall plotsをコピー
        waterfall_files = list(shap_dir.glob('waterfall_plot_*.png'))
        for src_file in waterfall_files:
            dst_file = self.viz_dir / src_file.name
            shutil.copy2(src_file, dst_file)
        
        # Dependence plotsをコピー
        dep_files = list(shap_dir.glob('dependence_plot_*.png'))
        for src_file in dep_files:
            dst_file = self.viz_dir / src_file.name
            shutil.copy2(src_file, dst_file)
        
        self.logger.info("SHAP分析の可視化を保存")
    
    def _create_misclassification_analysis(self, df: pd.DataFrame) -> None:
        """誤分類分析の可視化"""
        # 特徴量データを読み込み
        features_file = self.output_dir / self.data_config['features_file']
        if not features_file.with_suffix('.csv').exists():
            return
        
        features_df = pd.read_csv(features_file.with_suffix('.csv'))
        
        # モデルを読み込み
        model_file = self.output_dir / 'random_forest_model.pkl'
        if not model_file.exists():
            return
        
        import pickle
        with open(model_file, 'rb') as f:
            model_data = pickle.load(f)
        
        model = model_data['model']
        feature_columns = model_data['feature_columns']
        
        # 予測を実行
        X = features_df[feature_columns].values
        predictions = model.predict(X)
        probabilities = model.predict_proba(X)
        
        # 実際のラベルを準備
        features_df['is_impressionist'] = features_df['period'].str.contains(
            'impressionist|Impressionist|1860|1870|1880|1890', 
            case=False, 
            na=False
        )
        y_true = features_df['is_impressionist'].astype(int).values
        
        # 誤分類を特定
        misclassified = (predictions != y_true)
        misclassified_df = features_df[misclassified].copy()
        misclassified_df['predicted_prob'] = probabilities[:, 1]
        misclassified_df['actual_label'] = y_true[misclassified]
        misclassified_df['predicted_label'] = predictions[misclassified]
        
        if len(misclassified_df) == 0:
            self.logger.info("誤分類作品がありません")
            return
        
        # 誤分類作品の分析
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. 誤分類の分布
        misclass_counts = misclassified_df.groupby(['actual_label', 'predicted_label']).size()
        misclass_matrix = misclass_counts.unstack(fill_value=0)
        
        sns.heatmap(misclass_matrix, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0])
        axes[0, 0].set_title('誤分類マトリックス')
        axes[0, 0].set_xlabel('予測ラベル')
        axes[0, 0].set_ylabel('実際のラベル')
        
        # 2. 予測確率の分布
        axes[0, 1].hist(misclassified_df['predicted_prob'], bins=20, alpha=0.7)
        axes[0, 1].set_title('誤分類作品の予測確率分布')
        axes[0, 1].set_xlabel('印象派予測確率')
        axes[0, 1].set_ylabel('頻度')
        
        # 3. 期間別誤分類率
        period_misclass = misclassified_df.groupby('period').size()
        period_total = features_df.groupby('period').size()
        period_error_rate = (period_misclass / period_total).fillna(0).sort_values(ascending=False).head(10)
        
        axes[1, 0].bar(range(len(period_error_rate)), period_error_rate.values)
        axes[1, 0].set_xticks(range(len(period_error_rate)))
        axes[1, 0].set_xticklabels(period_error_rate.index, rotation=45, ha='right')
        axes[1, 0].set_title('期間別誤分類率 (上位10)')
        axes[1, 0].set_ylabel('誤分類率')
        
        # 4. 誤分類作品の例
        sample_misclass = misclassified_df.head(5)
        axes[1, 1].axis('off')
        
        text = "誤分類作品の例:\n\n"
        for _, row in sample_misclass.iterrows():
            text += f"• {row['title'][:30]}...\n"
            text += f"  Artist: {row['artist']}\n"
            text += f"  Period: {row['period']}\n"
            text += f"  Pred: {row['predicted_label']}, Prob: {row['predicted_prob']:.3f}\n\n"
        
        axes[1, 1].text(0.05, 0.95, text, transform=axes[1, 1].transAxes, 
                       fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        # 保存
        misclass_file = self.viz_dir / 'misclassification_analysis.png'
        plt.savefig(misclass_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"誤分類分析を保存: {misclass_file}")
    
    def _create_html_report(self) -> None:
        """HTMLレポートを生成"""
        html_content = f"""
<!DOCTYPE html>
<html lang="ja">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>説明可能AIによる絵画様式分類 - 結果レポート</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 0 20px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            margin-top: 30px;
            border-left: 4px solid #3498db;
            padding-left: 15px;
        }}
        .image-container {{
            text-align: center;
            margin: 20px 0;
        }}
        .image-container img {{
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        .summary {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .feature-list {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin: 20px 0;
        }}
        .feature-item {{
            background-color: #f8f9fa;
            padding: 10px;
            border-radius: 5px;
            border-left: 3px solid #3498db;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>AIは名画を理解できるか？</h1>
        <h2>説明可能AIによる絵画様式分類 - 結果レポート</h2>
        
        <div class="summary">
            <h3>プロジェクト概要</h3>
            <p>このプロジェクトでは、ランダムフォレストとSHAPを用いて、Metropolitan Museumの作品を印象派と非印象派に分類し、その判断根拠を解釈可能な形で可視化しました。</p>
        </div>
        
        <h2>1. データ概要</h2>
        <div class="image-container">
            <img src="data_overview.png" alt="データ概要">
        </div>
        
        <h2>2. 分類結果</h2>
        <div class="image-container">
            <img src="confusion_matrix.png" alt="混同行列">
        </div>
        <div class="image-container">
            <img src="feature_importance.png" alt="特徴量重要度">
        </div>
        
        <h2>3. 特徴量分析</h2>
        <div class="image-container">
            <img src="feature_analysis.png" alt="特徴量分析">
        </div>
        
        <h2>4. SHAP説明</h2>
        <div class="image-container">
            <img src="shap_summary_plot.png" alt="SHAP Summary Plot">
        </div>
        <div class="image-container">
            <img src="shap_feature_importance.png" alt="SHAP特徴量重要度">
        </div>
        
        <h2>5. 誤分類分析</h2>
        <div class="image-container">
            <img src="misclassification_analysis.png" alt="誤分類分析">
        </div>
        
        <h2>6. 結論</h2>
        <div class="summary">
            <p>このプロジェクトにより、機械学習モデルが芸術作品の様式を分類する際に、どの特徴量を重視しているかを定量的に分析できました。SHAPを用いることで、モデルの判断根拠を人間が理解しやすい形で可視化し、AIと芸術の関係について新たな知見を得ることができました。</p>
        </div>
        
        <footer style="margin-top: 50px; text-align: center; color: #7f8c8d;">
            <p>説明可能AIによる絵画様式分類プロジェクト - 2024</p>
        </footer>
    </div>
</body>
</html>
        """
        
        # HTMLファイルを保存
        html_file = self.viz_dir / 'report.html'
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"HTMLレポートを保存: {html_file}")
    
    def create_interactive_dashboard(self) -> None:
        """インタラクティブダッシュボードを作成"""
        # データを読み込み
        df = self._load_data()
        
        # Plotlyでインタラクティブな可視化を作成
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('期間別分布', '印象派 vs 非印象派', '部門別分布', '特徴量重要度'),
            specs=[[{"type": "bar"}, {"type": "pie"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # 1. 期間別分布
        period_counts = df['period'].value_counts().head(10)
        fig.add_trace(
            go.Bar(x=list(period_counts.index), y=list(period_counts.values),
                   name="期間別作品数"),
            row=1, col=1
        )
        
        # 2. 印象派 vs 非印象派
        impressionist_counts = df['is_impressionist'].value_counts()
        fig.add_trace(
            go.Pie(labels=['非印象派', '印象派'], values=list(impressionist_counts.values),
                   name="印象派分布"),
            row=1, col=2
        )
        
        # 3. 部門別分布
        dept_counts = df['department'].value_counts().head(8)
        fig.add_trace(
            go.Bar(x=list(dept_counts.values), y=list(dept_counts.index),
                   orientation='h', name="部門別作品数"),
            row=2, col=1
        )
        
        # 4. 特徴量重要度（ダミーデータ）
        feature_names = ['mean_hue', 'mean_saturation', 'mean_value', 'contrast']
        importance_values = [0.3, 0.25, 0.2, 0.15]
        fig.add_trace(
            go.Bar(x=feature_names, y=importance_values, name="特徴量重要度"),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False, title_text="絵画様式分類ダッシュボード")
        
        # HTMLファイルとして保存
        dashboard_file = self.viz_dir / 'dashboard.html'
        fig.write_html(dashboard_file)
        
        self.logger.info(f"インタラクティブダッシュボードを保存: {dashboard_file}")
