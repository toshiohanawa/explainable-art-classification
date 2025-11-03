# -*- coding: utf-8 -*-
"""
MetObjects.csvを分析して絵画関連のフィルター条件を特定するスクリプト
"""

import sys
from pathlib import Path

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from collections import Counter
from src.utils.config_manager import ConfigManager
from src.utils.logger import setup_logging, get_logger


class PaintingDataAnalyzer:
    """絵画データ分析クラス"""
    
    def __init__(self, csv_path: str = "data/raw_data/MetObjects.csv"):
        self.csv_path = Path(csv_path)
        self.df = None
        self.results = {}
        self.logger = get_logger(__name__)
        self.output_file = Path("docs") / "painting_analysis_results.txt"
        
    def load_data(self):
        """CSVデータを読み込み"""
        self.logger.info(f"CSVデータを読み込み中: {self.csv_path}")
        try:
            self.df = pd.read_csv(self.csv_path, low_memory=False)
            self.logger.info(f"データ読み込み完了: {len(self.df):,}件")
            return True
        except Exception as e:
            self.logger.error(f"データ読み込みエラー: {e}")
            return False
    
    def analyze_departments(self):
        """Department別の分析"""
        self.logger.info("Department別の分析を開始...")
        
        dept_counts = self.df['Department'].value_counts()
        self.results['departments'] = dept_counts.to_dict()
        
        # 絵画関連のDepartmentを特定
        painting_keywords = ['paint', 'art', 'european', 'american', 'modern', 'contemporary']
        painting_departments = []
        
        for dept in dept_counts.index:
            if pd.isna(dept):
                continue
            dept_lower = str(dept).lower()
            if any(keyword in dept_lower for keyword in painting_keywords):
                painting_departments.append(dept)
        
        self.results['painting_departments'] = painting_departments
        
        self.logger.info(f"Department総数: {len(dept_counts)}")
        self.logger.info(f"絵画関連Department: {len(painting_departments)}")
        for dept in painting_departments[:10]:  # 上位10件表示
            count = dept_counts[dept]
            self.logger.info(f"  {dept}: {count:,}件")
    
    def analyze_classifications(self):
        """Classification別の分析"""
        self.logger.info("Classification別の分析を開始...")
        
        class_counts = self.df['Classification'].value_counts()
        self.results['classifications'] = class_counts.to_dict()
        
        # 絵画関連のClassificationを特定
        painting_keywords = ['paint', 'canvas', 'oil', 'watercolor', 'tempera', 'fresco']
        painting_classifications = []
        
        for classification in class_counts.index:
            if pd.isna(classification):
                continue
            class_lower = str(classification).lower()
            if any(keyword in class_lower for keyword in painting_keywords):
                painting_classifications.append(classification)
        
        self.results['painting_classifications'] = painting_classifications
        
        self.logger.info(f"Classification総数: {len(class_counts)}")
        self.logger.info(f"絵画関連Classification: {len(painting_classifications)}")
        for classification in painting_classifications[:10]:  # 上位10件表示
            count = class_counts[classification]
            self.logger.info(f"  {classification}: {count:,}件")
    
    def analyze_object_names(self):
        """Object Name別の分析"""
        self.logger.info("Object Name別の分析を開始...")
        
        # 絵画関連キーワードでフィルタリング
        painting_keywords = [
            'painting', 'canvas', 'oil', 'watercolor', 'tempera', 'fresco',
            'portrait', 'landscape', 'still life', 'genre', 'history',
            'panel', 'canvas', 'board', 'copper', 'wood'
        ]
        
        # Object Nameが存在するレコードのみ
        valid_names = self.df['Object Name'].dropna()
        painting_objects = []
        
        for name in valid_names:
            name_lower = str(name).lower()
            if any(keyword in name_lower for keyword in painting_keywords):
                painting_objects.append(name)
        
        # 絵画関連Object Nameの件数
        painting_name_counts = Counter(painting_objects)
        self.results['painting_object_names'] = dict(painting_name_counts.most_common(50))
        
        self.logger.info(f"絵画関連Object Name: {len(painting_objects):,}件")
        self.logger.info("上位20件:")
        for name, count in painting_name_counts.most_common(20):
            self.logger.info(f"  {name}: {count:,}件")
    
    def analyze_mediums(self):
        """Medium別の分析"""
        self.logger.info("Medium別の分析を開始...")
        
        medium_counts = self.df['Medium'].value_counts()
        self.results['mediums'] = medium_counts.to_dict()
        
        # 絵画関連のMediumを特定
        painting_keywords = [
            'oil', 'canvas', 'watercolor', 'tempera', 'fresco', 'gouache',
            'acrylic', 'pastel', 'charcoal', 'ink', 'pencil', 'brush'
        ]
        painting_mediums = []
        
        for medium in medium_counts.index:
            if pd.isna(medium):
                continue
            medium_lower = str(medium).lower()
            if any(keyword in medium_lower for keyword in painting_keywords):
                painting_mediums.append(medium)
        
        self.results['painting_mediums'] = painting_mediums
        
        self.logger.info(f"Medium総数: {len(medium_counts)}")
        self.logger.info(f"絵画関連Medium: {len(painting_mediums)}")
        for medium in painting_mediums[:15]:  # 上位15件表示
            count = medium_counts[medium]
            self.logger.info(f"  {medium}: {count:,}件")
    
    def analyze_cultures(self):
        """Culture別の分析"""
        self.logger.info("Culture別の分析を開始...")
        
        culture_counts = self.df['Culture'].value_counts()
        self.results['cultures'] = culture_counts.to_dict()
        
        # 絵画が盛んな文化圏を特定
        painting_cultures = [
            'European', 'American', 'French', 'Italian', 'Dutch', 'Flemish',
            'German', 'Spanish', 'British', 'Russian', 'Japanese', 'Chinese'
        ]
        
        painting_culture_counts = {}
        for culture in painting_cultures:
            if culture in culture_counts.index:
                painting_culture_counts[culture] = culture_counts[culture]
        
        self.results['painting_cultures'] = painting_culture_counts
        
        self.logger.info("絵画関連Culture（上位15件）:")
        for culture, count in sorted(painting_culture_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
            self.logger.info(f"  {culture}: {count:,}件")
    
    def test_filter_combinations(self):
        """フィルター条件の組み合わせをテスト"""
        self.logger.info("フィルター条件の組み合わせをテスト中...")
        
        # 基本フィルター条件
        filters = {
            'department': self.results.get('painting_departments', []),
            'classification': self.results.get('painting_classifications', []),
            'medium': self.results.get('painting_mediums', [])
        }
        
        # 各フィルター単体での効果
        for filter_name, filter_values in filters.items():
            if not filter_values:
                continue
                
            if filter_name == 'department':
                filtered_df = self.df[self.df['Department'].isin(filter_values)]
            elif filter_name == 'classification':
                filtered_df = self.df[self.df['Classification'].isin(filter_values)]
            elif filter_name == 'medium':
                filtered_df = self.df[self.df['Medium'].isin(filter_values)]
            
            count = len(filtered_df)
            percentage = (count / len(self.df)) * 100
            self.logger.info(f"{filter_name}フィルター: {count:,}件 ({percentage:.1f}%)")
        
        # 複合フィルターのテスト
        self.logger.info("\n複合フィルターのテスト:")
        
        # Department + Classification
        if filters['department'] and filters['classification']:
            dept_class_df = self.df[
                (self.df['Department'].isin(filters['department'])) &
                (self.df['Classification'].isin(filters['classification']))
            ]
            count = len(dept_class_df)
            percentage = (count / len(self.df)) * 100
            self.logger.info(f"Department + Classification: {count:,}件 ({percentage:.1f}%)")
        
        # Department + Medium
        if filters['department'] and filters['medium']:
            dept_medium_df = self.df[
                (self.df['Department'].isin(filters['department'])) &
                (self.df['Medium'].isin(filters['medium']))
            ]
            count = len(dept_medium_df)
            percentage = (count / len(self.df)) * 100
            self.logger.info(f"Department + Medium: {count:,}件 ({percentage:.1f}%)")
        
        # 3つの条件すべて
        if all(filters.values()):
            all_filters_df = self.df[
                (self.df['Department'].isin(filters['department'])) &
                (self.df['Classification'].isin(filters['classification'])) &
                (self.df['Medium'].isin(filters['medium']))
            ]
            count = len(all_filters_df)
            percentage = (count / len(self.df)) * 100
            self.logger.info(f"全条件: {count:,}件 ({percentage:.1f}%)")
            
            # 最適なフィルター条件を保存
            self.results['recommended_filters'] = {
                'departments': filters['department'][:5],  # 上位5件
                'classifications': filters['classification'][:3],  # 上位3件
                'mediums': filters['medium'][:10],  # 上位10件
                'expected_count': count,
                'percentage': percentage
            }
    
    def save_results(self, output_file: str = None):
        """分析結果をファイルに保存"""
        if output_file is None:
            output_file = Path("docs") / "painting_analysis_results.txt"
        else:
            output_file = Path(output_file)
        self.logger.info(f"分析結果を保存中: {output_file}")
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=== 絵画データ分析結果 ===\n\n")
            
            # Department分析結果
            f.write("## Department分析\n")
            f.write(f"総数: {len(self.results.get('departments', {}))}\n")
            f.write("絵画関連Department:\n")
            for dept in self.results.get('painting_departments', [])[:10]:
                count = self.results['departments'].get(dept, 0)
                f.write(f"  {dept}: {count:,}件\n")
            f.write("\n")
            
            # Classification分析結果
            f.write("## Classification分析\n")
            f.write(f"総数: {len(self.results.get('classifications', {}))}\n")
            f.write("絵画関連Classification:\n")
            for classification in self.results.get('painting_classifications', [])[:10]:
                count = self.results['classifications'].get(classification, 0)
                f.write(f"  {classification}: {count:,}件\n")
            f.write("\n")
            
            # Medium分析結果
            f.write("## Medium分析\n")
            f.write(f"総数: {len(self.results.get('mediums', {}))}\n")
            f.write("絵画関連Medium:\n")
            for medium in self.results.get('painting_mediums', [])[:15]:
                count = self.results['mediums'].get(medium, 0)
                f.write(f"  {medium}: {count:,}件\n")
            f.write("\n")
            
            # 推奨フィルター条件
            if 'recommended_filters' in self.results:
                f.write("## 推奨フィルター条件\n")
                rec = self.results['recommended_filters']
                f.write(f"予想件数: {rec['expected_count']:,}件 ({rec['percentage']:.1f}%)\n")
                f.write(f"Department: {rec['departments']}\n")
                f.write(f"Classification: {rec['classifications']}\n")
                f.write(f"Medium: {rec['mediums']}\n")
    
    def run_analysis(self):
        """分析を実行"""
        self.logger.info("=== 絵画データ分析を開始 ===")
        
        if not self.load_data():
            return False
        
        self.analyze_departments()
        self.analyze_classifications()
        self.analyze_object_names()
        self.analyze_mediums()
        self.analyze_cultures()
        self.test_filter_combinations()
        self.save_results()
        
        self.logger.info("=== 分析完了 ===")
        return True


def main():
    """メイン関数"""
    # 設定読み込み
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    # ログ設定
    setup_logging(
        log_file='csv_analysis.log',
        config=config
    )
    
    analyzer = PaintingDataAnalyzer()
    success = analyzer.run_analysis()
    
    if success:
        print(f"\n分析が完了しました。結果は '{analyzer.output_file}' に保存されています。")
    else:
        print("分析中にエラーが発生しました。ログを確認してください。")


if __name__ == "__main__":
    main()

