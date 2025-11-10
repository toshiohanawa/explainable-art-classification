# -*- coding: utf-8 -*-
"""
Department="European Paintings"のみを残すフィルタリングスクリプト
既存のデータセットと画像ファイルを整理します。
"""

import pandas as pd
import json
from pathlib import Path
import sys
from datetime import datetime

# プロジェクトルートをパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.config_manager import ConfigManager
from src.utils.logger import setup_logging, get_logger

# ログ設定
config_manager = ConfigManager()
config = config_manager.get_config()
setup_logging(config)
logger = get_logger(__name__)


def filter_european_paintings():
    """Department="European Paintings"でフィルタリング"""
    logger.info("=== European Paintingsフィルタリング開始 ===")
    
    # パスの設定
    input_csv = Path("data/curated/filtered_data/paintings_complete_dataset.csv")
    output_csv = Path("data/curated/filtered_data/paintings_complete_dataset.csv")
    backup_csv = Path("data/curated/filtered_data/paintings_complete_dataset_backup.csv")
    images_dir = Path("data/curated/filtered_data/paintings_images")
    report_file = Path("docs") / "european_paintings_filter_report.txt"
    
    # 入力ファイルの存在確認
    if not input_csv.exists():
        logger.error(f"入力ファイルが見つかりません: {input_csv}")
        return False
    
    # バックアップ作成
    logger.info("バックアップを作成中...")
    import shutil
    shutil.copy2(input_csv, backup_csv)
    logger.info(f"バックアップ作成完了: {backup_csv}")
    
    # データ読み込み
    logger.info(f"データ読み込み中: {input_csv}")
    df = pd.read_csv(input_csv, encoding='utf-8')
    initial_count = len(df)
    logger.info(f"フィルタリング前: {initial_count:,}件")
    
    # Departmentの分布確認
    if 'Department' not in df.columns:
        logger.error("Department列が見つかりません")
        return False
    
    dept_counts = df['Department'].value_counts()
    logger.info("\nDepartment別件数:")
    for dept, count in dept_counts.items():
        logger.info(f"  {dept}: {count:,}件")
    
    # European Paintingsでフィルタリング
    logger.info("\nDepartment='European Paintings'でフィルタリング中...")
    filtered_df = df[df['Department'] == 'European Paintings'].copy()
    filtered_count = len(filtered_df)
    reduction_rate = ((initial_count - filtered_count) / initial_count) * 100
    
    logger.info(f"フィルタリング後: {filtered_count:,}件")
    logger.info(f"削減率: {reduction_rate:.1f}%")
    
    if filtered_count == 0:
        logger.error("フィルタリング後のデータが0件です。条件を確認してください。")
        return False
    
    # フィルタリング後のデータを保存（一時ファイル経由）
    logger.info(f"フィルタリング後のデータを保存中: {output_csv}")
    temp_file = output_csv.parent / f"{output_csv.stem}_temp{output_csv.suffix}"
    try:
        filtered_df.to_csv(temp_file, index=False, encoding='utf-8')
        # 一時ファイルを元のファイルに置き換え（ファイルが開かれている場合は別名で保存）
        if output_csv.exists():
            try:
                output_csv.unlink()  # 既存ファイルを削除
                temp_file.rename(output_csv)  # 一時ファイルをリネーム
                logger.info("データ保存完了（元のファイルを置き換えました）")
            except (PermissionError, OSError) as e:
                # ファイルが開かれている場合は別名で保存
                new_file = output_csv.parent / f"{output_csv.stem}_european_only{output_csv.suffix}"
                temp_file.rename(new_file)
                logger.warning(f"元のファイルが開かれているため、別名で保存しました: {new_file}")
                logger.warning("元のファイルを閉じた後、手動で置き換えてください:")
                logger.warning(f"  1. {output_csv} を閉じる")
                logger.warning(f"  2. {new_file} を {output_csv} にリネーム")
                output_csv = new_file  # 以降の処理で新しいファイル名を使用
        else:
            temp_file.rename(output_csv)
            logger.info("データ保存完了")
    except Exception as e:
        logger.error(f"ファイル保存中にエラー: {e}")
        if temp_file.exists():
            temp_file.unlink()  # エラー時は一時ファイルを削除
        raise
    
    # 画像ファイルの整理
    if images_dir.exists():
        logger.info("\n画像ファイルの整理を開始...")
        # フィルタリング後のObject_IDセット
        valid_object_ids = set(filtered_df['Object_ID'].astype(str))
        logger.info(f"有効なObject_ID数: {len(valid_object_ids):,}件")
        
        # 画像ファイル一覧
        image_files = list(images_dir.iterdir())
        total_images = len([f for f in image_files if f.is_file()])
        logger.info(f"総画像ファイル数: {total_images:,}件")
        
        # 不要な画像ファイルを削除
        deleted_count = 0
        kept_count = 0
        delete_errors = []
        
        for image_file in image_files:
            if not image_file.is_file():
                continue
            
            # ファイル名からObject_IDを抽出（拡張子を除く）
            object_id_str = image_file.stem
            
            if object_id_str not in valid_object_ids:
                # 不要な画像ファイルを削除
                try:
                    image_file.unlink()
                    deleted_count += 1
                except PermissionError as e:
                    delete_errors.append(f"{image_file.name}: {e}")
                    logger.warning(f"画像ファイル削除エラー: {image_file.name} - {e}")
            else:
                kept_count += 1
        
        logger.info(f"保持した画像: {kept_count:,}件")
        logger.info(f"削除した画像: {deleted_count:,}件")
        if delete_errors:
            logger.warning(f"削除に失敗した画像: {len(delete_errors)}件（画像ファイルが開かれている可能性があります）")
    else:
        logger.warning(f"画像ディレクトリが見つかりません: {images_dir}")
    
    # レポート生成
    logger.info("\nレポート生成中...")
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("European Paintings フィルタリングレポート\n")
        f.write("=" * 60 + "\n")
        f.write(f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("【フィルタリング前】\n")
        f.write(f"  総件数: {initial_count:,}件\n")
        f.write(f"  Department別件数:\n")
        for dept, count in dept_counts.items():
            f.write(f"    {dept}: {count:,}件\n")
        
        f.write("\n【フィルタリング後】\n")
        f.write(f"  総件数: {filtered_count:,}件\n")
        f.write(f"  削減率: {reduction_rate:.1f}%\n")
        
        if images_dir.exists():
            f.write("\n【画像ファイル】\n")
            f.write(f"  保持した画像: {kept_count:,}件\n")
            f.write(f"  削除した画像: {deleted_count:,}件\n")
        
        f.write("\n【データファイル】\n")
        f.write(f"  元データ（バックアップ）: {backup_csv}\n")
        f.write(f"  フィルタリング後データ: {output_csv}\n")
    
    logger.info(f"レポート保存完了: {report_file}")
    
    logger.info("\n=== フィルタリング完了 ===")
    logger.info(f"✅ フィルタリング後: {filtered_count:,}件（{reduction_rate:.1f}%削減）")
    
    return True


def main():
    """メイン関数"""
    try:
        success = filter_european_paintings()
        if success:
            print("\n✅ フィルタリングが正常に完了しました！")
            print("\n次のステップ:")
            print("  1. config.yamlのpainting_filtersを更新")
            print("  2. フィルタリング結果を確認: data/curated/filtered_data/european_paintings_filter_report.txt")
            print("  3. バックアップファイル: data/curated/filtered_data/paintings_complete_dataset_backup.csv")
        else:
            print("\n❌ フィルタリング中にエラーが発生しました。ログを確認してください。")
    except Exception as e:
        logger.error(f"予期しないエラー: {e}", exc_info=True)
        print(f"\n❌ エラーが発生しました: {e}")


if __name__ == "__main__":
    main()

