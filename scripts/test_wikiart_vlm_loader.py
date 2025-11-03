"""
WikiArt_VLM Data Loader Test Script
Test dataset loading and statistics verification
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_collection.wikiart_vlm_loader import WikiArtVLMDataLoader
from src.utils.config_manager import ConfigManager
from src.utils.logger import setup_logging


def main():
    """Main execution function"""
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.get_config()
    
    # Setup logging
    setup_logging(config=config)
    
    # Create data loader
    loader = WikiArtVLMDataLoader(config)
    
    print("=" * 60)
    print("WikiArt_VLM Dataset Statistics")
    print("=" * 60)
    
    # Get dataset statistics for each generation model
    for model in ['Stable-Diffusion', 'FLUX', 'F-Lite']:
        print(f"\n[{model}]")
        try:
            stats = loader.get_dataset_stats(model)
            print(f"  Total images: {stats['total_images']}")
            print(f"  Original images: {stats['original_count']}")
            print(f"  Generated images: {stats['generated_count']}")
            print(f"  Unique image IDs: {stats['image_ids']}")
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n" + "=" * 60)
    print("Dataset Integrity Check")
    print("=" * 60)
    
    # Integrity check for Stable-Diffusion
    print("\n[Stable-Diffusion]")
    try:
        checks = loader.verify_dataset_integrity('Stable-Diffusion')
        print(f"  Total pairs: {checks['total_pairs']}")
        print(f"  Has original images: {checks['has_original_images']}")
        print(f"  Has generated images: {checks['has_generated_images']}")
        print(f"  Balanced dataset: {checks['balanced_dataset']}")
        print(f"  Sample images exist: {checks['sample_images_exist']}")
        
        if checks['images_exist']:
            print("\n  Sample image existence check:")
            for check_item in checks['images_exist'][:5]:  # Show first 5 only
                status = "[OK]" if check_item['exists'] else "[NG]"
                print(f"    {status} {check_item['image_id']}: {check_item['exists']}")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n" + "=" * 60)
    print("Dataset Split Test")
    print("=" * 60)
    
    # Load and split dataset
    try:
        df = loader.load_image_pairs('Stable-Diffusion')
        train_df, val_df, test_df = loader.split_dataset(df)
        
        print(f"\n  Total data: {len(df)}")
        print(f"  Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
        print(f"  Val: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
        print(f"  Test: {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n" + "=" * 60)
    print("Prompt Information Loading")
    print("=" * 60)
    
    # Load prompt information
    try:
        prompts_df = loader.load_prompts()
        if len(prompts_df) > 0:
            print(f"\n  Number of prompts: {len(prompts_df)}")
            print(f"  Columns: {list(prompts_df.columns)}")
            print("\n  Sample prompts (first 3):")
            for idx, row in prompts_df.head(3).iterrows():
                print(f"    {idx}: {str(row)[:100]}...")
        else:
            print("\n  Prompt file not found")
    except Exception as e:
        print(f"  Error: {e}")
    
    print("\n" + "=" * 60)
    print("Test Completed")
    print("=" * 60)


if __name__ == "__main__":
    main()

