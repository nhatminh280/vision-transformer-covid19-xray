from multiprocessing.util import get_logger
import os
import sys
import argparse
import shutil
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import json
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import cv2
from PIL import Image
import yaml

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.logger import setup_logger
from src.utils.helpers import create_directories
from src.utils.config import load_config

config = load_config("configs/data/base_config.yaml")
logger = setup_logger(__name__, config.get('logging', {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'save_to_file': False
}))


class DataPreprocessor:
    """Class to handle data preprocessing and preparation."""
    
    def __init__(self, config_path: str = "configs/data/base_config.yaml"):
        self.config = load_config(config_path)
        
        # Paths
        self.raw_data_path = Path(self.config['paths']['raw_data'])
        self.processed_data_path = Path(self.config['paths']['processed_data'])
        self.metadata_path = Path(self.config['paths']['metadata'])
        
        # Dataset configuration
        self.dataset_config = self.config['dataset']
        self.classes = [
            'COVID',
            'Lung_Opacity',
            'Normal',
            'Viral_Pneumonia'
        ]
        self.train_split = self.dataset_config['train_split']
        self.val_split = self.dataset_config['val_split']
        self.test_split = self.dataset_config['test_split']
        
        # Image configuration
        self.image_size = self.dataset_config['image_size']
        self.channels = self.dataset_config['channels']
        
        # Statistics
        self.dataset_stats = {}
        
    def create_directories(self, force: bool = False):
        """Create necessary directories for processed data."""
        logger.info("Creating directories...")
        
        # Check if processed directory exists
        if self.processed_data_path.exists():
            if force:
                logger.warning(f"Removing existing processed data directory: {self.processed_data_path}")
                import shutil
                shutil.rmtree(self.processed_data_path)
            else:
                logger.warning("Processed data directory already exists. Use --force to overwrite.")
                return False
        
        # Create main directories
        create_directories([
            self.processed_data_path,
            self.metadata_path
        ])
        
        # Create split directories with images and masks subdirectories
        splits = ['train', 'val', 'test']
        for split in splits:
            split_path = self.processed_data_path / split
            
            # Create class directories within each split
            for class_name in self.classes:
                class_path = split_path / class_name
                images_path = class_path / "images"
                masks_path = class_path / "masks"
                create_directories([images_path, masks_path])
    
        return True
    
    def preprocess_image(self, image_path: str, output_path: str) -> bool:
        """Preprocess a single image for ViT."""
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                logger.error(f"Could not read image: {image_path}")
                return False
                
            # Convert BGR to RGB
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Center crop (if aspect ratio is different)
            h, w = img.shape[:2]
            min_dim = min(h, w)
            top = (h - min_dim) // 2
            left = (w - min_dim) // 2
            img = img[top:top+min_dim, left:left+min_dim]
            
            # Resize to ViT input size
            img = cv2.resize(img, (self.image_size, self.image_size))
            
            # Normalize pixel values to [0, 1] range s
            img = img.astype(np.float32) / 255.0
            
            # Ensure 3 channels
            if len(img.shape) == 2:
                img = np.stack([img] * 3, axis=-1)
            
            # Extract patches
            patches = self.extract_patches(img)
            
            # Save as structured array with patches and positions
            data = {
                'image': img,
                'patches': patches,
                'patch_size': 16,
                'n_patches': patches.shape[0]
            }
            np.savez_compressed(
                str(output_path).replace('.png', '.npz'),
                **data
            )
            return True
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            return False
            
    def copy_and_preprocess_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """Copy and preprocess images to processed data directory."""
        logger.info("Copying and preprocessing images...")
        
        splits = {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }
        
        for split_name, df in splits.items():
            logger.info(f"Processing {split_name} split...")
            
            for idx, row in df.iterrows():
                # Source and destination paths
                src_path = row['file_path']
                dst_dir = self.processed_data_path / split_name / row['class']
                dst_path = dst_dir / row['filename']
                
                # Preprocess and save image
                if not self.preprocess_image(src_path, str(dst_path)):
                    logger.error(f"Failed to preprocess: {src_path}")
                    
                # Update file path in dataframe
                df.at[idx, 'processed_path'] = str(dst_path).replace('.png', '.npz')
        
        # Save updated metadata after preprocessing
        self.save_metadata(train_df, val_df, test_df)
                
    def save_metadata(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """Save metadata and statistics."""
        logger.info("Saving metadata...")
        
        # Save split metadata
        train_df.to_csv(self.metadata_path / "train_metadata.csv", index=False)
        val_df.to_csv(self.metadata_path / "val_metadata.csv", index=False)
        test_df.to_csv(self.metadata_path / "test_metadata.csv", index=False)
        
        # Save dataset statistics
        with open(self.metadata_path / "dataset_stats.json", "w") as f:
            json.dump(self.dataset_stats, f, indent=2)
            
        # Save class mapping
        class_mapping = {
            'classes': self.classes,
            'class_to_id': {cls: idx for idx, cls in enumerate(self.classes)},
            'id_to_class': {idx: cls for idx, cls in enumerate(self.classes)}
        }
        
        with open(self.metadata_path / "class_mapping.json", "w") as f:
            json.dump(class_mapping, f, indent=2)
            
        logger.info("Metadata saved successfully!")
        
    def calculate_statistics(self, train_df: pd.DataFrame):
        """Calculate dataset statistics."""
        logger.info("Calculating dataset statistics...")
        
        # Class distribution
        class_counts = train_df['class'].value_counts().to_dict()
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(train_df['class']),
            y=train_df['class']
        )
        
        self.dataset_stats = {
            'class_counts': class_counts,
            'class_weights': dict(zip(self.classes, class_weights)),
            'pixel_mean': [0.485, 0.456, 0.406], 
            'pixel_std': [0.229, 0.224, 0.225],  
            'num_samples': {
                'train': len(train_df),
                'total': len(train_df)
            },
            'patch_info': {
                'patch_size': 16,
                'num_patches': (self.image_size // 16) ** 2,
                'patch_dim': 3 * 16 * 16
            }
        }
        
    def generate_summary_report(self):
        """Generate a summary report of the data preparation."""
        logger.info("Generating summary report...")
        
        report = {
            'dataset_info': {
                'name': self.dataset_config['name'],
                'num_classes': len(self.classes),
                'classes': self.classes,
                'image_size': self.image_size,
                'channels': self.channels
            },
            
            'data_splits': {
                'train_split': self.train_split,
                'val_split': self.val_split,
                'test_split': self.test_split
            },
            'statistics': self.dataset_stats
        }
        
        # Save report
        with open(self.metadata_path / "data_preparation_report.json", "w") as f:
            json.dump(report, f, indent=2)
            
        # Print summary
        logger.info("=== DATA PREPARATION SUMMARY ===")
        logger.info(f"Dataset: {self.dataset_config['name']}")
        logger.info(f"Classes: {', '.join(self.classes)}")
        logger.info(f"Image size: {self.image_size}x{self.image_size}")
        logger.info(f"Train samples: {self.dataset_stats['num_samples']['train']}")
        logger.info("Class distribution:")
        for class_name, count in self.dataset_stats['class_counts'].items():
            logger.info(f"  {class_name}: {count}")
            
    def apply_vit_augmentation(self, img: np.ndarray) -> np.ndarray:
        """Apply ViT-specific augmentations."""
        if not self.dataset_config['augmentation']['enabled']:
            return img
            
        # Random horizontal flip
        if self.dataset_config['augmentation']['horizontal_flip']:
            if np.random.random() > 0.5:
                img = np.fliplr(img)
        
        # Random color jittering
        if np.random.random() > 0.8:
            img = img * np.random.uniform(0.8, 1.2)
            img = np.clip(img, 0, 1)  # Clip to [0,1] instead of [-1,1]
        
        # Random patch dropping (simulating attention dropout)
        if np.random.random() > 0.9:
            patch_size = 16
            num_patches = (self.image_size // patch_size) ** 2
            num_drop = int(0.1 * num_patches)
            
            for _ in range(num_drop):
                px = np.random.randint(0, self.image_size - patch_size)
                py = np.random.randint(0, self.image_size - patch_size)
                img[px:px+patch_size, py:py+patch_size] = 0
                
        return img

    def add_position_embedding(self, patches: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Add position embeddings to patches."""
        n_patches = patches.shape[0]
        # +1 for CLS token
        position_embeddings = np.arange(n_patches + 1)
        return patches, position_embeddings
    
    def extract_patches(self, img: np.ndarray, patch_size: int = 16) -> np.ndarray:
        """Extract patches from image for ViT preprocessing."""
        h, w, c = img.shape
        patches = []
        
        for i in range(0, h, patch_size):
            for j in range(0, w, patch_size):
                patch = img[i:i+patch_size, j:j+patch_size]
                # Flatten patch for ViT: (16,16,3) -> (768,)
                patch_flat = patch.flatten()
                patches.append(patch_flat)
                
        return np.array(patches)  # Shape: (num_patches, 768)
    
    def _check_processed_data_exists(self) -> bool:
        """Check if processed data already exists."""
        splits = ['train', 'val', 'test']
        for split in splits:
            split_path = self.processed_data_path / split
            if not split_path.exists():
                return False
            # Check if directories have content
            for class_name in self.classes:
                class_path = split_path / class_name
                if not class_path.exists() or not any(class_path.iterdir()):
                    return False
        return True
    
    def prepare_data(self):
        """Main method to prepare the dataset."""
        logger.info("Starting data preparation...")
        
        # Create directories
        self.create_directories()
        
        try:
            all_train_dfs = []
            all_val_dfs = []
            all_test_dfs = []
            
            # Process each class separately
            for class_name in self.classes:
                logger.info(f"Processing class: {class_name}")
                
                # Load and process class metadata from xlsx
                try:
                    train_df, val_df, test_df = self.process_class_metadata(class_name)
                    
                    # Create class-specific directories in splits
                    for split in ['train', 'val', 'test']:
                        class_dir = self.processed_data_path / split / class_name
                        create_directories([class_dir])
                    
                    # Process images for this class
                    self.copy_and_preprocess_class_data(class_name, train_df, val_df, test_df)
                    
                    # Store DataFrames
                    all_train_dfs.append(train_df)
                    all_val_dfs.append(val_df)
                    all_test_dfs.append(test_df)
                    
                except Exception as e:
                    logger.error(f"Error processing class {class_name}: {str(e)}")
                    continue
            
            # Combine all splits for statistics
            final_train_df = pd.concat(all_train_dfs, ignore_index=True)
            final_val_df = pd.concat(all_val_dfs, ignore_index=True)
            final_test_df = pd.concat(all_test_dfs, ignore_index=True)
            
            # Calculate and save statistics
            self.calculate_statistics(final_train_df)
            self.save_metadata(final_train_df, final_val_df, final_test_df)
                
        except Exception as e:
            logger.error(f"Error processing data: {str(e)}")
            raise
        
        # Generate summary report
        self.generate_summary_report()
        logger.info("Data preparation completed successfully!")
    
    def _validate_metadata(self, train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame):
        """Validate loaded metadata format."""
        required_columns = ['file_path', 'class', 'class_id', 'filename']
        
        for df, split in [(train_df, 'train'), (val_df, 'val'), (test_df, 'test')]:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                raise ValueError(f"Missing columns in {split} metadata: {missing_cols}")
                
            # Validate classes
            if not all(cls in self.classes for cls in df['class'].unique()):
                raise ValueError(f"Invalid classes found in {split} metadata")
                
    def _update_file_paths(self, df: pd.DataFrame):
        """Update file paths in metadata if necessary."""
        df['file_path'] = df['file_path'].apply(
            lambda x: str(Path(x).resolve())
        )
        return df

    def split_metadata(self, metadata_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split metadata into train/val/test sets."""
        logger.info("Splitting metadata into train/val/test sets...")
        
        # First split into train and temp (val + test)
        train_df, temp_df = train_test_split(
            metadata_df, 
            train_size=self.train_split,
            stratify=metadata_df['class'],
            random_state=42
        )
        
        # Then split temp into val and test
        val_size = self.val_split / (self.val_split + self.test_split)
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_size,
            stratify=temp_df['class'],
            random_state=42
        )
        
        logger.info(f"Split sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        return train_df, val_df, test_df
    
    def process_class_metadata(self, class_name: str):
        """Process metadata for a single class."""
        logger.info(f"Processing metadata for class: {class_name}")
        
        # Load class-specific metadata
        excel_path = self.metadata_path / f"{class_name}.metadata.xlsx"
        if not excel_path.exists():
            raise FileNotFoundError(f"Excel metadata file not found: {excel_path}")

        # Read Excel file 
        df = pd.read_excel(excel_path)
        
        # Print columns for debugging
        logger.info(f"Available columns: {df.columns.tolist()}")
        
        # Check for 'FILE NAME' column
        if 'FILE NAME' in df.columns:
            df['filename'] = df['FILE NAME'].apply(lambda x: f"{x}.png")
        else:
            raise ValueError(f"Excel file must contain 'FILE NAME' column")

        # Construct file paths
        raw_data_dir = self.raw_data_path / "COVID-19_Radiography_Dataset"
        
        # Add file paths
        df['file_path'] = df['filename'].apply(
            lambda x: str(raw_data_dir / class_name / "images" / x)
        )
        df['mask_path'] = df['filename'].apply(
            lambda x: str(raw_data_dir / class_name / "masks" / x)
        )
        
        # Add class information 
        df['class'] = class_name
        df['class_id'] = self.classes.index(class_name)
        
        # Verify files exist
        valid_files = []
        for idx, row in df.iterrows():
            if Path(row['file_path']).exists():
                valid_files.append(idx)
            else:
                logger.warning(f"Image file not found: {row['file_path']}")
        
        # Keep only valid files
        df = df.loc[valid_files]
        
        if len(df) == 0:
            raise ValueError(f"No valid files found for class {class_name}")
        
        # Split into train/val/test
        train_df, temp_df = train_test_split(
            df, 
            train_size=self.train_split,
            random_state=42
        )
        
        val_size = self.val_split / (self.val_split + self.test_split)
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_size,
            random_state=42
        )
        
        logger.info(f"Split sizes for {class_name}: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
        return train_df, val_df, test_df
 
    def copy_and_preprocess_class_data(self, class_name: str, train_df: pd.DataFrame, 
                             val_df: pd.DataFrame, test_df: pd.DataFrame):
        """Process images for a specific class."""
        logger.info(f"Processing images for class: {class_name}")

        splits = {
            'train': train_df,
            'val': val_df,
            'test': test_df
        }

        for split_name, df in splits.items():
            # Filter data for current class
            class_df = df[df['class'] == class_name]
            
            # Create directory structure for both images and masks
            dst_base_dir = self.processed_data_path / split_name / class_name
            images_dir = dst_base_dir / "images"
            masks_dir = dst_base_dir / "masks"
            create_directories([images_dir, masks_dir])
            
            for idx, row in class_df.iterrows():
                try:
                    # Source paths
                    img_src = row['file_path']
                    mask_src = row['mask_path']
                    
                    # Destination paths
                    img_dst = images_dir / row['filename']
                    mask_dst = masks_dir / row['filename']
                    
                    # Preprocess and save image
                    if not self.preprocess_image(img_src, str(img_dst)):
                        logger.error(f"Failed to preprocess image: {img_src}")
                        continue
                        
                    # Copy mask file (without preprocessing)
                    shutil.copy2(mask_src, str(mask_dst))
                    
                    # Update file paths in dataframe
                    df.at[idx, 'processed_image_path'] = str(img_dst)
                    df.at[idx, 'processed_mask_path'] = str(mask_dst)
                    
                except Exception as e:
                    logger.error(f"Error processing files for {row['filename']}: {str(e)}")
                    continue


def main():
    """Main function to run data preparation."""
    parser = argparse.ArgumentParser(description="Prepare COVID-19 Radiography Dataset")
    parser.add_argument(
        "--config",
        default="configs/data/base_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing processed data"
    )
    
    args = parser.parse_args()
    
    # Check if processed data already exists
    config = load_config(args.config)
    processed_path = Path(config['paths']['processed_data'])
    
    if processed_path.exists() and not args.force:
        logger.warning(f"Processed data already exists at: {processed_path}")
        logger.warning("Use --force to overwrite existing data")
        response = input("Continue anyway? (y/N): ")
        if response.lower() != 'y':
            logger.info("Exiting...")
            return
            
    # Initialize preprocessor
    preprocessor = DataPreprocessor(config_path=args.config)
    
    # Prepare data
    try:
        preprocessor.prepare_data()
    except Exception as e:
        logger.error(f"Data preparation failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()