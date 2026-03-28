import os
import shutil
from pathlib import Path

def convert_cub_dataset():
    """
    Convert CUB_200_2011 dataset from its original format to train/val structure
    """
    # Paths
    base_path = Path('datasets/CUB_200_2011')
    images_dir = base_path / 'images'
    output_dir = Path('datasets/CUB_200_2011_formatted')
    
    # Read mapping files
    print("Reading dataset metadata...")
    
    # Read images.txt (image_id -> image_path)
    images_dict = {}
    with open(base_path / 'images.txt', 'r') as f:
        for line in f:
            img_id, img_path = line.strip().split()
            images_dict[img_id] = img_path
    
    # Read train_test_split.txt (image_id -> is_training)
    split_dict = {}
    with open(base_path / 'train_test_split.txt', 'r') as f:
        for line in f:
            img_id, is_training = line.strip().split()
            split_dict[img_id] = int(is_training)  # 1 = train, 0 = val/test
    
    # Read classes.txt (class_id -> class_name)
    classes_dict = {}
    with open(base_path / 'classes.txt', 'r') as f:
        for line in f:
            class_id, class_name = line.strip().split(' ', 1)
            classes_dict[class_id] = class_name
    
    print(f"Found {len(images_dict)} images across {len(classes_dict)} classes")
    
    # Create output directories
    train_dir = output_dir / 'train'
    val_dir = output_dir / 'val'
    
    print("Creating directory structure...")
    for class_id, class_name in classes_dict.items():
        (train_dir / class_name).mkdir(parents=True, exist_ok=True)
        (val_dir / class_name).mkdir(parents=True, exist_ok=True)
    
    # Copy images to appropriate directories
    print("Organizing images...")
    train_count = 0
    val_count = 0
    
    for img_id, img_path in images_dict.items():
        # Get source path
        src_path = images_dir / img_path
        
        # Extract class name from path (e.g., "001.Black_footed_Albatross/...")
        class_name = img_path.split('/')[0]
        img_filename = img_path.split('/')[1]
        
        # Determine train or val
        is_training = split_dict[img_id]
        
        if is_training == 1:
            dest_path = train_dir / class_name / img_filename
            train_count += 1
        else:
            dest_path = val_dir / class_name / img_filename
            val_count += 1
        
        # Copy the image
        shutil.copy2(src_path, dest_path)
        
        # Progress indicator
        if (train_count + val_count) % 500 == 0:
            print(f"Processed {train_count + val_count} images...")
    
    print(f"\nConversion complete!")
    print(f"Train images: {train_count}")
    print(f"Val images: {val_count}")
    print(f"Output directory: {output_dir.absolute()}")

if __name__ == '__main__':
    convert_cub_dataset()
