# To organize
import os
import shutil
from sklearn.model_selection import train_test_split 

def create_and_clear_directory(directory_path):
    """Ensures a directory exists and is empty."""
    if os.path.exists(directory_path):
        for item in os.listdir(directory_path):
            item_path = os.path.join(directory_path, item)
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        print(f"Cleared contents of: {directory_path}")
    else:
        os.makedirs(directory_path, exist_ok=True)
        print(f"Created directory: {directory_path}")


def organize_and_split_frames(source_root_folder, project_root_folder, train_ratio=0.8):
    """
    Moves images from source desktop folders into project's train/validation splits.

    Args:
        source_root_folder (str): Absolute path to the desktop folder containing
                                  'cat-present' and 'no-cat' subfolders.
        project_root_folder (str): Absolute or relative path to project's root.
                                   (e.g., '.' if running from project root).
        train_ratio (float): Proportion of data to use for training (e.g., 0.8 for 80%).
    """

    # Define the classes based on source folders and target folders
    # The order here defines the mapping from source to target
    class_mappings = {
        "cat-present": "cat-present",
        "no-cat": "no-cat"
    }

    # Define target paths relative to the project root
    labeled_frames_dir = os.path.join(project_root_folder, 'labeled-frames')
    train_dir = os.path.join(labeled_frames_dir, 'train')
    validation_dir = os.path.join(labeled_frames_dir, 'validation')

    # Create and clear destination folders to ensure a fresh start
    for class_folder_name in class_mappings.values():
        create_and_clear_directory(os.path.join(train_dir, class_folder_name))
        create_and_clear_directory(os.path.join(validation_dir, class_folder_name))


    print("\nStarting to organize and split frames...")

    for source_class_folder_name, dest_class_folder_name in class_mappings.items():
        source_path = os.path.join(source_root_folder, source_class_folder_name)

        if not os.path.exists(source_path):
            print(f"Warning: Source folder not found for '{source_class_folder_name}': {source_path}. Skipping.")
            continue

        all_images_in_class = [f for f in os.listdir(source_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not all_images_in_class:
            print(f"No image files found in {source_path}, skipping.")
            continue

        # Split images for this class into train and validation sets
        # test_size is 1 - train_ratio
        train_images, validation_images = train_test_split(
            all_images_in_class,
            test_size=(1 - train_ratio),
            random_state=42, # for reproducibility
            shuffle=True # important for random distribution
        )

        print(f"\nProcessing class: '{source_class_folder_name}'")
        print(f"  Total images: {len(all_images_in_class)}")
        print(f"  Training images: {len(train_images)}")
        print(f"  Validation images: {len(validation_images)}")

        # Define specific destination paths for this class's splits
        current_train_dest_path = os.path.join(train_dir, dest_class_folder_name)
        current_validation_dest_path = os.path.join(validation_dir, dest_class_folder_name)

        # Move train images
        for img_name in train_images:
            source_img_path = os.path.join(source_path, img_name)
            dest_img_path = os.path.join(current_train_dest_path, img_name)
            try:
                shutil.move(source_img_path, dest_img_path)
            except Exception as e:
                print(f"Error moving {source_img_path} to {dest_img_path}: {e}")

        # Move validation images
        for img_name in validation_images:
            source_img_path = os.path.join(source_path, img_name)
            dest_img_path = os.path.join(current_validation_dest_path, img_name)
            try:
                shutil.move(source_img_path, dest_img_path)
            except Exception as e:
                print(f"Error moving {source_img_path} to {dest_img_path}: {e}")

    print("\nFinished organizing and splitting frames.")
    print("project's 'labeled-frames' directory is now populated.")

# --- Configuration ---
desktop_frames_source_root = r"C:\Users\glori_7afg9d\Videos\cat-frames"

# Path to project's root folder
# Use '.' if running this script from the project's root directory
# Otherwise, provide the absolute path or correct relative path
project_root = "." # Assuming run this script from BLINK project root

# --- Execute the function ---
if __name__ == "__main__":
    organize_and_split_frames(desktop_frames_source_root, project_root, train_ratio=0.8)