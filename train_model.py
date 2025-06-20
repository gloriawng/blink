# Train pytorch model
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
import os
import time
import copy
import warnings

# Suppress harmless UserWarnings from Pillow or torchvision, if any
warnings.filterwarnings("ignore", category=UserWarning)

def train_model(model, dataloaders, dataset_sizes, criterion, optimizer, num_epochs=25):
    """
    Trains a PyTorch model and evaluates it on a validation set.

    Args:
        model: The PyTorch model to train.
        dataloaders (dict): Dictionary of DataLoaders for 'train' and 'validation' phases.
        dataset_sizes (dict): Dictionary of dataset sizes for 'train' and 'validation' phases.
        criterion: The loss function (e.g., nn.CrossEntropyLoss).
        optimizer: The optimization algorithm (e.g., optim.Adam).
        num_epochs (int): The number of training epochs.

    Returns:
        model: The trained model with the best validation accuracy.
    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    # Ensure device is set (GPU if available, otherwise CPU)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device) # Move model to the selected device

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device) # Move inputs to device
                labels = labels.to(device) # Move labels to device

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                # Track history only if in training phase
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1) # Get the predicted class
                    loss = criterion(outputs, labels)

                    # Backward pass + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # Deep copy the model if it's the best validation accuracy
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print() # Newline for better readability between epochs

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation Acc: {best_acc:.4f}')

    # Load best model weights
    model.load_state_dict(best_model_wts)
    return model

if __name__ == "__main__":
    # --- Configuration ---
    # Path to your project's root directory. Assuming this script is in the root.
    data_dir = './labeled-frames'
    model_save_path = 'cat_presence_classifier.pth' # Name for your saved model file

    # --- 1. Define Data Transformations ---
    # These are standard transformations for pre-trained models like ResNet
    # ImageNet normalization values are used as the pre-trained model was trained on ImageNet
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224), # Randomly crop and resize to 224x224
            transforms.RandomHorizontalFlip(), # Randomly flip images horizontally for augmentation
            transforms.ToTensor(),             # Convert PIL Image or NumPy array to PyTorch Tensor
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # Normalize with ImageNet stats
        ]),
        'validation': transforms.Compose([
            transforms.Resize(256),            # Resize the shorter side to 256
            transforms.CenterCrop(224),        # Crop the central 224x224 portion
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print(f"Loading data from: {data_dir}")

    # --- 2. Load Datasets ---
    # ImageFolder automatically labels based on subfolder names
    # e.g., 'labeled-frames/train/cat-present' will be class 0, 'labeled-frames/train/no-cat' will be class 1
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'validation']}

    # DataLoaders for batching and shuffling
    # num_workers: number of subprocesses to use for data loading (adjust based on your CPU cores)
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=(x=='train'), num_workers=os.cpu_count() // 2 or 1)
                   for x in ['train', 'validation']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}
    class_names = image_datasets['train'].classes # Get the class names from the dataset (e.g., ['cat-present', 'no-cat'])

    print(f"Detected classes: {class_names}")
    print(f"Training dataset size: {dataset_sizes['train']}")
    print(f"Validation dataset size: {dataset_sizes['validation']}")

    # --- 3. Determine Device ---
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 4. Load Pre-trained Model and Modify for our classes ---
    # We'll use a pre-trained ResNet18 model
    # 'weights=models.ResNet18_Weights.IMAGENET1K_V1' loads pre-trained weights from ImageNet
    model_ft = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # Get the number of input features for the final fully connected layer
    num_ftrs = model_ft.fc.in_features
    # Replace the final layer with a new one that has 'len(class_names)' output features
    # This adapts the model to our specific binary (or multi-class) problem
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))

    # --- 5. Define Loss Function and Optimizer ---
    # CrossEntropyLoss is suitable for multi-class classification (including binary)
    criterion = nn.CrossEntropyLoss()

    # Adam optimizer is a good general-purpose choice
    # model_ft.parameters() tells the optimizer which parameters to update
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)

    # Handle Class Imbalance
    # If 'cat-present' and 'no-cat' classes are very imbalanced,
    # uncomment and use class weights.
    # First, calculate class counts:
    # class_counts = [0] * len(class_names)
    # for _, label in image_datasets['train']:
    #     class_counts[label] += 1
    # total_samples = sum(class_counts)
    # class_weights_list = [total_samples / count if count > 0 else 0 for count in class_counts]
    # class_weights_tensor = torch.tensor(class_weights_list, dtype=torch.float).to(device)
    # criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    # print(f"Class weights used: {class_weights_list}")


    # --- 6. Train the Model ---
    print("\nStarting model training...")
    model_ft = train_model(model_ft, dataloaders, dataset_sizes, criterion, optimizer_ft, num_epochs=15) # Adjust num_epochs as needed

    # --- 7. Save the Trained Model ---
    torch.save(model_ft.state_dict(), model_save_path)
    print(f"\nModel saved to {model_save_path}")

