import argparse
import torch
import os # Added for path manipulation and directory creation
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import AudioClassifier
from tqdm import tqdm # Added for progress bar
from utils import AudioDataset, collate_fn, get_device
import matplotlib.pyplot as plt # Added for plotting

def train(args):
    # Get device (GPU/CPU)
    device = get_device(args.device)
    
    # Load dataset and DataLoader
    dataset = AudioDataset(args.data_path, args.csv_path)
    dataloader = DataLoader(dataset, batch_size=args.batch, shuffle=True, num_workers=args.workers, collate_fn=collate_fn)
    
    # Initialize model, loss, and optimizer
    model = AudioClassifier(num_classes=dataset.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Lists to store metrics for plotting
    epoch_losses = []
    epoch_accuracies = []

    # Training loop
    model.train()
    for epoch in range(args.epoches):
        current_epoch_loss = 0.0
        correct_total = 0
        total_samples = 0

        # Wrap dataloader with tqdm for a progress bar
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epoches}", leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            current_epoch_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct_total += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            # Update progress bar description with current loss (optional)
            # progress_bar.set_postfix(loss=loss.item())

        # Calculate epoch accuracy
        epoch_avg_loss = current_epoch_loss / len(dataloader)
        epoch_accuracy = 100 * correct_total / total_samples
        
        epoch_losses.append(epoch_avg_loss)
        epoch_accuracies.append(epoch_accuracy)
        print(f"Epoch {epoch + 1}/{args.epoches} | Loss: {epoch_avg_loss:.4f} | Accuracy: {epoch_accuracy:.2f}%")

    # --- Modified model saving logic ---
    model_dir = 'model'
    os.makedirs(model_dir, exist_ok=True) # Create model directory if it doesn't exist

    base_model_name = 'sound_model'
    model_extension = '.pth'
    model_save_path = os.path.join(model_dir, f"{base_model_name}{model_extension}")
    
    counter = 1
    while os.path.exists(model_save_path):
        model_save_path = os.path.join(model_dir, f"{base_model_name}_{counter}{model_extension}")
        counter += 1
    # --- End of modified model saving logic ---

    # Save the model after training
    torch.save({
        'model_state_dict': model.state_dict(),
        'label_to_idx': dataset.label_to_idx, # Save the label mapping
        'num_classes': dataset.num_classes   # Save the number of classes
    }, model_save_path)
    print(f"Model and label information saved as {model_save_path}")
    print(f"Labels used for training: {sorted(dataset.label_to_idx.keys())}")

    # --- Plotting and saving statistics ---
    stats_dir = 'train_data'
    os.makedirs(stats_dir, exist_ok=True) # Create train_data directory if it doesn't exist

    base_stats_name = 'stats'
    stats_extension = '.png'
    stats_save_path = os.path.join(stats_dir, f"{base_stats_name}{stats_extension}")
    
    counter = 1
    while os.path.exists(stats_save_path):
        stats_save_path = os.path.join(stats_dir, f"{base_stats_name}_{counter}{stats_extension}")
        counter += 1

    epochs_range = range(1, args.epoches + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, epoch_losses, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss vs. Epochs')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, epoch_accuracies, label='Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Training Accuracy vs. Epochs')
    plt.legend()

    plt.tight_layout()
    plt.savefig(stats_save_path)
    print(f"Training statistics plot saved as {stats_save_path}")
    # plt.show() # Uncomment if you want to display the plot as well

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=5)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--epoches', type=int, default=50)
    parser.add_argument('data_path', type=str)
    parser.add_argument('csv_path', type=str)
    args = parser.parse_args()

    # Call the train function
    train(args)
