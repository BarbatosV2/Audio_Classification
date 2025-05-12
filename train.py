import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import AudioClassifier
from utils import AudioDataset, collate_fn, get_device

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

    # Training loop
    model.train()
    for epoch in range(args.epoches):
        running_loss = 0.0
        correct_total = 0
        total_samples = 0

        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct_total += (predicted == labels).sum().item()
            total_samples += labels.size(0)

        # Calculate epoch accuracy
        accuracy = 100 * correct_total / total_samples
        print(f"Epoch {epoch + 1}/{args.epoches} | Loss: {running_loss:.4f} | Accuracy: {accuracy:.2f}%")

    # Save the model after training
    model_save_path = 'sound_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'label_to_idx': dataset.label_to_idx, # Save the label mapping
        'num_classes': dataset.num_classes   # Save the number of classes
    }, model_save_path)
    print(f"Model and label information saved as {model_save_path}")
    print(f"Labels used for training: {sorted(dataset.label_to_idx.keys())}")

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
