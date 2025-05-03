from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from sequential_dataset import SequentialDataset  # Import the updated dataset class
from deep_emotion import Deep_Emotion  # Replace or modify based on your model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def Train(epochs, train_loader, val_loader, criterion, optimizer, device):
    '''
    Training Loop
    '''
    print("===================================Start Training===================================")
    for e in range(epochs):
        train_loss = 0
        validation_loss = 0
        train_correct = 0
        val_correct = 0

        # Train the model
        net.train()
        for sequences, labels in train_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data)

        # Validate the model
        net.eval()
        for sequences, labels in val_loader:
            sequences, labels = sequences.to(device), labels.to(device)
            val_outputs = net(sequences)
            val_loss = criterion(val_outputs, labels)
            validation_loss += val_loss.item()
            _, val_preds = torch.max(val_outputs, 1)
            val_correct += torch.sum(val_preds == labels.data)

        train_loss = train_loss / len(train_dataset)
        train_acc = train_correct.double() / len(train_dataset)
        validation_loss = validation_loss / len(validation_dataset)
        val_acc = val_correct.double() / len(validation_dataset)
        print('Epoch: {} \tTraining Loss: {:.8f} \tValidation Loss {:.8f} \tTraining Accuracy {:.3f}% \tValidation Accuracy {:.3f}%'
              .format(e + 1, train_loss, validation_loss, train_acc * 100, val_acc * 100))

    torch.save(net.state_dict(), 'deep_emotion-{}-{}-{}.pt'.format(epochs, batchsize, lr))
    print("===================================Training Finished===================================")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Configuration of setup and training process")
    parser.add_argument('-train_dir', '--train_dataset_path', type=str, required=True,
                        help='Path to the directory containing the training sequences')
    parser.add_argument('-val_dir', '--val_dataset_path', type=str, required=True,
                        help='Path to the directory containing the validation sequences')
    parser.add_argument('-e', '--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.005, help='Learning rate')
    parser.add_argument('-bs', '--batch_size', type=int, default=128, help='Batch size')

    args = parser.parse_args()

    # Hyperparameters
    epochs = args.epochs
    lr = args.learning_rate
    batchsize = args.batch_size

    # Load datasets
    train_dataset = SequentialDataset(dataset_path=args.train_dataset_path)
    val_dataset = SequentialDataset(dataset_path=args.val_dataset_path)

    train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batchsize, shuffle=False, num_workers=0)

    # Define the model, loss function, and optimizer
    net = Deep_Emotion()  # Replace with your model
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=lr)

    # Train the model
    Train(epochs, train_loader, val_loader, criterion, optimizer, device)