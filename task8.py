import time

import cv2

import matplotlib.pyplot as plt

import torch

import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import DataLoader

from tqdm import tqdm

from sklearn.metrics import confusion_matrix

import numpy as np

from utils.dataset import TeamMateDataset

from torchvision.models import mobilenet_v3_small

 

# Function to plot and save the confusion matrix

def plot_confusion_matrix(cm, epoch, save_path):

    plt.figure(figsize=(6, 6))

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)

    plt.title(f'Confusion Matrix - Epoch {epoch}')

    plt.colorbar()

    tick_marks = np.arange(cm.shape[0])

    plt.xticks(tick_marks, tick_marks)

    plt.yticks(tick_marks, tick_marks)

 

    # Labels on the plot

    plt.xlabel('Predicted Label')

    plt.ylabel('True Label')

    plt.tight_layout()

 

    # Save the plot

    plt.savefig(save_path)

    plt.close()

 

if __name__ == '__main__':

   

    # Setting up device

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

 

    # Create the datasets and dataloaders

    trainset = TeamMateDataset(n_images=50, train=True)

    testset = TeamMateDataset(n_images=10, train=False)

    trainloader = DataLoader(trainset, batch_size=4, shuffle=True)

    testloader = DataLoader(testset, batch_size=1, shuffle=False)

 

    # Create the model and optimizer

    model = mobilenet_v3_small(weights=None, num_classes=2).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    criterion = torch.nn.CrossEntropyLoss()

 

    # Tracking parameters

    best_test_accuracy = 0

    patience = 5  # Early stopping patience

    early_stop_counter = 0

    max_epochs = 100  # Set a maximum epoch limit

    min_loss = 1e9

 

    # Loss and accuracy lists

    train_losses = []

    test_losses = []

    test_accuracies = []

 

    # Epoch Loop

    for epoch in range(1, max_epochs + 1):

 

        # Start timer

        t = time.time_ns()

 

        # Train the model

        model.train()

        train_loss = 0

 

        for i, (images, labels) in tqdm(enumerate(trainloader), total=len(trainloader), leave=False):

            images = images.reshape(-1, 3, 64, 64).to(device)

            labels = labels.to(device)

 

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            train_loss += loss.item()

 

        # Test the model

        model.eval()

        test_loss = 0

        correct = 0

        total = 0

        all_labels = []

        all_preds = []

 

        with torch.no_grad():

            for images, labels in tqdm(testloader, total=len(testloader), leave=False):

                images = images.reshape(-1, 3, 64, 64).to(device)

                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(outputs, labels)

                test_loss += loss.item()

 

                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)

                correct += (predicted == labels).sum().item()

                all_labels.extend(labels.cpu().numpy())

                all_preds.extend(predicted.cpu().numpy())

 

        test_accuracy = correct / total

        test_accuracies.append(test_accuracy)

 

        # Calculate and save confusion matrix

        cm = confusion_matrix(all_labels, all_preds)

        plot_confusion_matrix(cm, epoch, f'/home/pi/ee347/lab-8-pytorch-and-deep-learning-2-group-10/live_confusion_matrix.png')

 

        # Update loss lists

        train_losses.append(train_loss / len(trainloader))

        test_losses.append(test_loss / len(testloader))

 

        # Print the epoch statistics

        print(f'Epoch: {epoch}, Train Loss: {train_loss / len(trainloader):.4f}, '

              f'Test Loss: {test_loss / len(testloader):.4f}, Test Accuracy: {test_accuracy:.4f}, '

              f'Time: {(time.time_ns() - t) / 1e9:.2f}s')

 

        # Save the best model based on test accuracy

        if test_accuracy > best_test_accuracy:

            best_test_accuracy = test_accuracy

            torch.save(model.state_dict(), '/home/pi/ee347/lab-8-pytorch-and-deep-learning-2-group-10/best_model.pth')

            print(f'New best model saved with Test Accuracy: {best_test_accuracy:.4f}')

            early_stop_counter = 0  # Reset the early stopping counter

        else:

            early_stop_counter += 1

 

        # Early stopping check

        if early_stop_counter >= patience:

            print(f"Stopping early after {epoch} epochs due to no improvement in test accuracy.")

            break

 

        # Save the model and plot loss

        torch.save(model.state_dict(), '/home/pi/ee347/lab-8-pytorch-and-deep-learning-2-group-10/current_model.pth')

 

        plt.plot(train_losses, label='Train Loss')

        plt.plot(test_losses, label='Test Loss')

        plt.xlabel('Epoch')

        plt.ylabel('Loss')

        plt.legend()

        plt.savefig('/home/pi/ee347/lab-8-pytorch-and-deep-learning-2-group-10/task8_loss_plot.png')

        plt.close()

 

    print(f"Training complete. Best model with Test Accuracy: {best_test_accuracy:.4f} saved.")