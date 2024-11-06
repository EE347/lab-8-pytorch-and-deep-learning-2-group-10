import time

import cv2

import matplotlib.pyplot as plt

import torch

import torch.nn.functional as F

import torch.optim as optim

from torch.utils.data import DataLoader

from tqdm import tqdm

from utils.dataset import TeamMateDataset

from torchvision import transforms

from torchvision.models import mobilenet_v3_small

 

# Define transformations

random_rotation = transforms.RandomRotation(degrees=10)

 

if __name__ == '__main__':

    if torch.cuda.is_available():

        device = torch.device('cuda')

    else:

        device = torch.device('cpu')

 

    # Create the datasets and dataloaders

    trainset = TeamMateDataset(n_images=50, train=True)

    testset = TeamMateDataset(n_images=10, train=False)

    trainloader = DataLoader(trainset, batch_size=4, shuffle=True)

    testloader = DataLoader(testset, batch_size=1, shuffle=False)

 

    # Create the model and optimizer

    model = mobilenet_v3_small(weights=None, num_classes=2).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)

 

    # Loss functions

    loss_functions = {

        'CrossEntropyLoss': torch.nn.CrossEntropyLoss(),

        'NLLLoss': torch.nn.NLLLoss()

    }

 

    results = {'Epoch': [], 'Loss Type': [], 'Train Loss': [], 'Test Loss': [], 'Test Accuracy': []}

 

    for loss_name, criterion in loss_functions.items():

        best_train_loss = 1e9

        train_losses = []

        test_losses = []

 

        # Epoch Loop

        for epoch in range(1, 5):

            t = time.time_ns()

            model.train()

            train_loss = 0

 

            for i, (images, labels) in tqdm(enumerate(trainloader), total=len(trainloader), leave=False):

                images = images.reshape(-1, 3, 64, 64).to(device)

                labels = labels.to(device)

 

                # Apply random transformations

                images = torch.stack([random_rotation(image) for image in images])

                flip_prob = 0.5

                flip_mask = torch.rand(images.size(0)) < flip_prob

                if flip_mask.any():

                    images[flip_mask] = torch.flip(images[flip_mask], dims=[3])

 

                optimizer.zero_grad()

 

                # Forward pass (apply log_softmax if using NLLLoss)

                outputs = model(images)

                if loss_name == 'NLLLoss':

                    outputs = F.log_softmax(outputs, dim=1)

 

                # Compute the loss

                loss = criterion(outputs, labels)

                loss.backward()

                optimizer.step()

 

                train_loss += loss.item()

 

            # Test the model

            model.eval()

            test_loss = 0

            correct = 0

            total = 0

 

            for images, labels in tqdm(testloader, total=len(testloader), leave=False):

                images = images.reshape(-1, 3, 64, 64).to(device)

                labels = labels.to(device)

 

                outputs = model(images)

                if loss_name == 'NLLLoss':

                    outputs = F.log_softmax(outputs, dim=1)

                loss = criterion(outputs, labels)

                test_loss += loss.item()

 

                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)

                correct += (predicted == labels).sum().item()

 

            train_loss_avg = train_loss / len(trainloader)

            test_loss_avg = test_loss / len(testloader)

            accuracy = correct / total

 

            # Log the results for tabulation

            results['Epoch'].append(epoch)

            results['Loss Type'].append(loss_name)

            results['Train Loss'].append(train_loss_avg)

            results['Test Loss'].append(test_loss_avg)

            results['Test Accuracy'].append(accuracy)

 

            print(f'Epoch: {epoch}, Loss Type: {loss_name}, Train Loss: {train_loss_avg:.4f}, Test Loss: {test_loss_avg:.4f}, Test Accuracy: {accuracy:.4f}, Time: {(time.time_ns() - t) / 1e9:.2f}s')

 

            train_losses.append(train_loss_avg)

            test_losses.append(test_loss_avg)

 

            if train_loss < best_train_loss:

                best_train_loss = train_loss

                torch.save(model.state_dict(), f'/home/pi/ee347/lab-8-pytorch-and-deep-learning-2-group-10/best_model_{loss_name}.pth')

 

        plt.plot(train_losses, label=f'{loss_name} Train Loss')

        plt.plot(test_losses, label=f'{loss_name} Test Loss')

 

    plt.xlabel('Epoch')

    plt.ylabel('Loss')

    plt.legend()

    plt.savefig('/home/pi/ee347/lab-8-pytorch-and-deep-learning-2-group-10/task5_loss_plot.png')

 

# Save the results table for tabulation

import pandas as pd

results_df = pd.DataFrame(results)

results_df.to_csv('/home/pi/ee347/lab-8-pytorch-and-deep-learning-2-group-10/task5_results.csv', index=False)

print("Results saved to task5_results.csv")