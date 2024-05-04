
# onthou om n program te maak waarmee jy kan teken en dan se hy vir jou watse nommer dit is

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


def init_MNIST(download_data:bool, batch_size = 128) -> tuple[DataLoader, DataLoader]:
    """Initializes several fields needed for classification on the MNIST dataset.

    Args:
        download_data (bool): Specifies whether to download the data. Pass in False if data is already downloaded.
        batch_size (int, optional): Batch size of the dataloaders. Defaults to 256.

    Returns:
        tuple[DataLoader, DataLoader]: Training and testing- torch dataloaders
    """
    
    training_data = datasets.MNIST(
    root="../data",
    train=True,
    download=download_data
    )


    mean = torch.true_divide((training_data.data), 255).float().mean().item()
    std = torch.true_divide((training_data.data), 255).float().std().item()
    print(mean)
    print(std)
    transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Normalize((mean,), (std,))
                               ])
    training_data = datasets.MNIST(
    root="../data",
    train=True,
    download=download_data,
    transform=transform,
    )
    
    test_data = datasets.MNIST(
        root="../data",
        train=False,
        download=download_data,
        transform=transform,
    )
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle= True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
    return (train_dataloader, test_dataloader)
    
class CNN(nn.Module):
    
    def __init__(self):
        super(CNN, self).__init__()
        self.device = "cuda" if torch.cuda.is_available else "cpu"
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3, 3),stride = 1, padding = 1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2)
            
        )
        self.classify = nn.Sequential(
            nn.Linear(in_features=7*7*32, out_features=256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Linear(in_features=256, out_features=10)         
        )
        

    def forward(self, x):
        x = self.features(x)
        #print(x.shape)  # Print the shape for debugging
        x = x.view(-1, 32*7*7)
        #print(x.shape)  # Print the shape for debugging
        x = self.classify(x)
        return x
    
    def train_from_dataset(self, epochs : int, train_dataloader : DataLoader, test_dataloader : DataLoader):
        
        device = self.device
        loss_function = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr = 0.01)
        for i in range(epochs):
            print(f"Epoch {i+1}\n")
            self.train() # type: ignore
            for batch, (X, y) in enumerate(train_dataloader): #here, batch is the number of the batch, and X and y are lists with the input out output values of that batch
                X, y = X.to(device), y.to(device)

                # Compute prediction error
                pred = self(X)

                loss = loss_function(pred, y)

                # Backpropagation
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            size = len(test_dataloader.dataset) # type: ignore
            num_batches = len(test_dataloader)
            self.eval()
            test_loss, correct = 0, 0
            with torch.no_grad():
                for X, y in test_dataloader:
                    X, y = X.to(device), y.to(device)
                    pred = self(X) #sakjdfl
                    test_loss += loss_function(pred, y).item()
                    correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            test_loss /= num_batches
            correct /= size
            print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    
    
    
                            
    
        
        

