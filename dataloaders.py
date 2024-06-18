from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch

class ComplexDataset(Dataset):
    def __init__(self, real_dataset):
        self.real_dataset = real_dataset
        # Ensure the number of images is even
        if len(self.real_dataset) % 2 != 0:
            print("Size of the dataset is not even!")

    def __len__(self):
        return len(self.real_dataset) // 2  # Number of pairs

    def __getitem__(self, idx):
        image1 = self.real_dataset[2 * idx][0]
        image2 = self.real_dataset[2 * idx + 1][0]
        image = torch.cat((image1, image2), dim=0)  # Concatenate along channel dimension
        return image, []


def get_complex_mnist_dataloaders(batch_size=32):
    """MNIST dataloader with (64, 64) sized images."""
    all_transforms = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor()
    ])
    # Get train and test data
    train_data = datasets.MNIST('../data', train=True, download=True, transform=all_transforms)
    test_data = datasets.MNIST('../data', train=False, transform=all_transforms)
    train_data = ComplexDataset(train_data)
    test_data = ComplexDataset(test_data)
    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
