from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import torch
import math

class ComplexDataset(Dataset):
    def __init__(self, real_dataset):
        self.real_dataset = real_dataset
        # Ensure the number of images is even
        if len(self.real_dataset) % 2 != 0:
            print("Size of the dataset is not even!")

    def __len__(self):
        return len(self.real_dataset) // 2  # Number of pairs

    def __getitem__(self, idx):
        # obtain the magnitude and phase parts
        magnitude = self.real_dataset[2 * idx][0]
        phase = self.real_dataset[2 * idx + 1][0]
        # construct the complex tensor
        complex = torch.polar(magnitude, phase)
        # real and imaginary parts
        real = torch.real(complex)
        imag = torch.imag(complex)
        image = torch.cat((real, imag), dim=0)  # Concatenate along channel dimension
        return image, []


def get_complex_mnist_dataloaders(batch_size=64, image_size=256):
    """MNIST dataloader with (256, 256) sized images."""
    all_transforms = transforms.Compose([
        transforms.Resize(image_size),
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


def get_complex_celeba_dataloaders(batch_size=64, image_size=256):
    """CelebA dataloader with (256, 256) sized images."""
    all_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.Grayscale(),
        transforms.ToTensor()
    ])
    # Get train and test data
    train_data = datasets.CelebA('../data', split='train', transform=all_transforms, download=True)
    test_data = datasets.CelebA('../data', split='test', transform=all_transforms, download=True)
    train_data = ComplexDataset(train_data)
    test_data = ComplexDataset(test_data)
    # Create dataloaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader
