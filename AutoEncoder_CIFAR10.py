
from tqdm import tqdm
from matplotlib import pyplot as plt

import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class AutoEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self._encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),       # 32x32 -> 16x16
            nn.ReLU(), 
            nn.Conv2d(32, 64, 3, stride=2, padding=1),      # 16x16 -> 8x8
            nn.ReLU(), 
            nn.Conv2d(64, 128, 3, stride=2, padding=1),      # 8x8 -> 4x4
            nn.ReLU()
        )
    
        self._fc1 = nn.Linear(128*4*4, 128*4)
        self._fc2 = nn.Linear(128*4, 128*4*4)

        self._decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(), 
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._encoder.forward(x)
        x = x.view(x.size(0), -1)
        x = self._fc1.forward(x)
        x = self._fc2.forward(x)
        x = x.view(x.size(0), 128, 4, 4)
        x = self._decoder.forward(x)
        return x

def main():
    transform = transforms.ToTensor()
    train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10(root='./data',  train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoEncoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 500
    for epoch in tqdm(range(epochs)):
        model.train()
        total_train_loss = 0.0
        for images, _ in train_loader:
            images = images.to(device)
            outputs = model(images)

            loss: torch.Tensor
            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
        ave_train_loss = total_train_loss / len(train_loader)

        # 評価
        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                outputs = model(images)
                loss = criterion(outputs, images)
                total_test_loss += loss.item()
        ave_test_loss = total_test_loss / len(test_loader)

        tqdm.write(f'epoch: {epoch+1}, train loss: {ave_train_loss}, test loss: {ave_test_loss}')
    
    model.eval()
    test_imgs = next(iter(test_loader))[0][:8].to(device)

    with torch.no_grad():
        recon_imgs = model(test_imgs)

    fig, axs = plt.subplots(2, 8, figsize=(15, 4))
    for i in range(8):
        # permute (C, H, W) -> (H, W, C) for imshow
        axs[0, i].imshow(test_imgs[i].cpu().permute(1, 2, 0))
        axs[0, i].axis("off")
        axs[1, i].imshow(recon_imgs[i].cpu().permute(1, 2, 0))
        axs[1, i].axis("off")

    axs[0, 0].set_ylabel("Input", fontsize=14)
    axs[1, 0].set_ylabel("Output", fontsize=14)
    plt.suptitle("AutoEncoder Reconstruction (CIFAR)", fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()