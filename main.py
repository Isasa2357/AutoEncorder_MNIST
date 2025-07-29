
from tqdm import tqdm
from matplotlib import pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from AutoEncorder.AutoEncorder import AutoEncoder

def main():
    # --- データローダ設定 ---
    transform = transforms.ToTensor()
    train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_data  = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=128, shuffle=True)
    test_loader  = DataLoader(test_data, batch_size=128, shuffle=False)

    # --- 学習準備 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoEncoder()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # --- 学習ループ ---
    n_epochs = 100
    for epoch in tqdm(range(n_epochs), position=1):
        model.train()
        total_train_loss = 0
        for images, _ in tqdm(train_loader, position=0):
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)

        # テストローダで評価
        model.eval()
        total_test_loss = 0
        with torch.no_grad():
            for images, _ in test_loader:
                images = images.to(device)
                outputs = model(images)
                loss = criterion(outputs, images)
                total_test_loss += loss.item()
        avg_test_loss = total_test_loss / len(test_loader)

        tqdm.write(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {avg_train_loss:.4f} | Test Loss: {avg_test_loss:.4f}")

    # --- 結果の可視化（最終エポック） ---
    model.eval()
    test_imgs = next(iter(test_loader))[0][:8].to(device)
    with torch.no_grad():
        recon_imgs = model(test_imgs)

    fig, axs = plt.subplots(2, 8, figsize=(15, 4))
    for i in range(8):
        axs[0, i].imshow(test_imgs[i][0].cpu(), cmap="gray")
        axs[0, i].axis("off")
        axs[1, i].imshow(recon_imgs[i][0].cpu(), cmap="gray")
        axs[1, i].axis("off")
    axs[0, 0].set_ylabel("Input", fontsize=14)
    axs[1, 0].set_ylabel("Output", fontsize=14)
    plt.suptitle("AutoEncoder Reconstruction", fontsize=16)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()