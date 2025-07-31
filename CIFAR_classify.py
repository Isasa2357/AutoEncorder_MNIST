# torch
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from tqdm import tqdm
from matplotlib import pyplot as plt


class CIFAR10Classifier(nn.Module):
    '''
    CIFARの分類器
    '''

    def __init__(self):
        super().__init__()

        self._featureExtraction = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1), 
            nn.BatchNorm2d(16),
            nn.ReLU(), 
            nn.Conv2d(16, 32, 3, stride=2, padding=1), 
            nn.BatchNorm2d(32),
            nn.ReLU(), 
            nn.Conv2d(32, 64, 3, stride=2, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        self._classifierHeader = nn.Sequential(
            nn.Dropout(0.5), 
            nn.Linear(4*4*64, 10)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._featureExtraction(x)
        x = x.view(x.size(0), -1)
        x = self._classifierHeader(x)

        return x

def main():
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])
    train_data = datasets.CIFAR10(root='./data', 
                                  train=True, 
                                  download=True, 
                                  transform=transform)
    test_data = datasets.CIFAR10(root='./data', 
                                 train=False, 
                                 download=True, 
                                 transform=transform)
    
    batch_size = 256
    train_loader = DataLoader(train_data, 
                              batch_size=batch_size, 
                              shuffle=True)
    test_loader = DataLoader(test_data, 
                             batch_size=batch_size, 
                             shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model = CIFAR10Classifier().to(device)

    # hyper parameter
    lr = 1e-3
    lossF = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    epochs = 500

    for epoch in tqdm(range(epochs), ncols=100):
        model.train()
        total_train_loss = 0.0

        for images, labels in train_loader:

            images = images.to(device)
            labels = labels.to(device)
            outputs = model.forward(images)

            loss: torch.Tensor = lossF(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
        
        # 評価
        success_count = 0
        total_test_loss = 0.0
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model.forward(images)

            pred = outputs.argmax(dim=1)

            loss: torch.Tensor = lossF(outputs, labels)
            total_test_loss += loss.item()
            success_count += sum(pred == labels)
        
        tqdm.write(f'epoch: {epoch+1}, train loss: {total_train_loss / len(train_loader)}, test loss: {total_test_loss / len(test_loader)}, success rate: {success_count / len(test_data)}')

    # 結果の例表示
    images = next(iter(test_loader))[0][:8].to(device)
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        preds = outputs.argmax(dim=1)

    plt.figure(figsize=(12, 3))
    for i in range(8):
        plt.subplot(1, 8, i+1)
        plt.imshow(images[i].cpu().permute(1, 2, 0))
        plt.title(f'{preds[i].item()}')
        plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()