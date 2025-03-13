import os
import re
import argparse
import time
import cpuinfo
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torchvision.models import vit_b_16, ViT_B_16_Weights

class NagatoSakuraImageQualityClassificationTransformer(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(NagatoSakuraImageQualityClassificationTransformer, self).__init__()
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        self.transformer = vit_b_16(weights=weights)
        self.dropout = nn.Dropout(p=dropout_rate)  
        self.fc = nn.Linear(self.transformer.heads.head.in_features, 1)
        self.transformer.heads.head = nn.Identity()

    def forward(self, x):
        x = self.transformer(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x
    
class QualityDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.lower().endswith(('jpg', 'jpeg', 'png'))]
        self.transform = transform
        self.images = []
        self.quality_scores = []
        for image_path in self.image_paths:
            image = Image.open(image_path).convert("RGB")
            if self.transform:
                image = self.transform(image)
            self.images.append(image)
            match = re.search(r'_q(\d+)', os.path.basename(image_path))
            if not match:
                raise ValueError(f"檔案名 {image_path} 無法提取品質分數！")
            quality_score = int(match.group(1))
            quality_score = quality_score / 100.0
            quality_score = torch.tensor(quality_score, dtype=torch.float32)
            self.quality_scores.append(quality_score)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        return self.images[idx], self.quality_scores[idx]

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    best_loss = float('inf')
    training_start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                outputs = model(images).squeeze(1)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        avg_val_loss = val_loss / len(val_loader)
        average_loss = avg_train_loss
        elapsed_time = time.time() - start_time
        eta = (num_epochs - epoch - 1) * elapsed_time
        total_training_time = time.time() - training_start_time
        progress = epoch + 1
        percentage = progress / num_epochs * 100
        fill_length = int(50 * progress / num_epochs)
        space_length = 50 - fill_length
        print(f"Processing: {percentage:3.0f}%|{'█' * fill_length}{' ' * space_length}| {progress}/{num_epochs} [{total_training_time:.2f}<{eta:.2f}, {1 / elapsed_time:.2f}it/s, Loss: {average_loss:.4f}] ")

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            if not os.path.exists('./models'):
                os.makedirs('./models')
            torch.save(model.state_dict(), "./models/best_quality_predictor_transformer.pth")

    return model

# 主程式
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='長門櫻影像品質分類訓練器')
    parser.add_argument('--data_dir', type=str, default='./data/quality_dataset', help='訓練集路徑')
    parser.add_argument('--num_epochs', type=int, default=100, help='訓練步數')
    parser.add_argument('--batch_size', type=int, default=32, help='訓練批量')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='學習率')
    parser.add_argument('--weight_decay', type=float, default=0, help='權重衰減')
    parser.add_argument('--dropout_rate', type=float, default=0.1, help='Dropout 丟棄率')
    args = parser.parse_args()

    data_dir = args.data_dir
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    weight_decay = args.weight_decay
    dropout_rate = args.dropout_rate

    # 檢查可用的 NVIDIA 顯示卡並設置為運算裝置及檢測環境
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch版本 : {torch.__version__}")
    print(f"計算平臺 : {device}")
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            print(f"計算設備 : GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        cpu_info = cpuinfo.get_cpu_info()
        cpu_name = cpu_info['brand_raw']
        print(f"計算設備 : {cpu_name}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = QualityDataset(data_dir, transform=transform)
    train_indices, val_indices = train_test_split(list(range(len(dataset))), test_size=0.2, random_state=42)
    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    model = NagatoSakuraImageQualityClassificationTransformer(dropout_rate=dropout_rate)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    trained_model = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)
    if not os.path.exists('./models'):
        os.makedirs('./models')
    torch.save(trained_model.state_dict(), "./models/final_quality_predictor_transformer.pth")
    print("最終模型已保存！")