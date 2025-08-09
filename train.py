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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.amp import GradScaler, autocast
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from typing import Tuple, Optional, List, Dict, Any

class DepthwiseSeparableConv(nn.Module):
    """深度可分離卷積層，將標準卷積分解為逐深度卷積和逐點卷積"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class NagatoSakuraImageQualityClassificationCNN(nn.Module):
    """
    輕量化CNN模型，專為圖像品質評分設計。
    使用深度可分離卷積、批量歸一化和全局平均池化減少參數量。
    """
    def __init__(self, dropout_rate: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            # 第一區塊
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第二區塊
            DepthwiseSeparableConv(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第三區塊
            DepthwiseSeparableConv(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 第四區塊
            DepthwiseSeparableConv(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class QualityDataset(Dataset):
    """
    圖像品質評分數據集，支持延遲讀取圖像以節省記憶體。
    從檔名中提取品質分數，格式為 *_q<分數>.*
    """
    def __init__(self, image_dir: str, transform: Optional[transforms.Compose] = None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files: List[str] = []
        self.quality_scores: List[float] = []
        print(f"正在從 {image_dir} 載入數據...")
        try:
            with os.scandir(image_dir) as entries:
                for entry in tqdm(entries, desc="掃描目錄"):
                    if entry.is_file() and entry.name.lower().endswith(('jpg', 'jpeg', 'png')):
                        image_path = entry.path
                        match = re.search(r'_q(\d+)', entry.name)
                        if match:
                            quality_score = float(match.group(1))
                            self.image_files.append(image_path)
                            self.quality_scores.append(quality_score)
                        else:
                            print(f"警告：檔案名 {image_path} 無法提取品質分數，已跳過。")
        except FileNotFoundError:
             raise RuntimeError(f"錯誤：找不到目錄 {image_dir}。")
        except Exception as e:
             raise RuntimeError(f"讀取目錄 {image_dir} 時發生錯誤: {e}")
        if not self.image_files:
             raise RuntimeError(f"在目錄 {image_dir} 中找不到符合命名規則 (*_q<數字>.jpg/png) 的圖像文件。")
        print(f"成功載入 {len(self.image_files)} 個圖像路徑和分數。")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path = self.image_files[idx]
        try:
            with Image.open(image_path) as img:
                image = img.convert("RGB")
            if self.transform:
                image = self.transform(image)
        except FileNotFoundError:
             print(f"錯誤：找不到圖像文件 {image_path}")
             image = torch.zeros((3, 256, 256))
        except Exception as e:
            print(f"錯誤：無法載入或轉換圖像 {image_path}: {e}")
            image = torch.zeros((3, 256, 256))
        quality_score = torch.tensor(self.quality_scores[idx], dtype=torch.float32)
        return image, quality_score

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    num_epochs: int,
    device: torch.device,
    model_save_path: str = "./models/best_quality_predictor.pth"
) -> nn.Module:
    """訓練模型並保存最佳驗證損失狀態，返回訓練後的模型"""
    model.to(device)
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 個 GPUs 進行 DataParallel 訓練。")
        model = nn.DataParallel(model)
    use_amp = device.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)
    best_val_loss = float('inf')
    training_start_time = time.time()
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    print("\n--- 開始訓練 ---")
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for images, labels in train_pbar:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True).unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type=device.type, dtype=torch.float16 if use_amp else torch.float32, enabled=use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels.to(outputs.dtype))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            train_pbar.set_postfix(loss=f'{loss.item():.4f}')
        avg_train_loss = running_loss / len(train_loader)
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
        with torch.no_grad():
            for images, labels in val_pbar:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True).unsqueeze(1)
                with autocast(device_type=device.type, dtype=torch.float16 if use_amp else torch.float32, enabled=use_amp):
                    outputs = model(images)
                    loss = criterion(outputs, labels.to(outputs.dtype))
                val_loss += loss.item()
                mae = torch.abs(outputs - labels.to(outputs.dtype)).mean()
                val_mae += mae.item()
                val_pbar.set_postfix(loss=f'{loss.item():.4f}', mae=f'{mae.item():.4f}')
        avg_val_loss = val_loss / len(val_loader)
        avg_val_mae = val_mae / len(val_loader)
        epoch_elapsed_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"時間: {epoch_elapsed_time:.2f}s | "
              f"訓練損失: {avg_train_loss:.4f} | "
              f"驗證損失: {avg_val_loss:.4f} | "
              f"驗證 MAE: {avg_val_mae:.4f} | "
              f"LR: {current_lr:.6f}")
        if scheduler:
            scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model_to_save = model.module if isinstance(model, nn.DataParallel) else model
            try:
                torch.save(model_to_save.state_dict(), model_save_path)
                print(f"*** 發現新的最佳模型，已保存至 {model_save_path} (Val Loss: {best_val_loss:.4f}) ***")
            except Exception as e:
                print(f"錯誤：保存模型失敗: {e}")
    total_training_time = time.time() - training_start_time
    print(f"\n--- 訓練完成 ---")
    print(f"總訓練時間: {total_training_time // 60:.0f} 分 {total_training_time % 60:.0f} 秒")
    print(f"最佳驗證損失 (MSE): {best_val_loss:.4f}")
    try:
        best_model_state = torch.load(model_save_path, map_location=device)
        dropout_rate = model.module.classifier[0].p if isinstance(model, nn.DataParallel) else model.classifier[0].p
        final_model = NagatoSakuraImageQualityClassificationCNN(dropout_rate=dropout_rate)
        final_model.load_state_dict(best_model_state)
        final_model.to(device)
        final_model.eval()
        return final_model
    except FileNotFoundError:
        print(f"錯誤：找不到已保存的最佳模型文件 {model_save_path}。返回訓練結束時的模型。")
        model_to_return = model.module if isinstance(model, nn.DataParallel) else model
        model_to_return.eval()
        return model_to_return
    except Exception as e:
        print(f"錯誤：載入最佳模型失敗: {e}。返回訓練結束時的模型。")
        model_to_return = model.module if isinstance(model, nn.DataParallel) else model
        model_to_return.eval()
        return model_to_return

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='改良版長門櫻影像品質分類訓練器')
    parser.add_argument('--data_dir', type=str, default='./data/quality_dataset_01_J', help='包含 *_q<分數>.jpg/png 圖像的訓練集路徑')
    parser.add_argument('--num_epochs', type=int, default=10000, help='訓練週期數')
    parser.add_argument('--batch_size', type=int, default=32, help='訓練批量大小 (可根據 GPU 記憶體調整)')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='初始學習率')
    parser.add_argument('--num_workers', type=int, default=16, help='數據加載器的工作線程數')
    parser.add_argument('--dropout_rate', type=float, default=0, help='模型分類器中的 Dropout 率')
    parser.add_argument('--model_save_path', type=str, default='./models/best_quality_predictor_cnn_v2.pth', help='最佳模型保存路徑')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"計算平台: {device}")
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        for i in range(num_gpus):
            print(f"  計算設備 GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        try:
            cpu_info = cpuinfo.get_cpu_info()
            cpu_name = cpu_info['brand_raw']
            print(f"  計算設備 CPU: {cpu_name}")
        except Exception as e:
            print(f"  無法獲取 CPU 詳細信息: {e}")
    print(f"數據加載器 Workers: {args.num_workers}")
    print(f"批次大小: {args.batch_size}")
    print(f"初始學習率: {args.learning_rate}")
    print(f"訓練週期: {args.num_epochs}")
    print(f"Dropout 率: {args.dropout_rate}")
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    try:
        full_dataset = QualityDataset(args.data_dir, transform=transform)
    except RuntimeError as e:
        print(f"錯誤：初始化數據集失敗: {e}")
        exit(1)
    if len(full_dataset) == 0:
        print(f"錯誤：在 {args.data_dir} 中未找到有效的圖像數據。請檢查路徑和文件名格式。")
        exit(1)
    train_indices, val_indices = train_test_split(
        list(range(len(full_dataset))),
        test_size=0.2, # 80%訓練，20%驗證
        random_state=42, # 固定隨機種子確保可重複性
        stratify=None # 簡單隨機分割
    )
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)
    print(f"訓練集大小: {len(train_dataset)}")
    print(f"驗證集大小: {len(val_dataset)}")
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True, # 打亂訓練數據
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False, # GPU訓練時啟用固定記憶體
        persistent_workers=args.num_workers > 0, # 保持工作進程存活提高效率
        prefetch_factor=2 if args.num_workers > 0 else None, # 預取因子加速數據載入
        drop_last=True # 丟棄最後不完整批次提高訓練穩定性
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size * 2, # 驗證時使用更大批次
        shuffle=False, # 無需打亂驗證數據
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False,
        persistent_workers=args.num_workers > 0,
        prefetch_factor=2 if args.num_workers > 0 else None
    )
    model = NagatoSakuraImageQualityClassificationCNN(dropout_rate=args.dropout_rate)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4) # AdamW優化器添加權重衰減
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=8, verbose=False) # 當驗證損失停止改善時降低學習率
    trained_model = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        args.num_epochs,
        device,
        model_save_path=args.model_save_path
    )
    print(f"\n訓練完成！最佳模型已保存在 {args.model_save_path}")