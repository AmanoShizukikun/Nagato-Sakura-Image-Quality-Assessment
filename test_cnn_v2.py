import os
import argparse
import time
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, Optional

# 實現深度可分離卷積層，減少計算量並保持模型效能
class DepthwiseSeparableConv(nn.Module):
    """深度可分離卷積層，將標準卷積分解為逐深度卷積和逐點卷積"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int):
        super().__init__()
        # 逐深度卷積層，每個輸入通道獨立處理
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels, bias=False)
        # 1x1 逐點卷積層，整合通道間信息
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

# 定義輕量化CNN模型用於圖像品質評分任務
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
            nn.MaxPool2d(kernel_size=2, stride=2), # 縮小空間尺寸：256 -> 128

            # 第二區塊
            DepthwiseSeparableConv(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 縮小空間尺寸：128 -> 64

            # 第三區塊
            DepthwiseSeparableConv(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2), # 縮小空間尺寸：64 -> 32

            # 第四區塊
            DepthwiseSeparableConv(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 保留較大特徵圖進入全局平均池化層
        )
        # 全局平均池化替代展平和大型全連接層，減少參數
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, 1) # 輸出單一數值作為品質評分
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1) # 將特徵圖展平為向量
        x = self.classifier(x)
        return x

def load_model(model_path, dropout_rate=0, device='cuda'):
    """載入已訓練的模型"""
    device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
    
    # 初始化模型
    model = NagatoSakuraImageQualityClassificationCNN(dropout_rate=dropout_rate)
    
    try:
        # 載入模型權重
        model_state = torch.load(model_path, map_location=device)
        model.load_state_dict(model_state)
        model.to(device)
        model.eval()  # 設置為評估模式
        print(f"成功載入模型：{model_path}")
        return model, device
    except Exception as e:
        print(f"載入模型失敗：{e}")
        return None, device

def preprocess_image(image_path, transform=None):
    """處理輸入圖片，轉換為模型輸入格式"""
    try:
        with Image.open(image_path) as img:
            # 確保圖片是RGB模式
            img = img.convert('RGB')
            
            # 如果沒有提供轉換，使用默認轉換
            if transform is None:
                transform = transforms.Compose([
                    transforms.Resize((256, 256)),  # 調整大小
                    transforms.ToTensor(),          # 轉為張量
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 標準化
                ])
                
            return transform(img).unsqueeze(0)  # 添加批次維度
    except Exception as e:
        print(f"處理圖片 {image_path} 時發生錯誤：{e}")
        return None

def predict_quality(model, image_tensor, device):
    """使用模型預測圖片品質分數"""
    with torch.no_grad():  # 不計算梯度
        image_tensor = image_tensor.to(device)
        with torch.cuda.amp.autocast(enabled=device.type == 'cuda'):
            prediction = model(image_tensor)
        
        # 獲取分數
        score = prediction.item()
        return score

def visualize_result(image_path, score, output_dir=None):
    """可視化預測結果"""
    plt.figure(figsize=(10, 6))
    
    # 讀取並顯示圖片
    img = Image.open(image_path).convert('RGB')
    plt.imshow(img)
    
    # 設置標題，包括預測的品質分數
    plt.title(f"品質評分: {score:.2f}", fontsize=16, fontweight='bold')
    plt.axis('off')
    
    # 如果指定了輸出目錄，保存圖片
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.join(output_dir, f"{Path(image_path).stem}_score_{score:.2f}.jpg")
        plt.savefig(output_filename, bbox_inches='tight')
        print(f"已保存結果至 {output_filename}")
    
    plt.show()

def batch_test(model, image_dir, device, output_dir=None, visualize=False):
    """批量測試資料夾中的所有圖片"""
    # 圖片預處理轉換
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    results = {}
    image_files = [f for f in Path(image_dir).glob('*.jpg')]
    image_files.extend([f for f in Path(image_dir).glob('*.png')])
    
    if not image_files:
        print(f"在 {image_dir} 中找不到圖片文件 (.jpg, .png)")
        return {}
    
    print(f"\n開始批量測試 {len(image_files)} 個圖片...")
    
    for img_path in tqdm(image_files):
        img_tensor = preprocess_image(img_path, transform)
        if img_tensor is not None:
            score = predict_quality(model, img_tensor, device)
            results[str(img_path)] = score
            
            if visualize:
                visualize_result(img_path, score, output_dir)
    
    # 輸出摘要統計
    if results:
        scores = list(results.values())
        print(f"\n批量測試結果摘要:")
        print(f"圖片數量: {len(scores)}")
        print(f"平均品質分數: {np.mean(scores):.2f}")
        print(f"最高品質分數: {max(scores):.2f}")
        print(f"最低品質分數: {min(scores):.2f}")
    
    return results

def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='長門櫻圖像品質評分測試程序')
    parser.add_argument('--model_path', type=str, default='./models/best_quality_predictor_cnn_v2.pth', 
                        help='訓練好的模型路徑')
    parser.add_argument('--image_path', type=str, default=None, 
                        help='要評分的圖片路徑 (單張圖片模式)')
    parser.add_argument('--image_dir', type=str, default=None, 
                        help='要批量評分的圖片目錄 (批量模式)')
    parser.add_argument('--output_dir', type=str, default='./results', 
                        help='結果輸出目錄')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], 
                        help='計算設備 (cuda 或 cpu)')
    parser.add_argument('--dropout_rate', type=float, default=0, 
                        help='模型中的 Dropout 率 (需與訓練時相同)')
    parser.add_argument('--visualize', action='store_true', 
                        help='是否視覺化結果')
    
    args = parser.parse_args()
    
    # 檢查輸入參數
    if args.image_path is None and args.image_dir is None:
        print("錯誤：必須指定 --image_path (單張圖片) 或 --image_dir (圖片目錄)")
        return
    
    # 載入模型
    model, device = load_model(args.model_path, dropout_rate=args.dropout_rate, device=args.device)
    if model is None:
        return
    
    print(f"使用計算設備: {device}")
    
    start_time = time.time()
    
    # 單張圖片模式或批量模式
    if args.image_path:
        # 單張圖片測試
        image_tensor = preprocess_image(args.image_path)
        if image_tensor is not None:
            score = predict_quality(model, image_tensor, device)
            print(f"\n圖片: {args.image_path}")
            print(f"品質評分: {score:.2f}")
            
            if args.visualize:
                visualize_result(args.image_path, score, args.output_dir)
    else:
        # 批量測試
        batch_test(model, args.image_dir, device, args.output_dir, args.visualize)
    
    elapsed_time = time.time() - start_time
    print(f"\n完成！耗時: {elapsed_time:.2f} 秒")

if __name__ == '__main__':
    main()