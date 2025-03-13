import os
import torch
from PIL import Image
import torch.nn as nn
import torchvision.transforms as transforms
import argparse
from train_transformer import NagatoSakuraImageQualityClassificationTransformer 

def predict_quality(image_path, model_path, device):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"找不到圖片: {image_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到模型: {model_path}")
    
    model = NagatoSakuraImageQualityClassificationTransformer()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)
    
    with torch.no_grad():
        image = image.to(device)
        output = model(image).squeeze(1)
        quality_score = output.item() * 100 
    
    return round(quality_score, 2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='長門櫻影像品質分類測試器')
    parser.add_argument('image_path', type=str, help='測試圖片路徑')
    parser.add_argument('--model_path', type=str, default='./models/best_quality_predictor_transformer.pth', help='模型路徑')
    args = parser.parse_args()

    image_path = args.image_path
    model_path = args.model_path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        quality_score = predict_quality(image_path, model_path, device)
        print(f"圖片: {image_path} 的預測品質為: {quality_score}")
    except Exception as e:
        print(f"錯誤: {e}")