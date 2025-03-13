# Nagato-Sakura-Image-Quality-Classification

[![GitHub Repo stars](https://img.shields.io/github/stars/AmanoShizukikun/Nagato-Sakura-Image-Quality-Classification?style=social)](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Quality-Classification/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/AmanoShizukikun/Nagato-Sakura-Image-Quality-Classification)](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Quality-Classification/commits/main)
[![GitHub release](https://img.shields.io/github/v/release/AmanoShizukikun/Nagato-Sakura-Image-Quality-Classification)](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Quality-Classification/releases)

\[ 中文 | [English](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Quality-Classification/blob/main/assets/docs/README_en.md) | [日本語](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Quality-Classification/blob/main/assets/docs/README_jp.md) \]

## 簡介
Nagato-Sakura-Image-Quality-Classification 是「長門櫻計畫」的其中一個分支，是用來進行圖像品質評估的 AI 圖像評分程式。

## 公告
更改了模型架構，使用 Transformer 模型取代了以前的 CNN 模型。

## 近期變動
### 1.0.0 (2025 年 3 月 13 日)
![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Quality-Classification/blob/main/assets/preview/1.0.0.jpg)
### 重要變更
- 【重大】首個正式發行版本，使用 Transformer 模型取代了以前的 CNN 模型。
### 新增功能
- 【新增】添加了自動保存最佳損失函數模型的功能，可以保存最佳損失函數模型以及最終訓練步數模型。
- 【更新】改善了訓練效率並優化了模型的損失函數。
### 已知問題
- N/A

[所有發行版本](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Quality-Classification/blob/main/assets/docs/Changelog.md)

## 快速開始
> [!NOTE]
> 如果非 NVIDIA 顯示卡只需安裝前兩項即可。
### 環境設置
- **Python 3**
  - 下載: https://www.python.org/downloads/windows/
- **PyTorch**
  - 下載: https://pytorch.org/
- NVIDIA GPU驅動程式
  - 下載: https://www.nvidia.com/zh-tw/geforce/drivers/
- NVIDIA CUDA Toolkit
  - 下載: https://developer.nvidia.com/cuda-toolkit
- NVIDIA cuDNN
  - 下載: https://developer.nvidia.com/cudnn
> [!TIP]
> 請按照當前 PyTorch 支援安裝對應的 CUDA 版本。
### 安裝倉庫
> [!IMPORTANT]
> 此為必要步驟。
```shell
git clone https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Quality-Classification.git
cd Nagato-Sakura-Image-Quality-Classification
pip install -r requirements.txt
```
- 訓練前置準備
```shell
python data_processor.py
```

- 開始訓練
```shell
python train_transformer.py
```

- 開始測試
```shell
python test_transformer.py + 圖片路徑
```

## 待辦事項
N/A

## 致謝
特別感謝以下項目和貢獻者：
### 項目
N/A
### 貢獻者
<a href="https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Quality-Classification/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=AmanoShizukikun/Nagato-Sakura-Image-Quality-Classification" />
</a>
