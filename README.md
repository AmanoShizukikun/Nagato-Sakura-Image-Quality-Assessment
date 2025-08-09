# Nagato-Sakura-Image-Quality-Assessment

[![GitHub Repo stars](https://img.shields.io/github/stars/AmanoShizukikun/Nagato-Sakura-Image-Quality-Assessment?style=social)](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Quality-Assessment/stargazers)
[![GitHub last commit](https://img.shields.io/github/last-commit/AmanoShizukikun/Nagato-Sakura-Image-Quality-Assessment)](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Quality-Assessment/commits/main)
[![GitHub release](https://img.shields.io/github/v/release/AmanoShizukikun/Nagato-Sakura-Image-Quality-Assessment)](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Quality-Assessment/releases)

\[ 中文 | [English](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Quality-Assessment/blob/main/assets/docs/README_en.md) | [日本語](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Quality-Assessment/blob/main/assets/docs/README_jp.md) \]

## 簡介
Nagato-Sakura-Image-Quality-Assessment 是「長門櫻計畫」的其中一個分支，是用來進行圖像品質評估的 AI 圖像評分程式。

## 公告
重新修改了倉庫配置並新增預設模型，針對「長門櫻-影像魅影」進行適配。

## 近期變動
### 2.0.0 (2025 年 8 月 10 日)
![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Quality-Assessment/blob/main/assets/preview/2.0.0.jpg)
### 重要變更
- 【重大】更改了倉庫名從 Nagato-Sakura-Image-Quality-Classification 改為 Nagato-Sakura-Image-Quality-Assessment。
- 【重大】更改了模型架構，使用輕量化的 CNN 2代模型取代了肥大的 Transformer 模型。
- 【重大】修改倉庫配置新增預設模型，針對「長門櫻-影像魅影」擴充插件進行適配。
### 新增功能
- 【新增】圖形化操作介面(GUI)，方便快速選擇模型、批量處裡圖片。
### 已知問題
- N/A

### 1.0.0 (2025 年 3 月 13 日)
![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Quality-Assessment/blob/main/assets/preview/1.0.0.jpg)
### 重要變更
- 【重大】首個正式發行版本，使用 Transformer 模型取代了以前的 CNN 模型。
### 新增功能
- 【新增】添加了自動保存最佳損失函數模型的功能，可以保存最佳損失函數模型以及最終訓練步數模型。
- 【更新】改善了訓練效率並優化了模型的損失函數。
### 已知問題
- N/A

[所有發行版本](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Quality-Assessment/blob/main/assets/docs/Changelog.md)

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
git clone https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Quality-Assessment.git
cd Nagato-Sakura-Image-Quality-Assessment
pip install -r requirements.txt
```

- 開始訓練
```shell
python train.py
```

- 開始測試 (GUI)
```shell
python test.py
```

## GUI 介面
![t2i](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Quality-Assessment/blob/main/assets/samples/GUI_v2.0.0.png)

## 待辦事項
N/A

## 致謝
特別感謝以下項目和貢獻者：
### 項目
- [Nagato-Sakura-Image-Charm](https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Charm)

### 貢獻者
<a href="https://github.com/AmanoShizukikun/Nagato-Sakura-Image-Quality-Assessment/graphs/contributors" target="_blank">
  <img src="https://contrib.rocks/image?repo=AmanoShizukikun/Nagato-Sakura-Image-Quality-Assessment" />
</a>
