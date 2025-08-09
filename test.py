import os
import sys
import time
from pathlib import Path
from PIL import Image, ImageQt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from typing import Tuple, Optional
import threading

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QPushButton, QFileDialog, 
                           QTextEdit, QProgressBar, QScrollArea, QGridLayout,
                           QGroupBox, QSpinBox, QCheckBox, QFrame, QSplitter)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import QPixmap, QFont, QIcon, QPalette, QColor

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

class ImageQualityWorker(QThread):
    """背景執行緒進行圖像品質評分"""
    progress_updated = pyqtSignal(int, int)
    result_ready = pyqtSignal(str, float, str)
    batch_completed = pyqtSignal(dict)
    log_message = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.device = None
        self.image_paths = []
        self.transform = None
        self.is_cancelled = False
        
    def set_model(self, model, device):
        """設置模型和設備"""
        self.model = model
        self.device = device
        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def set_images(self, image_paths):
        """設置要處理的圖片路徑"""
        self.image_paths = image_paths
        self.is_cancelled = False
        
    def cancel(self):
        """取消處理"""
        self.is_cancelled = True
        
    def preprocess_image(self, image_path):
        """處理輸入圖片"""
        try:
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                return self.transform(img).unsqueeze(0)
        except Exception as e:
            self.log_message.emit(f"處理圖片失敗 {image_path}: {e}")
            return None

    def predict_quality(self, image_tensor):
        """預測圖片品質"""
        if image_tensor is None or self.model is None:
            return 0.0 
        try:
            with torch.no_grad():
                image_tensor = image_tensor.to(self.device)
                if self.device.type == 'cuda':
                    with torch.amp.autocast('cuda'):
                        prediction = self.model(image_tensor)
                else:
                    prediction = self.model(image_tensor)
                return prediction.item()
        except Exception as e:
            self.log_message.emit(f"預測失敗: {e}")
            return 0.0

    def run(self):
        """執行品質評分"""
        results = {}
        total = len(self.image_paths)
        
        for i, img_path in enumerate(self.image_paths):
            if self.is_cancelled:
                break
            self.log_message.emit(f"正在處理: {os.path.basename(img_path)}")
            img_tensor = self.preprocess_image(img_path)
            score = self.predict_quality(img_tensor)
            results[str(img_path)] = score
            self.result_ready.emit(str(img_path), score, str(img_path))
            self.progress_updated.emit(i + 1, total)
        if not self.is_cancelled:
            self.batch_completed.emit(results)

class ImageDisplayWidget(QLabel):
    """圖片顯示小工具"""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 200)
        self.setMaximumSize(400, 400)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setText("拖放圖片到此處\n或點擊選擇圖片")
        self.setAcceptDrops(True)
        
    def set_image(self, image_path, score=None):
        """設置顯示圖片"""
        try:
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(
                    self.width() - 10, self.height() - 10,
                    Qt.AspectRatioMode.KeepAspectRatio, 
                    Qt.TransformationMode.SmoothTransformation
                )
                self.setPixmap(scaled_pixmap)
                if score is not None:
                    filename = os.path.basename(image_path)
                    self.setToolTip(f"{filename}\n品質分數: {score:.2f}")
        except Exception as e:
            print(f"載入圖片失敗 {image_path}: {e}")

class NagatoSakuraIQAGUI(QMainWindow):
    """長門櫻圖像品質評分 GUI 主視窗"""
    def __init__(self):
        super().__init__()
        self.model = None
        self.device = None
        self.worker = ImageQualityWorker()
        self.results = {}
        self.init_ui()
        self.connect_signals()
        self.init_device()
        
    def init_ui(self):
        """初始化使用者介面"""
        self.setWindowTitle("Nagato-Sakura-Image-Quality-Assessment")
        self.setGeometry(100, 100, 1200, 800)
        icon_path = "./assets/icon/2.0.0.ico"
        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        self.create_control_panel(main_layout)
        self.create_result_panel(main_layout)
        self.statusBar().showMessage("就緒 - 請選擇圖片進行品質評分")
        
    def create_control_panel(self, parent_layout):
        """創建左側控制面板"""
        control_frame = QFrame()
        control_frame.setMaximumWidth(350)
        control_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        control_layout = QVBoxLayout(control_frame)
        title_label = QLabel("長門櫻圖像品質評分")
        title_label.setFont(QFont("微軟正黑體", 16, QFont.Weight.Bold))
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        control_layout.addWidget(title_label)
        model_group = QGroupBox("模型狀態")
        model_layout = QVBoxLayout(model_group)
        self.model_status_label = QLabel("尚未載入模型")
        model_layout.addWidget(self.model_status_label)
        self.device_label = QLabel("設備: 檢測中...")
        model_layout.addWidget(self.device_label)
        self.load_model_btn = QPushButton("選擇並載入模型")
        self.load_model_btn.clicked.connect(self.select_and_load_model)
        model_layout.addWidget(self.load_model_btn)
        control_layout.addWidget(model_group)
        selection_group = QGroupBox("選擇圖片")
        selection_layout = QVBoxLayout(selection_group)
        self.select_images_btn = QPushButton("選擇圖片檔案")
        self.select_images_btn.clicked.connect(self.select_images)
        self.select_images_btn.setEnabled(False)
        selection_layout.addWidget(self.select_images_btn)
        self.folder_btn = QPushButton("選擇圖片資料夾")
        self.folder_btn.clicked.connect(self.select_folder)
        self.folder_btn.setEnabled(False)
        selection_layout.addWidget(self.folder_btn)
        control_layout.addWidget(selection_group)
        options_group = QGroupBox("處理選項")
        options_layout = QVBoxLayout(options_group)
        self.show_preview_cb = QCheckBox("顯示圖片預覽")
        self.show_preview_cb.setChecked(True)
        options_layout.addWidget(self.show_preview_cb)
        self.auto_scroll_cb = QCheckBox("自動滾動到最新結果")
        self.auto_scroll_cb.setChecked(True)
        options_layout.addWidget(self.auto_scroll_cb)
        control_layout.addWidget(options_group)
        progress_group = QGroupBox("處理進度")
        progress_layout = QVBoxLayout(progress_group)
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        progress_layout.addWidget(self.progress_bar)
        self.progress_label = QLabel("")
        progress_layout.addWidget(self.progress_label)
        control_layout.addWidget(progress_group)
        action_layout = QHBoxLayout()
        self.cancel_btn = QPushButton("取消處理")
        self.cancel_btn.clicked.connect(self.cancel_processing)
        self.cancel_btn.setEnabled(False)
        action_layout.addWidget(self.cancel_btn)
        control_layout.addLayout(action_layout)
        self.clear_btn = QPushButton("清空結果")
        self.clear_btn.clicked.connect(self.clear_results)
        control_layout.addWidget(self.clear_btn)
        log_group = QGroupBox("處理日誌")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(150)
        self.log_text.setFont(QFont("Consolas", 9))
        log_layout.addWidget(self.log_text)
        control_layout.addWidget(log_group)
        control_layout.addStretch()
        parent_layout.addWidget(control_frame)
        
    def create_result_panel(self, parent_layout):
        """創建右側結果顯示面板"""
        splitter = QSplitter(Qt.Orientation.Vertical)
        stats_group = QGroupBox("評分統計")
        stats_layout = QVBoxLayout(stats_group)
        stats_layout.setContentsMargins(5, 5, 5, 5)
        self.stats_label = QLabel("尚未開始評分")
        self.stats_label.setFont(QFont("微軟正黑體", 9))
        self.stats_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.stats_label.setWordWrap(True)
        self.stats_label.setMinimumHeight(20)
        stats_layout.addWidget(self.stats_label)
        stats_group.setMaximumHeight(80)
        stats_group.setMinimumHeight(60)
        splitter.addWidget(stats_group)
        results_group = QGroupBox("評分結果")
        results_layout = QVBoxLayout(results_group)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.results_widget = QWidget()
        self.results_layout = QVBoxLayout(self.results_widget)
        self.results_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll_area.setWidget(self.results_widget)
        results_layout.addWidget(self.scroll_area)
        splitter.addWidget(results_group)
        parent_layout.addWidget(splitter)
        
    def connect_signals(self):
        """連接信號與槽"""
        self.worker.progress_updated.connect(self.update_progress)
        self.worker.result_ready.connect(self.add_result)
        self.worker.batch_completed.connect(self.on_batch_completed)
        self.worker.log_message.connect(self.add_log_message)
        
    def init_device(self):
        """初始化設備信息"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device_name = "GPU" if device.type == 'cuda' else "CPU"
        if device.type == 'cuda':
            device_name = f"GPU ({torch.cuda.get_device_name()})"
        self.device_label.setText(f"設備: {device_name}")
        
    def select_and_load_model(self):
        """選擇並載入模型"""
        model_path, _ = QFileDialog.getOpenFileName(
            self, "選擇模型檔案", "./models/", 
            "PyTorch 模型檔案 (*.pth *.pt);;所有檔案 (*.*)"
        )
        if model_path:
            self.load_model(model_path)
        
    def load_model(self, model_path="./models/NS-IQA.pth"):
        """載入模型"""
        self.load_model_btn.setEnabled(False)
        self.load_model_btn.setText("載入中...")
        
        def load_in_thread():
            try:
                if not os.path.exists(model_path):
                    self.update_model_status("模型檔案不存在", False)
                    return
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = NagatoSakuraImageQualityClassificationCNN(dropout_rate=0)
                model_state = torch.load(model_path, map_location=device)
                model.load_state_dict(model_state)
                model.to(device)
                model.eval()
                self.model = model
                self.device = device
                self.worker.set_model(model, device)
                device_name = "GPU" if device.type == 'cuda' else "CPU"
                if device.type == 'cuda':
                    device_name = f"GPU ({torch.cuda.get_device_name()})"
                self.update_model_status(f"模型已載入: {os.path.basename(model_path)}", True)
                self.device_label.setText(f"設備: {device_name}")
            except Exception as e:
                self.update_model_status(f"載入失敗: {str(e)}", False)
            finally:
                self.load_model_btn.setEnabled(True)
                self.load_model_btn.setText("選擇並載入模型")
                
        threading.Thread(target=load_in_thread, daemon=True).start()
        
    def update_model_status(self, message, success):
        """更新模型狀態"""
        self.model_status_label.setText(message)
        if success:
            self.select_images_btn.setEnabled(True)
            self.folder_btn.setEnabled(True)
        else:
            self.select_images_btn.setEnabled(False)
            self.folder_btn.setEnabled(False)
            
    def select_images(self):
        """選擇圖片檔案（支援單張或多張）"""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "選擇圖片檔案", "", 
            "圖片檔案 (*.jpg *.jpeg *.png *.bmp *.tiff)"
        )
        if file_paths:
            self.selected_images = file_paths
            if len(file_paths) == 1:
                self.add_log_message(f"已選擇圖片: {os.path.basename(file_paths[0])}")
            else:
                self.add_log_message(f"已選擇 {len(file_paths)} 張圖片")
            self.start_processing()
            
    def select_folder(self):
        """選擇圖片資料夾"""
        folder_path = QFileDialog.getExistingDirectory(self, "選擇圖片資料夾")
        if folder_path:
            extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
            image_files = []
            for ext in extensions:
                image_files.extend(Path(folder_path).glob(f'*{ext}'))
                image_files.extend(Path(folder_path).glob(f'*{ext.upper()}'))
            if image_files:
                self.selected_images = [str(f) for f in image_files]
                self.add_log_message(f"從資料夾選擇了 {len(image_files)} 張圖片")
                self.start_processing()
            else:
                self.add_log_message("資料夾中沒有找到圖片檔案")
                
    def start_processing(self):
        """開始處理"""
        if not hasattr(self, 'selected_images') or not self.selected_images:
            self.add_log_message("請先選擇圖片")
            return
        if self.model is None:
            self.add_log_message("模型尚未載入完成")
            return
        self.select_images_btn.setEnabled(False)
        self.folder_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_bar.setMaximum(len(self.selected_images))
        self.worker.set_images(self.selected_images)
        self.worker.start()
        self.add_log_message(f"開始處理 {len(self.selected_images)} 張圖片...")
        
    def cancel_processing(self):
        """取消處理"""
        self.worker.cancel()
        self.cancel_btn.setEnabled(False)
        self.add_log_message("正在取消處理...")
        
    def update_progress(self, current, total):
        """更新進度"""
        self.progress_bar.setValue(current)
        self.progress_label.setText(f"已處理: {current}/{total}")
        
    def add_result(self, image_path, score, display_path):
        """添加結果到顯示區域"""
        result_frame = QFrame()
        result_frame.setFrameStyle(QFrame.Shape.StyledPanel)
        result_frame.setMaximumHeight(120)
        result_layout = QHBoxLayout(result_frame)
        if self.show_preview_cb.isChecked():
            image_label = QLabel()
            image_label.setFixedSize(80, 80)
            image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            try:
                pixmap = QPixmap(image_path)
                if not pixmap.isNull():
                    scaled_pixmap = pixmap.scaled(
                        80, 80, 
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation
                    )
                    image_label.setPixmap(scaled_pixmap)
            except:
                image_label.setText("無法\n預覽")
                image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                
            result_layout.addWidget(image_label)
        info_layout = QVBoxLayout()
        filename_label = QLabel(os.path.basename(image_path))
        filename_label.setFont(QFont("微軟正黑體", 10, QFont.Weight.Bold))
        info_layout.addWidget(filename_label)
        path_label = QLabel(f"路徑: {image_path}")
        path_label.setWordWrap(True)
        info_layout.addWidget(path_label)
        score_label = QLabel(f"品質分數: {score:.2f}")
        score_label.setFont(QFont("微軟正黑體", 11, QFont.Weight.Bold))
        info_layout.addWidget(score_label)
        result_layout.addLayout(info_layout)
        result_layout.addStretch()
        self.results_layout.addWidget(result_frame)
        self.results[image_path] = score
        if self.auto_scroll_cb.isChecked():
            QTimer.singleShot(100, lambda: self.scroll_area.verticalScrollBar().setValue(
                self.scroll_area.verticalScrollBar().maximum()
            ))
            
    def on_batch_completed(self, results):
        """批次處理完成"""
        self.select_images_btn.setEnabled(True)
        self.folder_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.progress_bar.setVisible(False)
        if results:
            scores = list(results.values())
            stats_text = f"處理完成！圖片數: {len(scores)}  平均: {np.mean(scores):.2f}  最高: {max(scores):.2f}  最低: {min(scores):.2f}"
            self.stats_label.setText(stats_text)
            self.add_log_message("批次處理完成！")
        else:
            self.add_log_message("處理已取消或無有效結果")
            
    def clear_results(self):
        """清空結果"""
        while self.results_layout.count():
            child = self.results_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.results.clear()
        self.stats_label.setText("尚未開始評分")
        self.add_log_message("已清空所有結果")
        
    def add_log_message(self, message):
        """添加日誌訊息"""
        timestamp = time.strftime("%H:%M:%S")
        formatted_message = f"[{timestamp}] {message}"
        self.log_text.append(formatted_message)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

def main():
    """主函數"""
    app = QApplication(sys.argv)
    app.setApplicationName("Nagato-Sakura-Image-Quality-Assessment")
    app.setApplicationVersion("2.0.0")
    app.setOrganizationName("Nagato-Sakura-Image-Quality-Assessment")
    window = NagatoSakuraIQAGUI()
    window.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
