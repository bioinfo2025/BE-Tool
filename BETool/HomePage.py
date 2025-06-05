import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QLabel
)
import matplotlib

matplotlib.use('Agg')
# 环境变量配置
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = f"{os.environ['CONDA_PREFIX']}/plugins/platforms"
os.environ["DYLD_FRAMEWORK_PATH"] = f"{os.environ['CONDA_PREFIX']}/lib"
os.environ['PYTORCH_JIT'] = '0'

from PyQt5.QtGui import QFont, QPalette, QBrush, QPixmap
from PyQt5.QtCore import Qt


class HomePage(QWidget):
    def __init__(self):
        super().__init__()

        # 保存原始图片引用
        self.original_pixmap = None

        # 设置为无边距
        self.setContentsMargins(0, 0, 0, 0)

        # 加载背景图片
        image_path = get_resource_path("../../../resources/BASE-CRISPR.png")
        self.original_pixmap = QPixmap(image_path)

        if not self.original_pixmap.isNull():
            # 直接设置HomePage的背景，不使用额外容器
            self.update_background()

            # 窗口大小变化时更新背景
            self.resizeEvent = self._handle_resize
        else:
            # 图片加载失败时显示错误信息
            error_label = QLabel("Failed to load background image", self)
            error_label.setFont(QFont("Arial", 16, QFont.Bold))
            error_label.setAlignment(Qt.AlignCenter)
            error_label.setStyleSheet("color: red;")

            # 创建布局并添加错误标签
            layout = QVBoxLayout(self)
            layout.addWidget(error_label)

    def update_background(self):
        """更新背景图片"""
        if self.original_pixmap and not self.original_pixmap.isNull():
            # 使用IgnoreAspectRatio参数强制图片填充整个窗口
            scaled_pixmap = self.original_pixmap.scaled(
                self.size(),
                Qt.IgnoreAspectRatio,  # 关键修改：忽略宽高比，强制填充
                Qt.SmoothTransformation  # 平滑缩放
            )

            # 设置HomePage自身的背景
            palette = self.palette()
            palette.setBrush(QPalette.Window, QBrush(scaled_pixmap))
            self.setPalette(palette)
            self.setAutoFillBackground(True)

    def _handle_resize(self, event):
        """处理窗口大小变化事件"""
        # 立即更新背景，无需定时器
        self.update_background()
        super().resizeEvent(event)


def get_resource_path(relative_path):
    """获取资源文件路径"""
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)