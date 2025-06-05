import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QLabel, QPushButton, QStyle, QTabBar, QStylePainter,
    QStyleOptionTab
)
import matplotlib

from com.betool.ui.BELibPage import BELibPage
from com.betool.ui.BaseEditorPage import BaseEditorPage
from com.betool.ui.CenteredTabBar import CenteredTabBar
from com.betool.ui.HomePage import HomePage
from com.betool.ui.SgRNADesignPage import SgRNADesignPage

matplotlib.use('Agg')
# 环境变量配置
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = f"{os.environ['CONDA_PREFIX']}/plugins/platforms"
os.environ["DYLD_FRAMEWORK_PATH"] = f"{os.environ['CONDA_PREFIX']}/lib"
os.environ['PYTORCH_JIT'] = '0'

from PyQt5.QtWidgets import QTabBar, QStyle, QStyleOptionTab
from PyQt5.QtCore import Qt, QRect, QSize, QSettings
from PyQt5.QtGui import QPainter, QPen, QPalette, QIcon, QFontMetrics, QPixmap, QFont, QBrush

from PyQt5.QtWidgets import QTabBar, QStyle, QStyleOptionTab
from PyQt5.QtCore import Qt, QRect, QSize
from PyQt5.QtGui import QPainter, QPen, QPalette, QIcon, QFontMetrics

from PyQt5.QtWidgets import QTabBar, QStyle, QStyleOptionTab
from PyQt5.QtCore import Qt, QRect, QSize
from PyQt5.QtGui import QPainter, QPen, QPalette, QIcon, QFontMetrics




class BEToolUI(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("BE-Tool")

        # 初始化窗口尺寸设置（默认：800x600）
        self.settings = QSettings("BE-Tool", "UI")
        self.load_window_settings()

        # 保存原始图片引用
        self.original_pixmap = None

        # 设置为无边距
        self.setContentsMargins(0, 0, 0, 0)

        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 创建主TabWidget
        self.main_tab_widget = QTabWidget()
        self.main_tab_widget.setStyleSheet("""
               QTabWidget::pane {
                   border: none;
                   background: transparent;
               }
               QTabBar::tab {
                   background: rgba(240, 240, 240, 0.9);
                   color: #333;
                   border: 1px solid #ccc;
                   border-bottom: none;
                   border-radius: 4px 4px 0 0;
                   padding: 8px 16px;
                   margin-right: 2px;
                   min-width: 100px;
                   height: 40px; /* 增加选项卡高度以容纳图标和文字 */
               }
               QTabBar::tab:selected {
                   background: white;
                   color: #4a86e8;
                   font-weight: bold;
                   border-color: #4a86e8;
                   border-bottom: 1px solid white;
                   margin-bottom: -1px;
               }
               QTabBar::tab:hover {
                   background: rgba(245, 245, 245, 0.95);
               }
               QTabBar::tab:!selected {
                   margin-top: 5px; /* 非选中标签下移，使选中标签突出 */
               }
           """)

        # 使用自定义TabBar实现水平居中
        centered_tab_bar = CenteredTabBar()
        self.main_tab_widget.setTabBar(centered_tab_bar)

        main_layout.addWidget(self.main_tab_widget)
        # 加载自定义图标示例
        home_icon = QIcon(QPixmap(get_resource_path("../resources/home.png")))
        design_icon = QIcon(QPixmap(get_resource_path("../resources/sgrna.png")))
        editor_icon = QIcon(QPixmap(get_resource_path("../resources/editor.png")))
        library_icon = QIcon(QPixmap(get_resource_path("../resources/library.png")))



        # 创建第一个标签页（图片标签页）
        tab1 = HomePage()
        tab1.setStyleSheet("background: transparent;")
        tab1_layout = QVBoxLayout(tab1)
        tab1_layout.setContentsMargins(0, 0, 0, 0)  # 无边距，使图片完全填充

        # 创建第二个标签页
        tab2 = SgRNADesignPage()
        tab2.setStyleSheet("background: transparent;")
        tab2_layout = QVBoxLayout(tab2)
        tab2_layout.setContentsMargins(0, 0, 0, 0)
        tab2_layout.addStretch()  # 添加伸缩项使内容居中

        # 创建第三个标签页
        tab3 = BaseEditorPage()
        tab3.setStyleSheet("background: transparent;")
        tab3_layout = QVBoxLayout(tab3)
        tab3_layout.setContentsMargins(0, 0, 0, 0)
        tab3_layout.addStretch()  # 添加伸缩项使内容居中

        # 创建第4个标签页
        tab4 = BELibPage()
        tab4.setStyleSheet("background: transparent;")
        tab4_layout = QVBoxLayout(tab4)
        tab4_layout.setContentsMargins(0, 0, 0, 0)
        tab4_layout.addStretch()  # 添加伸缩项使内容居中

        # 添加标签页并设置图标
        self.main_tab_widget.addTab(tab1, home_icon, "Home")
        self.main_tab_widget.addTab(tab2, design_icon, "sgRNA Design")
        self.main_tab_widget.addTab(tab3, editor_icon, "BaseEditor")
        self.main_tab_widget.addTab(tab4, library_icon, "BE Lib")

        # 加载背景图片
        image_path = get_resource_path("../resources/background.png")
        self.original_pixmap = QPixmap(image_path)

        if not self.original_pixmap.isNull():
            # 直接设置HomePage的背景，不使用额外容器
            self.update_background()

            # 窗口大小变化时更新背景和标签页图片
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

        # 窗口显示后再更新标签页图片
        self.showEvent = self._handle_show

    def load_window_settings(self):
        """加载窗口尺寸（从配置中读取或使用默认值）"""
        width = self.settings.value("window/width", 1000, int)
        height = self.settings.value("window/height", 600, int)
        self.resize(width, height)  # 直接设置窗口尺寸

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

    def update_tab1_image(self):
        """更新标签页1的图片"""
        if not hasattr(self, 'tab1_pixmap') or self.tab1_pixmap.isNull():
            return

        # 检查当前标签页是否存在
        current_widget = self.main_tab_widget.currentWidget()
        if current_widget is None:
            return

        # 获取标签页内容区域大小
        tab_rect = current_widget.rect()

        # 缩放图片以填充标签页
        scaled_pixmap = self.tab1_pixmap.scaled(
            tab_rect.width(),
            tab_rect.height(),
            Qt.KeepAspectRatioByExpanding,  # 保持比例并扩展填充
            Qt.SmoothTransformation
        )

        self.image_label.setPixmap(scaled_pixmap)

    def _handle_resize(self, event):
        """处理窗口大小变化事件"""
        # 立即更新背景和标签页图片
        self.update_background()
        self.update_tab1_image()
        super().resizeEvent(event)

    def _handle_show(self, event):
        """处理窗口显示事件"""
        # 窗口显示后更新标签页图片
        self.update_tab1_image()
        super().showEvent(event)


def get_resource_path(relative_path):
    """获取资源文件路径"""
    try:
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(base_path, relative_path)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BEToolUI()
    window.show()
    sys.exit(app.exec_())