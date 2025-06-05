import random
import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QComboBox, QPushButton, QTableWidget, QTableWidgetItem,
    QFrame, QScrollArea, QSizePolicy, QMessageBox, QTextEdit
)
import matplotlib

matplotlib.use('Agg')
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = f"{os.environ['CONDA_PREFIX']}/plugins/platforms"
os.environ["DYLD_FRAMEWORK_PATH"] = f"{os.environ['CONDA_PREFIX']}/lib"
os.environ['PYTORCH_JIT'] = '0'

from PyQt5.QtGui import QFont, QPalette, QColor
from PyQt5.QtCore import Qt, QSize


class BELibPage(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        content_widget = QWidget()
        main_layout = QVBoxLayout(content_widget)
        main_layout.setContentsMargins(15, 15, 15, 15)
        main_layout.setSpacing(10)

        input_frame = QFrame()
        input_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        input_frame.setStyleSheet("""
            QFrame {
                background: #fcfcfc;
                border: 1px solid #ddd;
                border-radius: 6px;
                padding: 10px;
            }
        """)
        input_layout = QVBoxLayout(input_frame)
        input_layout.setSpacing(8)

        # BE Name输入行
        self.be_name_edit = QLineEdit()
        self.add_input_row(input_layout, "BE Name:", self.be_name_edit, label_width=130)

        # 编辑器类型行
        editor_row = QHBoxLayout()
        editor_row.setSpacing(10)

        editor_label = QLabel("Editor Type:")
        editor_label.setStyleSheet("color: #555; border: none;")
        editor_label.setFixedWidth(110)
        editor_row.addWidget(editor_label)

        self.editor_combo = QComboBox()
        self.editor_combo.addItems(["ABE", "CBE", "Other BE"])
        self.editor_combo.setStyleSheet(self.get_input_style())
        self.editor_combo.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        self.editor_combo.setFixedWidth(140)
        self.editor_combo.currentIndexChanged.connect(self.toggle_other_be)
        editor_row.addWidget(self.editor_combo)

        editor_row.setAlignment(Qt.AlignTop)

        # Target Base
        self.target_base_label = QLabel("Target Base:")
        self.target_base_label.setStyleSheet("color: #555; border: none;")
        self.target_base_label.setFixedWidth(120)
        self.target_base_label.setHidden(True)
        editor_row.addWidget(self.target_base_label)

        self.target_base_edit = QLineEdit()
        self.target_base_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.target_base_edit.setHidden(True)
        self.target_base_edit.setStyleSheet(self.get_input_style())
        editor_row.addWidget(self.target_base_edit)

        # Converted Base
        self.conv_base_label = QLabel("Converted Base:")
        self.conv_base_label.setStyleSheet("color: #555; border: none;")
        self.conv_base_label.setFixedWidth(140)
        self.conv_base_label.setHidden(True)
        editor_row.addWidget(self.conv_base_label)

        self.conv_base_edit = QLineEdit()
        self.conv_base_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.conv_base_edit.setHidden(True)
        self.conv_base_edit.setStyleSheet(self.get_input_style())
        editor_row.addWidget(self.conv_base_edit)

        input_layout.addLayout(editor_row)

        # 窗口起始和结束
        window_row = QHBoxLayout()
        window_row.setSpacing(10)

        self.window_start_label = QLabel("Window Start:")
        self.window_start_label.setStyleSheet("color: #555; border: none;")
        self.window_start_label.setFixedWidth(110)
        window_row.addWidget(self.window_start_label)

        self.window_start_edit = QLineEdit()
        self.window_start_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.window_start_edit.setStyleSheet(self.get_input_style())
        window_row.addWidget(self.window_start_edit)

        self.window_end_label = QLabel("Window End:")
        self.window_end_label.setStyleSheet("color: #555; border: none;")
        self.window_end_label.setFixedWidth(110)
        window_row.addWidget(self.window_end_label)

        self.window_end_edit = QLineEdit()
        self.window_end_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.window_end_edit.setStyleSheet(self.get_input_style())
        window_row.addWidget(self.window_end_edit)

        input_layout.addLayout(window_row)

        # BE描述多行文本框
        self.be_desc_edit = QTextEdit()
        self.be_desc_edit.setPlaceholderText("Enter BE description here...")
        self.be_desc_edit.setStyleSheet(self.get_input_style())
        self.be_desc_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.be_desc_edit.setMinimumHeight(100)  # 设置最小高度
        self.add_input_row(input_layout, "BE Desc:", self.be_desc_edit, label_width=130)

        # 提交按钮
        submit_btn = QPushButton("SUBMIT")
        submit_btn.setStyleSheet("""
            QPushButton {
                background: #007bff;
                color: #fff;
                border: none;
                padding: 6px 12px;
                border-radius: 3px;
                font-size: 12px;
            }
            QPushButton:hover {
                background: #0056b3;
            }
        """)
        submit_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        submit_btn.clicked.connect(self.simulate_submit)
        input_layout.addWidget(submit_btn, alignment=Qt.AlignCenter)

        main_layout.addWidget(input_frame)





        scroll_area.setWidget(content_widget)

        # 确保只设置一次布局
        if not self.layout():
            page_layout = QVBoxLayout(self)
            page_layout.setContentsMargins(0, 0, 0, 0)
            page_layout.addWidget(scroll_area)

    def add_input_row(self, parent_layout, label_text, widget, label_width=110):
        """通用添加输入行方法，支持自定义标签宽度"""
        row_layout = QHBoxLayout()
        label = QLabel(label_text)
        label.setStyleSheet("color: #555; border: none;")
        label.setFixedWidth(label_width)
        row_layout.addWidget(label)

        widget.setStyleSheet(self.get_input_style())
        widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        row_layout.addWidget(widget)
        parent_layout.addLayout(row_layout)

    def get_input_style(self):
        # 保持与SgRNADesignPage一致的样式
        return """
            QLineEdit, QComboBox, QTextEdit {
                border: 1px solid #ddd;
                border-radius: 2px;
                padding: 3px 5px;
                background: #ffffff;
                color: #000;
                min-height: 24px;
                font-size: 12px;
            }
            QLineEdit:focus, QComboBox:focus, QTextEdit:focus {
                border-color: #007bff;
                color: #000;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 20px;
                border-left-width: 1px;
                border-left-color: #ddd;
                border-left-style: solid;
                border-top-right-radius: 2px;
                border-bottom-right-radius: 2px;
            }
            QComboBox::down-arrow {
                image: url(data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAxNiAxNiI+PHBhdGggZmlsbD0iIzAwMCIgZD0iTTE1LjUsNC41TDgsMTJsLTcuNS03LjVMMywzbDUsNWw1LTUgWiIvPjwvc3ZnPg==);
                width: 10px;
                height: 10px;
                margin-right: 4px;
            }
            QComboBox QAbstractItemView {
                border: 1px solid #ddd;
                selection-background-color: #e0e0e0;
                selection-color: #000;
                color: #000;
                background-color: #ffffff;
                padding: 2px;
                font-size: 12px;
            }
        """

    def toggle_other_be(self):
        """切换Other BE时显示/隐藏额外输入"""
        if self.editor_combo.currentText() == "Other BE":
            self.target_base_label.setHidden(False)
            self.conv_base_label.setHidden(False)
            self.target_base_edit.setHidden(False)
            self.conv_base_edit.setHidden(False)
        else:
            self.target_base_label.setHidden(True)
            self.conv_base_label.setHidden(True)
            self.target_base_edit.setHidden(True)
            self.conv_base_edit.setHidden(True)

    def simulate_submit(self):
        """模拟提交逻辑，生成并显示模拟数据"""
        be_name = self.be_name_edit.text().strip()
        editor_type = self.editor_combo.currentText()
        window_start = self.window_start_edit.text().strip()
        window_end = self.window_end_edit.text().strip()
        target_base = self.target_base_edit.text().strip() if editor_type == "Other BE" else ""
        conv_base = self.conv_base_edit.text().strip() if editor_type == "Other BE" else ""
        be_desc = self.be_desc_edit.toPlainText().strip()

        # 校验必填字段
        if not all([be_name, editor_type, window_start, window_end, be_desc]):
            QMessageBox.warning(self, "输入缺失", "请填写所有必填字段")
            return

        # 校验数字字段
        try:
            window_start = int(window_start)
            window_end = int(window_end)
            if window_start >= window_end or window_start < 1 or window_end < 1:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "参数错误", "窗口范围必须为有效整数且Start < End")
            return

        # 生成模拟数据
        mock_data = [
            [be_name, editor_type, f"{window_start}-{window_end}", target_base, conv_base, be_desc],
            ["编辑效率", "85%", "PAM 要求", "NGG", "Off-target", "低"],
            ["活性窗口", "4-8nt", "温度适应性", "37°C", "pH 范围", "7.0-8.0"],
            ["递送方式", "RNP", "表达系统", "哺乳动物", "研发状态", "已验证"]
        ]

        # 清空旧结果
        while self.result_layout.count() > 0:
            item = self.result_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        # 结果展示
        headers = ["属性", "详情", "属性", "详情", "属性", "详情"]
        header_label = QLabel(" | ".join([f"{h}: {v}" for h, v in zip(headers, mock_data[0])]))
        header_label.setStyleSheet("""
            QLabel {
                color: #333;
                font-size: 12px;
                font-weight: bold;
                padding: 6px 8px;
                border-bottom: 1px solid #ccc;
                background-color: #f5f5f5;
            }
        """)
        header_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        header_label.setWordWrap(True)
        self.result_layout.addWidget(header_label)

        # 添加分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Sunken)
        self.result_layout.addWidget(line)

        # 展示其余数据
        for i, row in enumerate(mock_data[1:], start=1):
            line_text = " | ".join([f"{h}: {v}" for h, v in zip(headers, row)])
            line_label = QLabel(line_text)
            line_label.setStyleSheet(f"""
                QLabel {{
                    color: #333;
                    font-size: 12px;
                    padding: 4px 8px;
                    border-bottom: 1px solid #eee;
                    background-color: {"#fafafa" if i % 2 == 0 else "#fff"};
                }}
                QLabel:hover {{
                    background-color: #f0f0f0;
                }}
            """)
            line_label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            line_label.setWordWrap(True)
            self.result_layout.addWidget(line_label)

        # 添加成功消息
        success_label = QLabel("✓ BE 信息已成功添加到库中!")
        success_label.setStyleSheet("""
            QLabel {
                color: #28a745;
                font-size: 12px;
                font-weight: bold;
                padding: 6px 8px;
                border-top: 1px solid #ccc;
                background-color: #f0f8f2;
            }
        """)
        success_label.setAlignment(Qt.AlignCenter)
        self.result_layout.addWidget(success_label)



if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BELibPage()
    window.show()
    sys.exit(app.exec_())