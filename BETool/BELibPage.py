import json
import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QLabel, QFrame, QLineEdit,
    QComboBox, QPushButton, QMessageBox, QTextEdit, QHBoxLayout, QScrollArea,
    QMessageBox
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QSize

# 设置matplotlib后端和环境变量
import matplotlib

matplotlib.use('Agg')
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = f"{os.environ['CONDA_PREFIX']}/plugins/platforms"
os.environ["DYLD_FRAMEWORK_PATH"] = f"{os.environ['CONDA_PREFIX']}/lib"
os.environ['PYTORCH_JIT'] = '0'

JSON_PATH = "be_data.json"


class BELibPage(QWidget):
    def __init__(self):
        super().__init__()
        self.be_data = []
        self.load_from_json()
        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)

        # ====================== 上部分：BE信息列表展示 ======================
        display_frame = QFrame()
        display_frame.setStyleSheet("""
            QFrame {
                background: #fcfcfc;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        display_layout = QVBoxLayout(display_frame)
        display_layout.setSpacing(5)

        # 标题
        title_label = QLabel("Existing BE Information")
        title_label.setStyleSheet("font-size: 13px; font-weight: bold; color: #333;")
        display_layout.addWidget(title_label)

        # 动态生成BE信息标签
        self.result_layout = QVBoxLayout()
        self.result_layout.setSpacing(0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setStyleSheet("QScrollArea { border: none; }")

        scroll_content = QWidget()
        scroll_content.setLayout(self.result_layout)
        scroll_area.setWidget(scroll_content)
        scroll_area.setMaximumHeight(200)
        display_layout.addWidget(scroll_area)

        # 加载初始数据
        self.update_display()

        # ====================== 下部分：新增BE信息 ======================
        input_frame = QFrame()
        input_frame.setStyleSheet("""
            QFrame {
                background: #fcfcfc;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 8px;
            }
        """)
        input_layout = QVBoxLayout(input_frame)
        input_layout.setSpacing(6)

        # BE Name输入行
        self.be_name_edit = QLineEdit()
        self.add_input_row(input_layout, "BE Name:", self.be_name_edit, label_width=90)

        # 编辑器类型行
        editor_row = QHBoxLayout()
        editor_row.setSpacing(8)

        self.editor_label = QLabel("Editor Type:")
        self.editor_label.setFixedWidth(90)
        self.editor_label.setStyleSheet(self.get_label_style())

        self.editor_combo = QComboBox()
        self.editor_combo.addItems(["ABE", "CBE", "Other BE"])
        self.editor_combo.currentIndexChanged.connect(self.toggle_other_be)
        self.editor_combo.setStyleSheet("""
            QComboBox {
                border: 1px solid #ddd;
                border-radius: 2px;
                padding: 3px 5px;
                background: #ffffff;
                color: #333;
                font-size: 12px;
            }
            QComboBox::drop-down {
                border-left: 1px solid #ddd;
            }
            QComboBox QAbstractItemView {
                selection-background-color: #007bff;
                selection-color: white;
                color: #333;
                border: 1px solid #ddd;
                background-color: white;
                padding: 2px;
                font-size: 12px;
            }
        """)

        # 增加标签宽度避免文字截断
        self.target_base_label = QLabel("Target Base:")
        self.target_base_label.setFixedWidth(120)
        self.target_base_label.setHidden(True)
        self.target_base_label.setStyleSheet(self.get_label_style())

        self.target_base_edit = QLineEdit()
        self.target_base_edit.setHidden(True)
        self.target_base_edit.setStyleSheet(self.get_input_style())

        self.conv_base_label = QLabel("Converted Base:")
        self.conv_base_label.setFixedWidth(140)
        self.conv_base_label.setHidden(True)
        self.conv_base_label.setStyleSheet(self.get_label_style())

        self.conv_base_edit = QLineEdit()
        self.conv_base_edit.setHidden(True)
        self.conv_base_edit.setStyleSheet(self.get_input_style())

        editor_row.addWidget(self.editor_label)
        editor_row.addWidget(self.editor_combo)
        editor_row.addWidget(self.target_base_label)
        editor_row.addWidget(self.target_base_edit)
        editor_row.addWidget(self.conv_base_label)
        editor_row.addWidget(self.conv_base_edit)

        input_layout.addLayout(editor_row)

        # 窗口起始和结束
        window_row = QHBoxLayout()
        window_row.setSpacing(8)

        self.window_start_label = QLabel("Window Start:")
        self.window_start_label.setFixedWidth(90)
        self.window_start_label.setStyleSheet(self.get_label_style())

        self.window_start_edit = QLineEdit()
        self.window_start_edit.setStyleSheet(self.get_input_style())

        self.window_end_label = QLabel("Window End:")
        self.window_end_label.setFixedWidth(90)
        self.window_end_label.setStyleSheet(self.get_label_style())

        self.window_end_edit = QLineEdit()
        self.window_end_edit.setStyleSheet(self.get_input_style())

        window_row.addWidget(self.window_start_label)
        window_row.addWidget(self.window_start_edit)
        window_row.addWidget(self.window_end_label)
        window_row.addWidget(self.window_end_edit)

        input_layout.addLayout(window_row)

        # BE描述
        self.be_desc_edit = QTextEdit()
        self.be_desc_edit.setPlaceholderText("Enter BE description...")
        self.be_desc_edit.setMinimumHeight(80)
        self.add_input_row(input_layout, "BE Desc:", self.be_desc_edit, label_width=90)

        # 提交按钮 - 使用水平布局实现居中
        button_layout = QHBoxLayout()
        button_layout.setSpacing(0)

        submit_btn = QPushButton("SUBMIT")
        submit_btn.setStyleSheet("""
            QPushButton {
                background: #007bff;
                color: white;
                padding: 5px 20px;
                border-radius: 3px;
                font-size: 12px;
            }
            QPushButton:hover {
                background: #0069d9;
            }
            QPushButton:pressed {
                background: #0056b3;
            }
        """)
        submit_btn.setFixedHeight(28)

        # 绑定按钮点击事件
        submit_btn.clicked.connect(self.submit_data)

        # 使用伸缩项使按钮居中
        button_layout.addStretch(1)
        button_layout.addWidget(submit_btn)
        button_layout.addStretch(1)

        input_layout.addLayout(button_layout)

        # 组合布局
        main_layout.addWidget(display_frame)
        main_layout.addWidget(input_frame)

    def add_input_row(self, parent_layout, label_text, widget, label_width=90):
        """通用输入行添加方法"""
        row = QHBoxLayout()
        row.setSpacing(8)

        label = QLabel(label_text)
        label.setFixedWidth(label_width)
        label.setStyleSheet(self.get_label_style())

        widget.setStyleSheet(self.get_input_style())

        row.addWidget(label)
        row.addWidget(widget)
        parent_layout.addLayout(row)

    def get_input_style(self):
        """统一输入控件样式（包括边框）"""
        return """
            QLineEdit, QTextEdit {
                border: 1px solid #ddd;
                border-radius: 2px;
                padding: 3px 5px;
                background: #ffffff;
                color: #333;
                font-size: 12px;
            }
            QLineEdit:focus, QTextEdit:focus {
                border-color: #007bff;
            }
        """

    def get_label_style(self):
        """统一标签样式（无边框）"""
        return """
            QLabel {
                color: #333;
                font-size: 12px;
                border: none;
            }
        """

    def toggle_other_be(self):
        """切换Other BE时显示碱基输入框"""
        show = self.editor_combo.currentText() == "Other BE"
        self.target_base_label.setHidden(not show)
        self.target_base_edit.setHidden(not show)
        self.conv_base_label.setHidden(not show)
        self.conv_base_edit.setHidden(not show)

    def load_from_json(self):
        """从JSON文件加载BE信息"""
        try:
            with open(JSON_PATH, "r") as f:
                self.be_data = json.load(f)
        except FileNotFoundError:
            self.be_data = []

    def save_to_json(self):
        """将BE信息保存到JSON文件"""
        with open(JSON_PATH, "w") as f:
            json.dump(self.be_data, f, indent=2)

    def update_display(self):
        """更新BE信息展示"""
        while self.result_layout.count() > 0:
            item = self.result_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

        for idx, data in enumerate(self.be_data, 1):
            line_text = f"BE #{idx}: {data['BE name']} | " \
                        f"Start: {data['start']} | End: {data['end']} | " \
                        f"Target: {data['target base'] or 'N/A'} → {data['converted base'] or 'N/A'}"

            # 创建水平布局，包含标签和删除按钮
            item_layout = QHBoxLayout()

            label = QLabel(line_text)
            label.setStyleSheet(f"""
                QLabel {{
                    color: #333;
                    font-size: 11px;
                    padding: 3px 6px;
                    background-color: {"#f9f9f9" if idx % 2 == 0 else "#fff"};
                }}
            """)
            label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            label.setWordWrap(True)

            delete_btn = QPushButton("DELETE")
            delete_btn.setStyleSheet("""
                QPushButton {
                    background: #dc3545;
                    color: white;
                    padding: 2px 8px;
                    border-radius: 2px;
                    font-size: 10px;
                    margin: 2px;
                }
                QPushButton:hover {
                    background: #c82333;
                }
                QPushButton:pressed {
                    background: #bd2130;
                }
            """)
            delete_btn.setFixedHeight(20)
            delete_btn.setFixedWidth(70)

            # 使用lambda传递当前索引，并捕获idx参数
            delete_btn.clicked.connect(lambda checked, i=idx - 1: self.delete_be(i))

            item_layout.addWidget(label, 1)  # 标签占据大部分空间
            item_layout.addWidget(delete_btn)

            # 创建一个框架来包含整行，添加底部边框
            row_frame = QFrame()
            row_frame.setLayout(item_layout)
            row_frame.setStyleSheet("QFrame { border-bottom: 1px solid #eee; }")

            self.result_layout.addWidget(row_frame)

    def delete_be(self, index):
        """删除指定索引的BE信息"""
        # 确认对话框
        reply = QMessageBox.question(
            self, "Confirm Deletion",
            f"Are you sure you want to delete BE #{index + 1}: {self.be_data[index]['BE name']}?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            # 从列表中删除
            del self.be_data[index]
            # 保存到JSON文件
            self.save_to_json()
            # 更新显示
            self.update_display()
            # 显示成功消息
            QMessageBox.information(self, "Success", "BE information deleted successfully!")

    def submit_data(self):
        """处理提交数据"""
        # 校验BE名称
        be_name = self.be_name_edit.text().strip()
        if not be_name:
            QMessageBox.warning(self, "Input Error", "Please enter a BE name!")
            return

        # 校验起始位置
        try:
            start = int(self.window_start_edit.text())
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Window Start must be an integer!")
            return

        # 校验结束位置
        try:
            end = int(self.window_end_edit.text())
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Window End must be an integer!")
            return

        # 校验起始和结束位置关系
        if start >= end:
            QMessageBox.warning(self, "Input Error", "Window Start must be less than Window End!")
            return

        # 获取碱基转换信息
        editor_type = self.editor_combo.currentText()
        if editor_type == "Other BE":
            target_base = self.target_base_edit.text().strip()
            converted_base = self.conv_base_edit.text().strip()
            if not target_base or not converted_base:
                QMessageBox.warning(self, "Input Error", "Target Base and Converted Base are required for Other BE!")
                return
        else:
            # 根据编辑器类型自动设置碱基转换
            target_base = "A" if editor_type == "ABE" else "C"
            converted_base = "G" if editor_type == "ABE" else "T"

        # 获取描述信息
        be_desc = self.be_desc_edit.toPlainText().strip()
        if not be_desc:
            QMessageBox.warning(self, "Input Error", "Please enter a BE description!")
            return

        # 创建新数据对象
        new_data = {
            "BE name": be_name,
            "start": start,
            "end": end,
            "target base": target_base,
            "converted base": converted_base,
            "be desc": be_desc
        }

        # 添加到数据列表并保存
        self.be_data.append(new_data)
        self.save_to_json()
        self.update_display()
        self.clear_inputs()

        # 显示成功消息
        QMessageBox.information(self, "Success", "BE information added successfully!")

    def clear_inputs(self):
        """清空输入框"""
        for widget in [self.be_name_edit, self.window_start_edit, self.window_end_edit,
                       self.target_base_edit, self.conv_base_edit, self.be_desc_edit]:
            if isinstance(widget, QLineEdit):
                widget.clear()
            elif isinstance(widget, QTextEdit):
                widget.setPlainText("")
        self.editor_combo.setCurrentIndex(0)
        self.toggle_other_be()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BELibPage()
    window.setWindowTitle("BE Library Manager")
    window.setGeometry(100, 100, 700, 500)
    window.show()
    sys.exit(app.exec_())