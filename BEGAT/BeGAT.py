import sys
import os
import torch
from torch.utils.data import Dataset, DataLoader
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox, QPushButton, \
    QTableWidget, QTableWidgetItem, QFrame, QScrollArea, QSizePolicy, QMessageBox
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
import matplotlib
from sklearn.metrics import accuracy_score
import joblib

matplotlib.use('Agg')
os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = f"{os.environ['CONDA_PREFIX']}/plugins/platforms"
os.environ["DYLD_FRAMEWORK_PATH"] = f"{os.environ['CONDA_PREFIX']}/lib"
os.environ['PYTORCH_JIT'] = '0'


# 模拟模型相关类（需根据实际模型调整）
class BaseEditorEncoder(torch.nn.Module):
    def __init__(self, editor_type, window_start, window_end):
        super().__init__()
        self.editor_type = editor_type
        self.window = (window_start, window_end)


def load_begat_model(model_path, example_editor):
    return joblib.load(model_path)


def create_dummy_data(n):
    return [{"sgrna": "dummy_sgrna", "target": "dummy_target", "editor_type": "CBE",
             "window_start": 5, "window_end": 8, "edit_rate": 0.8} for _ in range(n)]


def predict_edit_rate(model, sgrna, target, editor_type, window_start, window_end):
    # 模拟预测逻辑，需替换为实际模型推理代码
    return torch.tensor([0.75])  # 示例返回值


# 主窗口类
class SgRNADesignPage(QWidget):
    def __init__(self):
        super().__init__()
        self.model = None
        self.example_editor = None
        self.loaded_model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.init_ui()
        self.load_model()  # 初始化时加载模型

    def init_ui(self):
        # ... [之前的UI初始化代码，保持不变] ...

        # 修改后的Submit按钮事件绑定
        submit_btn = QPushButton("SUBMIT")
        submit_btn.clicked.connect(self.handle_submit)  # 绑定事件处理函数
        # ... [其他UI代码保持不变] ...

    def load_model(self):
        """加载模型"""
        try:
            self.example_editor = BaseEditorEncoder(editor_type="CBE", window_start=5, window_end=8)
            self.loaded_model = load_begat_model('begat_model.pkl', self.example_editor)
            self.loaded_model = self.loaded_model.to(self.device)
            print(f"模型加载成功，使用设备: {self.device}")
        except Exception as e:
            QMessageBox.critical(self, "模型加载失败", f"加载模型时出错: {str(e)}")

    def validate_sgrna(self, sequence):
        """验证sgRNA是否符合设计原则"""
        gc_content = (sequence.count('G') + sequence.count('C')) / len(sequence) * 100
        if gc_content > 40:
            return False, f"GC含量过高 ({gc_content:.1f}%)"

        # 检查连续相同碱基
        max_consecutive = 1
        current = 1
        for i in range(1, len(sequence)):
            if sequence[i] == sequence[i - 1]:
                current += 1
                if current > max_consecutive:
                    max_consecutive = current
            else:
                current = 1
        if max_consecutive >= 4:
            return False, f"存在连续{max_consecutive}个相同碱基"

        return True, "验证通过"

    def get_candidate_sgrnas(self, target_sequence):
        """生成候补sgRNA（示例逻辑，需根据实际需求调整）"""
        candidates = []
        # 简单示例：取前20个碱基作为候选
        if len(target_sequence) >= 20:
            candidates.append(target_sequence[:20])
            candidates.append(target_sequence[1:21])
        return candidates

    def handle_submit(self):
        """提交按钮事件处理"""
        # 获取组件值
        sgrna_seq = self.sgrna_seq_edit.text().upper()
        target_seq = self.target_seq_edit.text().upper()
        editor_type = self.editor_combo.currentText()
        window_start = self.window_start_edit.text()
        window_end = self.window_end_edit.text()

        # 校验必填项
        if not all([sgrna_seq, target_seq, window_start, window_end]):
            QMessageBox.warning(self, "参数缺失", "请填写所有必填项")
            return

        # 校验数值型参数
        try:
            window_start = int(window_start)
            window_end = int(window_end)
            if window_start >= window_end or window_start < 0 or window_end < 0:
                raise ValueError
        except ValueError:
            QMessageBox.warning(self, "参数错误", "Window范围必须为有效整数且start < end")
            return

        # 校验sgRNA设计原则
        valid, msg = self.validate_sgrna(sgrna_seq)
        if not valid:
            QMessageBox.warning(self, "sgRNA验证失败", msg)
            return

        # 生成候补sgRNA
        candidate_sgrnas = self.get_candidate_sgrnas(target_seq)
        if not candidate_sgrnas:
            QMessageBox.warning(self, "无候选序列", "目标序列过短，无法生成候选sgRNA")
            return

        # 调用模型预测
        results = []
        for idx, sgrna in enumerate(candidate_sgrnas, 1):
            try:
                pred_rate = predict_edit_rate(
                    self.loaded_model,
                    sgrna,
                    target_seq,
                    editor_type,
                    window_start,
                    window_end
                ).item()
                results.append({
                    "#": idx,
                    "Loc": f"Pos-{idx}",
                    "score": f"{pred_rate * 100:.2f}%",
                    "Recommended BE": editor_type
                })
            except Exception as e:
                QMessageBox.warning(self, "预测失败", f"处理第{idx}条序列时出错: {str(e)}")
                return

        # 显示结果到表格
        self.display_results(results)

    def display_results(self, results):
        """更新结果表格"""
        self.result_table.setRowCount(len(results))
        self.result_table.setColumnCount(4)
        self.result_table.setHorizontalHeaderLabels(["#", "Loc", "score", "Recommended BE"])

        for row_idx, row_data in enumerate(results):
            for col_idx, (key, value) in enumerate(row_data.items()):
                item = QTableWidgetItem(str(value))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.result_table.setItem(row_idx, col_idx, item)
        self.result_table.resizeColumnsToContents()  # 自动调整列宽


# 主程序入口
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SgRNADesignPage()
    window.setWindowTitle("sgRNA设计工具")
    window.resize(1000, 600)
    window.show()
    sys.exit(app.exec_())