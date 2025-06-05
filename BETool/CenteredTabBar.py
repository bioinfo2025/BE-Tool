from PyQt5.QtWidgets import QTabBar, QStyle, QStyleOptionTab
from PyQt5.QtCore import Qt, QRect, QSize, QRectF, QPointF, QEasingCurve
from PyQt5.QtGui import (
    QPainter, QPen, QColor, QBrush, QPalette, QIcon, QFontMetrics,
    QPainterPath, QLinearGradient, QRadialGradient
)


class CenteredTabBar(QTabBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDrawBase(False)  # 不绘制TabBar底部线条
        self.setExpanding(False)  # 禁用自动扩展

        # 启用鼠标跟踪，以支持悬停效果
        self.setMouseTracking(True)
        self.current_hover = -1  # 当前悬停的标签索引

    def tabSizeHint(self, index):
        # 自定义标签大小，减小高度并增加宽度
        base_size = super().tabSizeHint(index)
        return QSize(base_size.width() + 20, base_size.height() - 8)

    def mouseMoveEvent(self, event):
        # 跟踪鼠标位置，更新悬停状态
        for i in range(self.count()):
            if self.tabRect(i).contains(event.pos()):
                self.current_hover = i
                self.update()
                return
        self.current_hover = -1
        self.update()

    def leaveEvent(self, event):
        # 鼠标离开时重置悬停状态
        self.current_hover = -1
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setRenderHint(QPainter.TextAntialiasing)

        for i in range(self.count()):
            # 获取标签选项
            option = QStyleOptionTab()
            self.initStyleOption(option, i)

            # 计算标签矩形
            tab_rect = self.tabRect(i)

            # 绘制标签背景（包括边框和填充）
            self._drawTabBackground(painter, option, tab_rect, i)

            # 绘制标签内容（图标和文字）
            self._drawTabContent(painter, option, tab_rect)

        painter.end()

    def _drawTabBackground(self, painter, option, rect, index):
        # 保存当前画家状态
        painter.save()

        # 判断标签状态（选中、悬停、普通）
        is_selected = option.state & QStyle.State_Selected
        is_hovered = index == self.current_hover

        # 设置背景渐变
        if is_selected:
            # 选中标签的渐变背景
            gradient = QLinearGradient(rect.topLeft(), rect.bottomLeft())
            gradient.setColorAt(0, QColor(240, 240, 240))
            gradient.setColorAt(1, QColor(220, 220, 220))
            border_color = QColor(180, 180, 180)
        elif is_hovered:
            # 悬停标签的渐变背景
            gradient = QLinearGradient(rect.topLeft(), rect.bottomLeft())
            gradient.setColorAt(0, QColor(250, 250, 250))
            gradient.setColorAt(1, QColor(230, 230, 230))
            border_color = QColor(200, 200, 200)
        else:
            # 普通标签的渐变背景
            gradient = QLinearGradient(rect.topLeft(), rect.bottomLeft())
            gradient.setColorAt(0, QColor(245, 245, 245))
            gradient.setColorAt(1, QColor(235, 235, 235))
            border_color = QColor(210, 210, 210)

        # 绘制背景
        painter.fillRect(rect, QBrush(gradient))

        # 创建带圆角的路径
        path = QPainterPath()
        radius = 6  # 增加圆角弧度，使外观更柔和
        rect_f = QRectF(rect.adjusted(1, 1, -1, -1))  # 微调矩形，使边框更美观

        # 绘制圆角背景
        path.addRoundedRect(rect_f, radius, radius)
        painter.fillPath(path, QBrush(gradient))

        # 设置边框线
        pen = QPen(border_color, 1)
        painter.setPen(pen)

        # 绘制圆角边框
        painter.drawPath(path)

        # 为选中的标签添加底部高亮线
        if is_selected:
            highlight_pen = QPen(QColor(74, 134, 232), 2)  # 蓝色高亮线
            painter.setPen(highlight_pen)
            highlight_rect = QRectF(
                rect_f.left() + 2,
                rect_f.bottom() - 1,
                rect_f.width() - 4,
                1
            )
            painter.drawLine(highlight_rect.topLeft(), highlight_rect.topRight())

        # 恢复画家状态
        painter.restore()

    def _drawTabContent(self, painter, option, rect):
        # 保存当前画家状态
        painter.save()

        # 使用当前画家的字体
        font_metrics = painter.fontMetrics()

        # 获取图标和文字
        icon = option.icon
        text = option.text

        # 计算图标和文字的位置
        icon_size = option.iconSize
        text_rect = QRect(rect)

        # 如果有图标，计算图标位置并绘制
        if not icon.isNull():
            # 图标垂直居中
            icon_y = rect.y() + (rect.height() - icon_size.height()) // 2

            # 如果有文字，图标放在左侧，留间距
            if text:
                # 计算图标和文字的总宽度
                total_width = icon_size.width() + font_metrics.width(text) + 10  # 增加间距
                icon_x = rect.x() + (rect.width() - total_width) // 2

                # 调整文本区域，为图标留出空间
                text_rect.setLeft(icon_x + icon_size.width() + 6)
            else:
                # 没有文字时，图标居中
                icon_x = rect.x() + (rect.width() - icon_size.width()) // 2

            # 绘制图标
            icon_rect = QRect(icon_x, icon_y, icon_size.width(), icon_size.height())
            icon_mode = QIcon.Selected if option.state & QStyle.State_Selected else QIcon.Normal
            icon_state = QIcon.On if option.state & QStyle.State_Enabled else QIcon.Off
            icon.paint(painter, icon_rect, Qt.AlignCenter, icon_mode, icon_state)

        # 设置文字颜色
        if option.state & QStyle.State_Selected:
            painter.setPen(QColor(50, 50, 50))  # 选中标签文字稍深
        else:
            painter.setPen(QColor(80, 80, 80))  # 未选中标签文字稍浅

        # 绘制文字（垂直和水平居中）
        flags = Qt.AlignCenter | Qt.TextShowMnemonic | Qt.TextSingleLine
        painter.drawText(text_rect, flags, text)

        # 恢复画家状态
        painter.restore()