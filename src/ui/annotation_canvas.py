"""
标注画布组件 — 封装 QGraphicsView 及所有标注渲染/交互逻辑
"""

from typing import List, Optional

from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsRectItem, QGraphicsTextItem, QGraphicsLineItem,
    QGraphicsEllipseItem, QMenu, QGraphicsItem
)
from PySide6.QtCore import Qt, QPointF, QRectF, Signal, QEvent
from PySide6.QtGui import (
    QPainter, QPen, QBrush, QColor, QPixmap, QAction,
    QIcon, QMouseEvent, QWheelEvent, QCursor, QFont
)

from src.core.annotation import Annotation
from src.utils.logger import get_logger_simple


class AnnotationRectItem(QGraphicsRectItem):
    """标注框图形项，支持选中状态和拖拽编辑"""

    def __init__(self, x, y, width, height, annotation, color, canvas, annotation_index=-1, parent=None):
        super().__init__(x, y, width, height, parent)
        self.annotation = annotation
        self.annotation_index = annotation_index
        self.color = color
        self.canvas = canvas  # 所属 AnnotationCanvas
        self.logger = get_logger_simple(__name__)

        # 拖拽状态
        self.is_selected = False
        self.is_dragging = False
        self.is_resizing = False
        self.drag_start_pos = QPointF()
        self.original_rect = QRectF()
        self.resize_handle = None

        self.handle_size = 8

        self.setFlag(QGraphicsRectItem.ItemIsMovable, True)
        self.setFlag(QGraphicsRectItem.ItemIsSelectable, True)
        self.setFlag(QGraphicsRectItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)

        self.context_menu = None

        self.update_appearance()

    def hoverMoveEvent(self, event):
        if self.is_selected:
            pos = event.pos()
            rect = self.rect()

            handle_positions = {
                'top-left': QPointF(rect.left(), rect.top()),
                'top-right': QPointF(rect.right(), rect.top()),
                'bottom-left': QPointF(rect.left(), rect.bottom()),
                'bottom-right': QPointF(rect.right(), rect.bottom())
            }

            for handle_name, handle_pos in handle_positions.items():
                if (pos - handle_pos).manhattanLength() < self.handle_size * 2:
                    if handle_name in ['top-left', 'bottom-right']:
                        self.setCursor(Qt.SizeFDiagCursor)
                    elif handle_name in ['top-right', 'bottom-left']:
                        self.setCursor(Qt.SizeBDiagCursor)
                    self.resize_handle = handle_name
                    return

            if (abs(pos.x() - rect.left()) < 5 or abs(pos.x() - rect.right()) < 5 or
                abs(pos.y() - rect.top()) < 5 or abs(pos.y() - rect.bottom()) < 5):
                self.setCursor(Qt.SizeAllCursor)
                self.resize_handle = 'edge'
            else:
                self.setCursor(Qt.ArrowCursor)
                self.resize_handle = None
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.canvas._deselect_all_except(self)

            if self.resize_handle:
                self.is_resizing = True
                self.drag_start_pos = event.pos()
                self.original_rect = self.rect()
            else:
                self.is_dragging = True
                self.drag_start_pos = event.pos()
                self.original_rect = self.rect()

            if not self.is_selected:
                self.set_selected(True)
            event.accept()

        elif event.button() == Qt.RightButton:
            self.show_context_menu(event.screenPos())
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.is_dragging:
            delta = event.pos() - self.drag_start_pos
            new_rect = self.original_rect.translated(delta)
            self.setRect(new_rect)
            self.annotation.x = new_rect.x()
            self.annotation.y = new_rect.y()
            event.accept()

        elif self.is_resizing:
            delta = event.pos() - self.drag_start_pos
            new_rect = QRectF(self.original_rect)

            if self.resize_handle == 'top-left':
                new_rect.setTopLeft(self.original_rect.topLeft() + delta)
            elif self.resize_handle == 'top-right':
                new_rect.setTopRight(self.original_rect.topRight() + delta)
            elif self.resize_handle == 'bottom-left':
                new_rect.setBottomLeft(self.original_rect.bottomLeft() + delta)
            elif self.resize_handle == 'bottom-right':
                new_rect.setBottomRight(self.original_rect.bottomRight() + delta)
            elif self.resize_handle == 'edge':
                new_rect = self.original_rect.adjusted(-delta.x()/2, -delta.y()/2, delta.x()/2, delta.y()/2)

            if new_rect.width() > 10 and new_rect.height() > 10:
                self.setRect(new_rect)
                self.annotation.x = new_rect.x()
                self.annotation.y = new_rect.y()
                self.annotation.width = new_rect.width()
                self.annotation.height = new_rect.height()

            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.is_dragging or self.is_resizing:
                self.canvas.annotation_changed.emit()
                self.is_dragging = False
                self.is_resizing = False
                event.accept()
            else:
                super().mouseReleaseEvent(event)
        else:
            super().mouseReleaseEvent(event)

    def set_selected(self, selected: bool):
        self.is_selected = selected
        self.update_appearance()

    def update_appearance(self):
        if self.is_selected:
            self.setPen(QPen(QColor(255, 255, 0), 3))
            self.setBrush(QBrush(QColor(255, 255, 0, 30)))
            self.paint_handles()
        else:
            self.setPen(QPen(self.color, 2))
            self.setBrush(QBrush(self.color, Qt.Dense4Pattern))
            self.resize_handle = None

    def paint_handles(self):
        self.update()

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)

        if self.is_selected:
            rect = self.rect()
            handle_size = self.handle_size

            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.setPen(QPen(QColor(0, 0, 0), 1))

            corners = [
                rect.topLeft(), rect.topRight(),
                rect.bottomLeft(), rect.bottomRight()
            ]

            for corner in corners:
                painter.drawRect(
                    corner.x() - handle_size/2,
                    corner.y() - handle_size/2,
                    handle_size, handle_size
                )

    def show_context_menu(self, screen_pos):
        if not self.context_menu:
            self.context_menu = QMenu()
            from src.utils.i18n import tr

            self.class_menu = QMenu(tr("modify_class"), self.context_menu)
            self.context_menu.addMenu(self.class_menu)

            delete_action = QAction(tr("delete_annotation"), self.context_menu)
            delete_action.triggered.connect(self.delete_annotation)
            self.context_menu.addAction(delete_action)

        self.class_menu.clear()
        from src.utils.i18n import tr

        cm = self.canvas.class_manager
        if cm:
            classes = cm.get_classes()
            for class_id, class_info in sorted(classes.items()):
                class_name = class_info["name"]
                color = class_info["color"]

                action = QAction(class_name, self.class_menu)
                action.setData(class_id)
                action.triggered.connect(lambda checked, cid=class_id: self.change_class(cid))

                pixmap = QPixmap(16, 16)
                pixmap.fill(QColor(*color))
                action.setIcon(QIcon(pixmap))

                self.class_menu.addAction(action)

        self.context_menu.exec(screen_pos)

    def change_class(self, class_id):
        self.annotation.class_id = class_id

        cm = self.canvas.class_manager
        if cm:
            class_info = cm.get_class(class_id)
            if class_info:
                self.color = QColor(*class_info["color"])
                self.update_appearance()
                # 同步更新关联的文本标签
                text_item = getattr(self, 'associated_text_item', None)
                if text_item is not None:
                    text_item.setPlainText(class_info["name"])
                    text_item.setDefaultTextColor(self.color)

        self.canvas.annotation_changed.emit()

    def delete_annotation(self):
        self.canvas.request_delete_annotation(self.annotation, self.annotation_index)


class AnnotationCanvas(QGraphicsView):
    """标注画布 — 管理图像显示、标注绘制、鼠标交互"""

    annotation_created = Signal(object)   # Annotation
    annotation_changed = Signal()          # 标注被修改（拖拽/改类）
    annotation_deleted = Signal(object, int)  # Annotation, index
    status_message = Signal(str)
    scale_changed = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setRenderHint(QPainter.Antialiasing)
        self.setRenderHint(QPainter.SmoothPixmapTransform)

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        # 默认使用 NoDrag，左键绘制/选择，中键平移
        self.setDragMode(QGraphicsView.NoDrag)
        self.setCursor(Qt.CrossCursor)
        self._panning = False

        # 状态
        self.class_manager = None
        self.selected_class_id = 0
        self._image_item = None
        self._is_drawing = False
        self._drawing_start = None
        self._drawing_end = None
        self._temp_rect_item = None
        self._crosshair_items = None

    # ---- 公共接口 ----

    def set_class_manager(self, cm):
        self.class_manager = cm

    def set_selected_class_id(self, class_id):
        self.selected_class_id = class_id

    def display_image(self, pixmap: QPixmap):
        """清除场景并显示背景图片"""
        self._scene.clear()
        self._crosshair_items = None

        self._image_item = QGraphicsPixmapItem(pixmap)
        self._scene.addItem(self._image_item)
        self.setSceneRect(self._image_item.boundingRect())

    def draw_annotations(self, annotations: List[Annotation]):
        """清除并重新绘制所有标注"""
        for item in self._scene.items():
            if hasattr(item, 'is_annotation_item') and item.is_annotation_item:
                self._scene.removeItem(item)

        for i, annotation in enumerate(annotations):
            self._draw_one_annotation(annotation, i)

    def clear_annotation_items(self):
        """仅移除标注相关图形项（保留图片）"""
        items_to_remove = []
        for item in self._scene.items():
            if hasattr(item, 'is_annotation_item'):
                items_to_remove.append(item)
        for item in items_to_remove:
            self._scene.removeItem(item)

    def get_annotation_items(self) -> List[Annotation]:
        """从场景项中提取 Annotation 列表"""
        annotations = []
        for item in self._scene.items():
            if hasattr(item, 'annotation'):
                annotations.append(item.annotation)
        return annotations

    def get_selected_annotation(self):
        """返回第一个选中的 Annotation，无选中时返回 None"""
        for item in self._scene.items():
            if hasattr(item, 'annotation') and hasattr(item, 'is_selected') and item.is_selected:
                return item.annotation
        return None

    def get_selected_annotation_index(self) -> int:
        """返回第一个选中的标注在列表中的索引，无选中时返回 -1"""
        for item in self._scene.items():
            if hasattr(item, 'annotation_index') and hasattr(item, 'is_selected') and item.is_selected:
                return item.annotation_index
        return -1

    def fit_to_window(self):
        if self._image_item:
            self.fitInView(self._image_item, Qt.KeepAspectRatio)

    def zoom_in(self):
        self.scale(1.25, 1.25)

    def zoom_out(self):
        self.scale(0.8, 0.8)

    def reset_view(self):
        self.resetTransform()

    def get_scale_factor(self) -> float:
        return self.transform().m11()

    def get_image_size(self):
        """返回当前显示图片的 (width, height)，无图片时返回 None"""
        if self._image_item:
            pixmap = self._image_item.pixmap()
            if pixmap:
                return pixmap.width(), pixmap.height()
        return None

    def request_delete_annotation(self, annotation: Annotation, index: int = -1):
        """由 AnnotationRectItem 调用，触发删除流程"""
        self.annotation_deleted.emit(annotation, index)

    # ---- 内部方法 ----

    def _draw_one_annotation(self, annotation: Annotation, index: int = -1):
        cm = self.class_manager
        if cm:
            class_info = cm.get_class(annotation.class_id)
            if class_info:
                color = QColor(*class_info["color"])
                class_name = class_info["name"]
            else:
                color = QColor(128, 128, 128)
                from src.utils.i18n import tr
                class_name = tr("unknown_class").replace("{class_id}", str(annotation.class_id))
        else:
            color = QColor(128, 128, 128)
            class_name = str(annotation.class_id)

        rect_item = AnnotationRectItem(
            annotation.x, annotation.y,
            annotation.width, annotation.height,
            annotation, color, self, index
        )
        rect_item.is_annotation_item = True
        rect_item.associated_text_item = None  # 稍后设置
        self._scene.addItem(rect_item)

        text_item = QGraphicsTextItem(class_name)
        text_item.setDefaultTextColor(color)
        text_item.setPos(annotation.x, annotation.y - 20)
        text_item.is_annotation_item = True
        text_item.associated_rect_item = rect_item
        rect_item.associated_text_item = text_item
        font = QFont()
        font.setPointSize(10)
        text_item.setFont(font)
        self._scene.addItem(text_item)

    def _deselect_all_except(self, keep_item):
        for item in self._scene.items():
            if (hasattr(item, 'is_annotation_item') and
                item != keep_item and
                hasattr(item, 'is_selected') and
                item.is_selected):
                item.set_selected(False)

    def _draw_crosshair(self, scene_pos: QPointF):
        if not self._crosshair_items:
            scene_rect = self._scene.sceneRect()
            if scene_rect.isNull():
                return
            vline = QGraphicsLineItem()
            vline.setPen(QPen(QColor(255, 255, 0, 128), 1, Qt.DashLine))
            vline.is_crosshair = True
            self._scene.addItem(vline)
            hline = QGraphicsLineItem()
            hline.setPen(QPen(QColor(255, 255, 0, 128), 1, Qt.DashLine))
            hline.is_crosshair = True
            self._scene.addItem(hline)
            dot = QGraphicsEllipseItem(0, 0, 4, 4)
            dot.setBrush(QBrush(QColor(255, 0, 0, 200)))
            dot.setPen(QPen(Qt.NoPen))
            dot.is_crosshair = True
            self._scene.addItem(dot)
            self._crosshair_items = (vline, hline, dot)
        else:
            stale = any(item.scene() is None for item in self._crosshair_items)
            if stale:
                self._crosshair_items = None
                return
            for item in self._crosshair_items:
                item.setVisible(False)

        scene_rect = self._scene.sceneRect()
        if scene_rect.isNull():
            return

        vline, hline, dot = self._crosshair_items
        vline.setLine(scene_rect.x(), scene_pos.y(), scene_rect.x() + scene_rect.width(), scene_pos.y())
        hline.setLine(scene_pos.x(), scene_rect.y(), scene_pos.x(), scene_rect.y() + scene_rect.height())
        dot.setRect(scene_pos.x() - 2, scene_pos.y() - 2, 4, 4)
        for item in self._crosshair_items:
            item.setVisible(True)

    # ---- 事件重写 ----

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MiddleButton:
            # 中键平移（手动实现，ScrollHandDrag 只响应左键）
            self._panning = True
            self._pan_start_pos = event.pos()
            self.viewport().setCursor(Qt.ClosedHandCursor)
            event.accept()
            return

        if event.button() == Qt.LeftButton:
            scene_pos = self.mapToScene(event.pos())

            clicked_items = self.items(event.pos())
            for item in clicked_items:
                if hasattr(item, 'annotation') and isinstance(item, QGraphicsRectItem):
                    self._is_drawing = False
                    super().mousePressEvent(event)
                    return

            self._is_drawing = True
            self._drawing_start = scene_pos
            self._drawing_end = scene_pos

            self._temp_rect_item = QGraphicsRectItem()
            self._temp_rect_item.setPen(QPen(Qt.red, 2, Qt.DashLine))
            self._scene.addItem(self._temp_rect_item)

            event.accept()
            return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._panning:
            delta = event.pos() - self._pan_start_pos
            self._pan_start_pos = event.pos()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - delta.x()
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - delta.y()
            )
            event.accept()
            return

        scene_pos = self.mapToScene(event.pos())

        if self._is_drawing and self._temp_rect_item:
            self._drawing_end = scene_pos
            rect = QRectF(
                min(self._drawing_start.x(), scene_pos.x()),
                min(self._drawing_start.y(), scene_pos.y()),
                abs(self._drawing_start.x() - scene_pos.x()),
                abs(self._drawing_start.y() - scene_pos.y())
            )
            self._temp_rect_item.setRect(rect)
            event.accept()
        else:
            self._draw_crosshair(scene_pos)
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.MiddleButton and self._panning:
            self._panning = False
            self.setCursor(Qt.CrossCursor)
            self.viewport().setCursor(Qt.CrossCursor)
            event.accept()
            return

        if event.button() == Qt.LeftButton and self._is_drawing:
            self._is_drawing = False

            if self._temp_rect_item:
                self._scene.removeItem(self._temp_rect_item)
                self._temp_rect_item = None

            if self._drawing_start and self._drawing_end:
                rect = QRectF(
                    min(self._drawing_start.x(), self._drawing_end.x()),
                    min(self._drawing_start.y(), self._drawing_end.y()),
                    abs(self._drawing_start.x() - self._drawing_end.x()),
                    abs(self._drawing_start.y() - self._drawing_end.y())
                )

                if rect.width() > 10 and rect.height() > 10:
                    annotation = Annotation(
                        rect.x(), rect.y(),
                        rect.width(), rect.height(),
                        self.selected_class_id
                    )
                    self.annotation_created.emit(annotation)

            self._drawing_start = None
            self._drawing_end = None
            event.accept()
            return

        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        if event.modifiers() & Qt.ControlModifier:
            delta = event.angleDelta().y()
            if delta > 0:
                self.zoom_in()
            else:
                self.zoom_out()
            event.accept()
        else:
            super().wheelEvent(event)
