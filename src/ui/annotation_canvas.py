"""
标注画布组件 — 封装 QGraphicsView 及所有标注渲染/交互逻辑
"""

from typing import List

from PySide6.QtWidgets import (
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsRectItem, QGraphicsTextItem, QGraphicsLineItem,
    QGraphicsEllipseItem, QGraphicsPolygonItem, QMenu
)
from PySide6.QtCore import Qt, QPointF, QRectF, Signal
from PySide6.QtGui import (
    QPainter, QPen, QBrush, QColor, QPixmap, QAction,
    QIcon, QMouseEvent, QWheelEvent, QFont, QPolygonF, QKeySequence
)

from src.core.annotation import Annotation, PolygonAnnotation
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


class AnnotationPolygonItem(QGraphicsPolygonItem):
    """多边形标注图形项：选中、顶点拖动、整体平移、边插入顶点、顶点删除"""

    def __init__(self, points, annotation, color, canvas, annotation_index=-1, parent=None):
        super().__init__(QPolygonF([QPointF(x, y) for x, y in points]), parent)
        self.annotation = annotation
        self.annotation_index = annotation_index
        self.color = color
        self.canvas = canvas  # 所属 AnnotationCanvas
        self.logger = get_logger_simple(__name__)

        # 编辑状态
        self.is_selected = False
        self.is_dragging = False          # 整体平移
        self.is_dragging_vertex = False   # 顶点拖动
        self.drag_vertex_index = -1
        self.drag_start_pos = QPointF()
        self.original_points: List[QPointF] = []
        self.handle_size = 8

        self.setFlag(QGraphicsPolygonItem.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)

        self.context_menu = None
        self.class_menu = None

        self.update_appearance()

    # ---- 命中检测 ----

    def _hit_vertex(self, pos: QPointF) -> int:
        """返回命中的顶点索引；未命中返回 -1"""
        for i, point in enumerate(self.polygon()):
            if (pos - point).manhattanLength() < self.handle_size * 2:
                return i
        return -1

    def _hit_edge(self, pos: QPointF) -> int:
        """返回最近边的起点顶点索引；未命中返回 -1"""
        points = [QPointF(p) for p in self.polygon()]
        count = len(points)
        if count < 3:
            return -1
        for i in range(count):
            a = points[i]
            b = points[(i + 1) % count]
            if self._point_segment_distance(pos, a, b) < self.handle_size:
                return i
        return -1

    @staticmethod
    def _point_segment_distance(point: QPointF, a: QPointF, b: QPointF) -> float:
        """点到线段的距离"""
        ab_x = b.x() - a.x()
        ab_y = b.y() - a.y()
        length_sq = ab_x * ab_x + ab_y * ab_y
        if length_sq <= 0:
            return ((point.x() - a.x()) ** 2 + (point.y() - a.y()) ** 2) ** 0.5
        t = ((point.x() - a.x()) * ab_x + (point.y() - a.y()) * ab_y) / length_sq
        t = max(0.0, min(1.0, t))
        proj_x = a.x() + t * ab_x
        proj_y = a.y() + t * ab_y
        return ((point.x() - proj_x) ** 2 + (point.y() - proj_y) ** 2) ** 0.5

    # ---- 事件 ----

    def hoverMoveEvent(self, event):
        if self.is_selected:
            if self._hit_vertex(event.pos()) >= 0:
                self.setCursor(Qt.SizeAllCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.canvas._deselect_all_except(self)
            if not self.is_selected:
                self.set_selected(True)

            self.canvas._begin_edit_capture()
            vertex_index = self._hit_vertex(event.pos())
            if vertex_index >= 0:
                self.is_dragging_vertex = True
                self.drag_vertex_index = vertex_index
            else:
                self.is_dragging = True
            self.drag_start_pos = event.pos()
            self.original_points = [QPointF(p) for p in self.polygon()]
            event.accept()

        elif event.button() == Qt.RightButton:
            self.show_context_menu(event.screenPos(), event.pos())
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.is_dragging_vertex:
            delta = event.pos() - self.drag_start_pos
            points = [QPointF(p) for p in self.original_points]
            points[self.drag_vertex_index] = (
                self.original_points[self.drag_vertex_index] + delta
            )
            self._apply_points(points)
            event.accept()

        elif self.is_dragging:
            delta = event.pos() - self.drag_start_pos
            points = [point + delta for point in self.original_points]
            self._apply_points(points)
            event.accept()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and (self.is_dragging or self.is_dragging_vertex):
            self.is_dragging = False
            self.is_dragging_vertex = False
            self.drag_vertex_index = -1
            self._sync_text_item()
            self.canvas._end_edit_capture()
            event.accept()
        else:
            super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        """双击边线：在边上插入一个顶点（可撤销）"""
        if event.button() == Qt.LeftButton:
            edge_index = self._hit_edge(event.pos())
            if edge_index >= 0:
                self.canvas._begin_edit_capture()
                points = [QPointF(p) for p in self.polygon()]
                points.insert(edge_index + 1, event.pos())
                self._apply_points(points)
                self.canvas._end_edit_capture()
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    # ---- 编辑操作 ----

    def _apply_points(self, points: List[QPointF]):
        """应用新的顶点列表：同步图形项、annotation 与文本标签"""
        self.setPolygon(QPolygonF(points))
        self.annotation.points = [(point.x(), point.y()) for point in points]
        self._sync_text_item()

    def _sync_text_item(self):
        """文本标签锚定在外接矩形左上角"""
        text_item = getattr(self, 'associated_text_item', None)
        if text_item is None:
            return
        bounding = self.polygon().boundingRect()
        text_item.setPos(bounding.x(), bounding.y() - 20)

    def delete_vertex(self, vertex_index: int):
        """删除指定顶点（剩余顶点不足 3 个时拒绝，可撤销）"""
        points = [QPointF(p) for p in self.polygon()]
        if len(points) <= 3:
            from src.utils.i18n import tr
            self.canvas.status_message.emit(tr("polygon_min_points_hint"))
            return
        if not (0 <= vertex_index < len(points)):
            return

        self.canvas._begin_edit_capture()
        points.pop(vertex_index)
        self._apply_points(points)
        self.canvas._end_edit_capture()

    def set_selected(self, selected: bool):
        self.is_selected = selected
        self.update_appearance()

    def update_appearance(self):
        if self.is_selected:
            self.setPen(QPen(QColor(255, 255, 0), 3))
            self.setBrush(QBrush(QColor(255, 255, 0, 40)))
        else:
            self.setPen(QPen(self.color, 2))
            fill = QColor(self.color)
            fill.setAlpha(60)
            self.setBrush(QBrush(fill))

    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget)

        if self.is_selected:
            handle_size = self.handle_size
            painter.setBrush(QBrush(QColor(255, 255, 255)))
            painter.setPen(QPen(QColor(0, 0, 0), 1))

            for point in self.polygon():
                painter.drawRect(
                    point.x() - handle_size / 2,
                    point.y() - handle_size / 2,
                    handle_size, handle_size
                )

    def show_context_menu(self, screen_pos, item_pos=None):
        """右键菜单：顶点上为删除顶点，其他位置为改类/删除标注"""
        from src.utils.i18n import tr

        vertex_index = self._hit_vertex(item_pos) if item_pos is not None else -1
        if vertex_index >= 0:
            menu = QMenu()
            delete_vertex_action = QAction(tr("delete_vertex"), menu)
            delete_vertex_action.triggered.connect(
                lambda checked=False, idx=vertex_index: self.delete_vertex(idx)
            )
            menu.addAction(delete_vertex_action)
            menu.exec(screen_pos)
            return

        if not self.context_menu:
            self.context_menu = QMenu()
            self.class_menu = QMenu(tr("modify_class"), self.context_menu)
            self.context_menu.addMenu(self.class_menu)

            delete_action = QAction(tr("delete_annotation"), self.context_menu)
            delete_action.triggered.connect(self.delete_annotation)
            self.context_menu.addAction(delete_action)

        self.class_menu.clear()
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
        """修改类别（可撤销）"""
        self.canvas._begin_edit_capture()
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

        self.canvas._end_edit_capture()

    def delete_annotation(self):
        self.canvas.request_delete_annotation(self.annotation, self.annotation_index)


class AnnotationCanvas(QGraphicsView):
    """标注画布 — 管理图像显示、标注绘制、鼠标交互"""

    annotation_created = Signal(object)   # Annotation
    annotation_changed = Signal()          # 标注被修改（拖拽/改类）
    annotation_deleted = Signal(object, int)  # Annotation, index
    status_message = Signal(str)
    scale_changed = Signal(float)
    edit_gesture_started = Signal()        # 可撤销编辑手势开始（多边形拖动等）
    edit_gesture_finished = Signal()       # 可撤销编辑手势结束

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

        # 任务模式：detect（矩形绘制）/ segment（多边形绘制）
        self.task_mode = "detect"
        # 多边形绘制状态
        self._poly_points: List[QPointF] = []
        self._temp_poly_item = None
        self._temp_vertex_items: List[QGraphicsEllipseItem] = []
        # 可撤销编辑手势状态
        self._edit_capture_active = False

        # Esc/Backspace/Enter 需要键盘焦点
        self.setFocusPolicy(Qt.StrongFocus)

    # ---- 公共接口 ----

    def set_class_manager(self, cm):
        self.class_manager = cm

    def set_selected_class_id(self, class_id):
        self.selected_class_id = class_id

    def set_task_mode(self, mode: str):
        """设置任务模式：detect（矩形绘制）/ segment（多边形绘制）"""
        if mode not in ("detect", "segment"):
            return
        if mode == self.task_mode:
            return
        # 切换模式时放弃未完成的多边形绘制，避免模式串扰
        self._cancel_polygon_drawing()
        self.task_mode = mode

    def is_polygon_drawing(self) -> bool:
        """是否正处于多边形绘制中（存在待闭合顶点）"""
        return len(self._poly_points) > 0

    def polygon_undo_last_point(self):
        """移除多边形绘制中的最后一个顶点；全部移除则取消绘制"""
        if not self._poly_points:
            return
        self._poly_points.pop()
        if not self._poly_points:
            self._cancel_polygon_drawing()
        else:
            self._update_polygon_preview(None)

    # ---- 可撤销编辑手势（由图形项调用）----

    def _begin_edit_capture(self):
        if not self._edit_capture_active:
            self._edit_capture_active = True
            self.edit_gesture_started.emit()

    def _end_edit_capture(self):
        if self._edit_capture_active:
            self._edit_capture_active = False
            self.edit_gesture_finished.emit()

    def display_image(self, pixmap: QPixmap):
        """清除场景并显示背景图片"""
        # 切图时丢弃未完成的绘制/编辑手势，避免状态串到新图
        self._cancel_polygon_drawing()
        self._edit_capture_active = False
        self._is_drawing = False
        self._drawing_start = None
        self._drawing_end = None
        self._temp_rect_item = None

        self._scene.clear()
        self._crosshair_items = None

        self._image_item = QGraphicsPixmapItem(pixmap)
        self._scene.addItem(self._image_item)
        self.setSceneRect(self._image_item.boundingRect())

    def clear_image(self):
        """清空场景并把状态复位到"未加载图片"

        仅调用 _scene.clear() 不够：_image_item 会残留指向已从场景移除的
        图形项，导致 has_image 永远为真（关夹后再绘制标注就又崩一次）。
        所有"不再显示图片"的路径都必须走这里，而不是直接清场景。
        """
        self._cancel_polygon_drawing()
        self._edit_capture_active = False
        self._scene.clear()
        self._crosshair_items = None
        self._image_item = None
        self._is_drawing = False
        self._drawing_start = None
        self._drawing_end = None
        self._temp_rect_item = None

    @property
    def has_image(self) -> bool:
        """当前是否已加载并显示图片

        绘制类操作的唯一判据。不要再直接读 self._image_item——
        它会随 _scene.clear() 残留而说谎。
        """
        return self._image_item is not None

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

        if isinstance(annotation, PolygonAnnotation):
            item = AnnotationPolygonItem(
                annotation.points, annotation, color, self, index
            )
            bounding = item.polygon().boundingRect()
            label_x, label_y = bounding.x(), bounding.y() - 20
        else:
            item = AnnotationRectItem(
                annotation.x, annotation.y,
                annotation.width, annotation.height,
                annotation, color, self, index
            )
            label_x, label_y = annotation.x, annotation.y - 20
        item.is_annotation_item = True
        item.associated_text_item = None  # 稍后设置
        self._scene.addItem(item)

        text_item = QGraphicsTextItem(class_name)
        text_item.setDefaultTextColor(color)
        text_item.setPos(label_x, label_y)
        text_item.is_annotation_item = True
        # 文本标签纯装饰，不参与鼠标事件：否则标签覆盖在标注项上方，
        # 点击/拖动角落（多边形常与标签重叠）会被标签吞掉，无法选中标注
        text_item.setAcceptedMouseButtons(Qt.NoButton)
        text_item.associated_rect_item = item
        item.associated_text_item = text_item
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

    # ---- 多边形绘制（segment 模式）----

    def _class_color(self) -> QColor:
        """当前选中类别的颜色（无类别时使用默认色）"""
        cm = self.class_manager
        if cm:
            class_info = cm.get_class(self.selected_class_id)
            if class_info:
                return QColor(*class_info["color"])
        return QColor(0, 170, 255)

    def _update_polygon_preview(self, cursor_pos):
        """刷新绘制中的多边形预览（已确定顶点 + 跟随鼠标的橡皮筋）"""
        color = self._class_color()

        if self._temp_poly_item is None:
            self._temp_poly_item = QGraphicsPolygonItem()
            self._scene.addItem(self._temp_poly_item)
        self._temp_poly_item.setPen(QPen(color, 2, Qt.DashLine))
        fill = QColor(color)
        fill.setAlpha(40)
        self._temp_poly_item.setBrush(QBrush(fill))

        preview_points = list(self._poly_points)
        if cursor_pos is not None and (
            not preview_points
            or (cursor_pos - preview_points[-1]).manhattanLength() > 0.1
        ):
            preview_points.append(cursor_pos)
        self._temp_poly_item.setPolygon(QPolygonF(preview_points))

        # 顶点标记
        for item in self._temp_vertex_items:
            if item.scene() is not None:
                self._scene.removeItem(item)
        self._temp_vertex_items = []
        for point in self._poly_points:
            dot = QGraphicsEllipseItem(point.x() - 3, point.y() - 3, 6, 6)
            dot.setBrush(QBrush(QColor(255, 255, 255)))
            dot.setPen(QPen(color, 1))
            self._scene.addItem(dot)
            self._temp_vertex_items.append(dot)

    def _cancel_polygon_drawing(self):
        """放弃当前多边形绘制并清理预览图形项"""
        self._poly_points = []
        items = list(self._temp_vertex_items)
        if self._temp_poly_item is not None:
            items.append(self._temp_poly_item)
        for item in items:
            if item.scene() is not None:
                self._scene.removeItem(item)
        self._temp_poly_item = None
        self._temp_vertex_items = []

    def _finish_polygon(self):
        """闭合当前多边形并发出创建信号（不足 3 个顶点时提示）"""
        if len(self._poly_points) < 3:
            from src.utils.i18n import tr
            self.status_message.emit(tr("polygon_min_points_hint"))
            return
        if not self.has_image:
            self._cancel_polygon_drawing()
            return

        annotation = PolygonAnnotation(
            points=[(point.x(), point.y()) for point in self._poly_points],
            class_id=self.selected_class_id,
        )
        self._cancel_polygon_drawing()
        self.annotation_created.emit(annotation)

    def _handle_segment_click(self, event: QMouseEvent, scene_pos: QPointF):
        """segment 模式左键：加点 / 点击起点闭合 / 委托选中已有标注"""
        if self._poly_points:
            first_point = self._poly_points[0]
            if len(self._poly_points) >= 3 and self._is_near_point(
                event.pos(), first_point, 10
            ):
                self._finish_polygon()
            else:
                self._poly_points.append(scene_pos)
                self._update_polygon_preview(None)
            event.accept()
            return

        clicked_items = self.items(event.pos())
        for item in clicked_items:
            if hasattr(item, 'annotation') and isinstance(
                item, (QGraphicsRectItem, QGraphicsPolygonItem)
            ):
                super().mousePressEvent(event)
                return

        self._poly_points = [scene_pos]
        self._update_polygon_preview(None)
        event.accept()

    def _is_near_point(self, view_pos, scene_point: QPointF, threshold: float) -> bool:
        """视图坐标与场景点（换算到视图坐标）的距离是否小于阈值"""
        point_view_pos = self.mapFromScene(scene_point)
        delta_x = view_pos.x() - point_view_pos.x()
        delta_y = view_pos.y() - point_view_pos.y()
        return (delta_x * delta_x + delta_y * delta_y) ** 0.5 <= threshold

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
            for item in self._crosshair_items:
                # 装饰性十字准线不参与鼠标事件：否则光标下的红点/准线会
                # 拦在标注项上方吞掉点击与拖动
                item.setAcceptedMouseButtons(Qt.NoButton)
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

        if (event.button() == Qt.RightButton
                and self.task_mode == "segment"
                and self._poly_points):
            # 绘制中的右键：移除最后一个顶点（与 Backspace 一致）
            self.polygon_undo_last_point()
            event.accept()
            return

        if event.button() == Qt.LeftButton:
            # 未加载图片时不进入绘制：空场景的 mapToScene 依然返回合法坐标，
            # 放行会让用户拖出矩形，最终在 main_window 侧因 current_image_path
            # 为 None 而崩在 Path(None)。手势干脆不开始，同时给状态栏一句提示，
            # 否则用户只看到"点了没反应"。
            if not self.has_image:
                from src.utils.i18n import tr
                self.status_message.emit(tr("canvas_no_image_hint"))
                event.accept()
                return

            scene_pos = self.mapToScene(event.pos())

            if self.task_mode == "segment":
                self._handle_segment_click(event, scene_pos)
                return

            clicked_items = self.items(event.pos())
            for item in clicked_items:
                if hasattr(item, 'annotation') and isinstance(
                    item, (QGraphicsRectItem, QGraphicsPolygonItem)
                ):
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
        elif self.task_mode == "segment" and self._poly_points:
            self._update_polygon_preview(scene_pos)
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

                if (rect.width() > 10 and rect.height() > 10
                        and self.has_image):
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

    def mouseDoubleClickEvent(self, event: QMouseEvent):
        if (self.task_mode == "segment"
                and event.button() == Qt.LeftButton
                and self._poly_points
                and self.has_image):
            # 双击闭合：若双击位置与最后顶点明显不同，则其作为最后一个顶点
            scene_pos = self.mapToScene(event.pos())
            last_view_pos = self.mapFromScene(self._poly_points[-1])
            delta_x = event.pos().x() - last_view_pos.x()
            delta_y = event.pos().y() - last_view_pos.y()
            if (delta_x * delta_x + delta_y * delta_y) ** 0.5 > 4:
                self._poly_points.append(scene_pos)
            self._finish_polygon()
            event.accept()
            return

        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event):
        if self.task_mode == "segment" and self._poly_points:
            if event.key() == Qt.Key_Escape:
                self._cancel_polygon_drawing()
                event.accept()
                return
            if event.key() == Qt.Key_Backspace:
                self.polygon_undo_last_point()
                event.accept()
                return
            if event.key() in (Qt.Key_Return, Qt.Key_Enter):
                self._finish_polygon()
                event.accept()
                return
            # 撤销 QAction 被禁用时（撤销栈为空），Ctrl+Z 快捷键不会触发窗口
            # undo()，按键直接落到画布：绘制中同样解释为回退顶点，与窗口层
            # 拦截保持一致（与 Esc/Backspace 走同一条投递路径）。
            if event.matches(QKeySequence.StandardKey.Undo):
                self.polygon_undo_last_point()
                event.accept()
                return
        super().keyPressEvent(event)

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
