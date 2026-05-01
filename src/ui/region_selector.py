"""Window highlighter and region selector dialogs for screen capture."""

from typing import Optional

from PySide6.QtCore import Qt, QRect, QSize, QPoint
from PySide6.QtGui import QGuiApplication
from PySide6.QtWidgets import QDialog, QRubberBand


class WindowHighlighter(QDialog):
    """Transparent overlay that draws a colored frame around a target window."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.setStyleSheet("background-color: transparent;")

        self._border_color = Qt.red
        self._border_width = 4

    def set_target_rect(self, rect: QRect):
        """Move and resize the overlay to surround the given rectangle."""
        expanded = rect.adjusted(-self._border_width, -self._border_width,
                                 self._border_width, self._border_width)
        self.setGeometry(expanded)

    def paintEvent(self, event):
        """Draw a colored border around the overlay."""
        from PySide6.QtGui import QPainter, QPen
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        pen = QPen(self._border_color, self._border_width)
        painter.setPen(pen)
        half = self._border_width // 2
        rect = self.rect().adjusted(half, half, -half, -half)
        painter.drawRect(rect)


class RegionSelector(QDialog):
    """Fullscreen region selector on virtual desktop."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setModal(True)
        flags = Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Window
        self.setWindowFlags(flags)
        self.setCursor(Qt.CrossCursor)
        self.setAttribute(Qt.WA_DeleteOnClose, False)
        self.setGeometry(self._virtual_geometry())
        self.setFocusPolicy(Qt.StrongFocus)

        self.setStyleSheet("background-color: #1a1a1a;")
        self.setWindowOpacity(0.25)

        self._origin_global = QPoint()
        self._rubber_band = QRubberBand(QRubberBand.Rectangle, self)
        self._selected_rect: Optional[QRect] = None
        self._selection_started = False

    @staticmethod
    def _virtual_geometry() -> QRect:
        screens = QGuiApplication.screens()
        if not screens:
            return QRect(0, 0, 1920, 1080)
        rect = screens[0].geometry()
        for screen in screens[1:]:
            rect = rect.united(screen.geometry())
        return rect

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.reject()
            return
        super().keyPressEvent(event)

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton:
            return super().mousePressEvent(event)

        self._selection_started = True
        self._origin_global = event.globalPosition().toPoint()
        origin_local = self.mapFromGlobal(self._origin_global)

        self._rubber_band.setGeometry(QRect(origin_local, QSize()))
        self._rubber_band.show()

    def mouseMoveEvent(self, event):
        if not self._rubber_band.isVisible():
            return super().mouseMoveEvent(event)

        current_global = event.globalPosition().toPoint()
        current_local = self.mapFromGlobal(current_global)
        origin_local = self.mapFromGlobal(self._origin_global)
        rect = QRect(origin_local, current_local).normalized()
        self._rubber_band.setGeometry(rect)
        event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() != Qt.LeftButton:
            return super().mouseReleaseEvent(event)

        if not self._rubber_band.isVisible():
            return super().mouseReleaseEvent(event)

        current_global = event.globalPosition().toPoint()
        self._selected_rect = QRect(self._origin_global, current_global).normalized()

        if self._selected_rect.width() < 5 or self._selected_rect.height() < 5:
            self._rubber_band.hide()
            return

        self._rubber_band.hide()
        self.accept()
        event.accept()

    def closeEvent(self, event):
        if self._selected_rect is None:
            self.reject()
        super().closeEvent(event)
