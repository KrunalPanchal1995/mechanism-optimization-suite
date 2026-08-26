"""
mso_studio.gui.widgets.digitizer_view
========================================
The interactive click-to-calibrate/digitize canvas, built on
``QGraphicsView``/``QGraphicsScene`` -- the genuinely GUI-specific part of
the digitizer (see ``core/digitizer.py`` for the pure coordinate-mapping and
export logic this widget calls into).
"""
from __future__ import annotations

from typing import Optional

from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QPen, QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QGraphicsEllipseItem, QGraphicsPixmapItem, QGraphicsScene, QGraphicsSimpleTextItem,
    QGraphicsView,
)

_MODE_COLORS = {
    "x_cal": QColor("#ef4444"),
    "y_cal": QColor("#4f8ef7"),
    "data": QColor("#22c55e"),
    "err_bar": QColor("#f59e0b"),
}


class DigitizerView(QGraphicsView):
    """Shows the loaded plot image; emits ``pointClicked(px, py)`` in
    *original image pixel* coordinates (not scene/view coordinates) on every
    left click, and supports ctrl+wheel zoom."""

    pointClicked = Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHints(self.renderHints())
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._marker_items: list = []

    def load_image(self, path: str) -> None:
        pixmap = QPixmap(path)
        self._scene.clear()
        self._marker_items.clear()
        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def clear_markers(self) -> None:
        for item in self._marker_items:
            self._scene.removeItem(item)
        self._marker_items.clear()

    def pop_last_marker(self) -> None:
        if self._marker_items:
            # Each point draws 2 items (ellipse + label); drop both.
            for _ in range(2):
                if self._marker_items:
                    self._scene.removeItem(self._marker_items.pop())

    def draw_marker(self, px: float, py: float, mode: str, label: str) -> None:
        color = _MODE_COLORS.get(mode, QColor("white"))
        r = 5
        ellipse = QGraphicsEllipseItem(px - r, py - r, 2 * r, 2 * r)
        ellipse.setPen(QPen(color, 2))
        ellipse.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self._scene.addItem(ellipse)
        text = QGraphicsSimpleTextItem(label)
        text.setBrush(QBrush(color))
        text.setPos(px + 8, py - 14)
        self._scene.addItem(text)
        self._marker_items.extend([ellipse, text])

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt override
        if event.button() == Qt.MouseButton.LeftButton and self._pixmap_item is not None:
            scene_pos: QPointF = self.mapToScene(event.pos())
            if self._pixmap_item.boundingRect().contains(scene_pos):
                self.pointClicked.emit(scene_pos.x(), scene_pos.y())
                return  # don't also start a scroll-hand drag on a data click
        super().mousePressEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt override
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.scale(factor, factor)
        else:
            super().wheelEvent(event)
