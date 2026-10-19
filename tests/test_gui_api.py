"""GUI module import/offscreen tests.

PySide6 widgets are instantiated offscreen (QT_QPA_PLATFORM=offscreen) so the
test works headless. The full Video pipeline (GPU RAFT over 82 frames) is NOT
run here; that belongs to integration tests.
"""
import os
import sys

import pytest

pyside = pytest.importorskip("PySide6")


def test_pyside6_version():
    from PySide6 import __version__ as v
    assert v.startswith("6.")


def test_main_window_import():
    """windows.main_window imports cleanly (module-level Qt classes only)."""
    import windows.main_window as mw
    assert hasattr(mw, "MainWindow")


def test_canvas_widget_class_structure():
    """CanvasWidget / MainWindow expose the attributes the app wires together."""
    from windows.main_window import MainWindow
    from windows.canvas_widget import CanvasWidget
    for attr in ("onPredictCurves", "initSignalAndSlots", "initComponentLayouts"):
        assert hasattr(MainWindow, attr)
    for attr in ("reDraw", "paintEvent", "mousePressEvent", "mouseMoveEvent", "mouseReleaseEvent"):
        assert hasattr(CanvasWidget, attr)


def test_qapplication_offscreen(qt_offscreen):
    from PySide6.QtWidgets import QApplication, QLabel
    app = QApplication.instance() or QApplication(sys.argv)
    label = QLabel("roto-plus offscreen smoke")
    label.resize(200, 40)
    label.show()
    app.processEvents()
    assert label.isVisible()
