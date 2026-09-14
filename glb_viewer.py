import sys
import vtk
import pyvista as pv
import os
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTreeWidget, QTreeWidgetItem, QPushButton, QSplitter, QLabel
)
from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap, QIcon
from pyvistaqt import QtInteractor

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ICON_PATH = os.path.join(BASE_DIR, "icon-1.png")

# VTK OpenGL 텍스처 수 초과 및 내부 경고 메시지 콘솔 도배 억제
vtk.vtkObject.GlobalWarningDisplayOff()
vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_ERROR)

# Qt 위젯 소멸 시 libshiboken C++ 객체 해제 순서로 인한 잔여 콜백 예외 억제
def _unraisable_hook(unraisable):
    if "already deleted" in str(unraisable.exc_value):
        return
    sys.__unraisablehook__(unraisable)

sys.unraisablehook = _unraisable_hook

class GLBViewerApp(QMainWindow):
    def __init__(self, glb_path):
        super().__init__()
        self.setWindowTitle("NanoStudio GLB Viewer")
        self.resize(1100, 750)
        
        if os.path.exists(ICON_PATH):
            self.setWindowIcon(QIcon(ICON_PATH))
        
        self.actors = {}
        
        # 메인 위젯 및 분할 패널(Splitter)
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(6, 6, 6, 6)
        
        splitter = QSplitter(Qt.Horizontal)
        main_layout.addWidget(splitter)
        
        # 1. 왼쪽 패널: 아웃라이너 트리
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        
        title_layout = QHBoxLayout()
        if os.path.exists(ICON_PATH):
            logo_lbl = QLabel()
            logo_lbl.setPixmap(QPixmap(ICON_PATH).scaled(20, 20, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            title_layout.addWidget(logo_lbl)
        title_label = QLabel("<b>부품 계층 구조 (아웃라이너)</b>")
        title_layout.addWidget(title_label)
        title_layout.addStretch()
        left_layout.addLayout(title_layout)
        
        btn_layout = QHBoxLayout()
        self.btn_select_all = QPushButton("☑️ 전체 선택")
        self.btn_deselect_all = QPushButton("☐ 전체 해제")
        self.btn_select_all.setToolTip("아웃라이너의 모든 부품 항목을 선택(표시)합니다.")
        self.btn_deselect_all.setToolTip("아웃라이너의 모든 부품 항목을 해제(숨김)합니다.")
        self.btn_select_all.clicked.connect(lambda: self.toggle_all(True))
        self.btn_deselect_all.clicked.connect(lambda: self.toggle_all(False))
        btn_layout.addWidget(self.btn_select_all)
        btn_layout.addWidget(self.btn_deselect_all)
        left_layout.addLayout(btn_layout)
        
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabels(["부품 이름", "타입"])
        self.tree_widget.setColumnWidth(0, 180)
        self.tree_widget.itemChanged.connect(self.on_item_changed)
        left_layout.addWidget(self.tree_widget)
        
        splitter.addWidget(left_panel)
        
        # 2. 오른쪽 패널: 3D 뷰어
        self.plotter = QtInteractor(self)
        splitter.addWidget(self.plotter.interactor)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 4)
        
        # 환경 설정
        self.plotter.set_background('white', top='lightblue')
        self.plotter.enable_eye_dome_lighting()
        
        # GLB 불러오기
        self.load_glb(glb_path)

    def toggle_all(self, visible):
        self.tree_widget.blockSignals(True)
        
        def _toggle_recursive(item):
            item.setCheckState(0, Qt.Checked if visible else Qt.Unchecked)
            name = item.data(0, Qt.UserRole)
            if name and name in self.actors:
                self.actors[name].SetVisibility(visible)
            for i in range(item.childCount()):
                _toggle_recursive(item.child(i))
                
        root = self.tree_widget.invisibleRootItem()
        for i in range(root.childCount()):
            _toggle_recursive(root.child(i))
            
        self.tree_widget.blockSignals(False)
        self.plotter.render()

    def on_item_changed(self, item, column):
        if column == 0:
            name = item.data(0, Qt.UserRole)
            if name in self.actors:
                is_visible = (item.checkState(0) == Qt.Checked)
                self.actors[name].SetVisibility(is_visible)
                self.plotter.render()

    def load_glb(self, path):
        try:
            scene = pv.read(path)
            self.tree_widget.blockSignals(True)
            
            if isinstance(scene, pv.MultiBlock):
                tag_items = {}
                for i in range(scene.n_blocks):
                    block_name = scene.keys()[i] if scene.keys() else f"Part_{i}"
                    sub_mesh = scene[i]
                    if sub_mesh is None:
                        continue
                    
                    if "/" in block_name:
                        tag_name, sub_name = block_name.split("/", 1)
                        if tag_name not in tag_items:
                            parent_item = QTreeWidgetItem(self.tree_widget)
                            parent_item.setText(0, tag_name)
                            parent_item.setText(1, "Tag/Layer")
                            parent_item.setData(0, Qt.UserRole, tag_name)
                            parent_item.setCheckState(0, Qt.Checked)
                            tag_items[tag_name] = parent_item
                        else:
                            parent_item = tag_items[tag_name]
                            
                        item = QTreeWidgetItem(parent_item)
                        item.setText(0, sub_name)
                        item.setText(1, type(sub_mesh).__name__)
                        item.setData(0, Qt.UserRole, block_name)
                        item.setCheckState(0, Qt.Checked)
                    else:
                        item = QTreeWidgetItem(self.tree_widget)
                        item.setText(0, block_name)
                        item.setText(1, type(sub_mesh).__name__)
                        item.setData(0, Qt.UserRole, block_name)
                        item.setCheckState(0, Qt.Checked)
                    
                    actor = self.plotter.add_mesh(sub_mesh, name=block_name, show_edges=False)
                    self.actors[block_name] = actor
                self.tree_widget.expandAll()
            else:
                item = QTreeWidgetItem(self.tree_widget)
                name = "Merged_Car"
                item.setText(0, name)
                item.setText(1, type(scene).__name__)
                item.setData(0, Qt.UserRole, name)
                item.setCheckState(0, Qt.Checked)
                actor = self.plotter.add_mesh(scene, name=name, show_edges=False)
                self.actors[name] = actor
                
            self.tree_widget.blockSignals(False)
            
            # 바닥 평면 추가
            bounds = self.plotter.bounds
            if bounds:
                center = [(bounds[0]+bounds[1])/2, (bounds[2]+bounds[3])/2, bounds[4]]
                plane = pv.Plane(center=center, direction=(0, 0, 1), i_size=200, j_size=200)
                self.plotter.add_mesh(plane, color='lightgray', name="Floor")
                
            self.plotter.reset_camera()
            self.plotter.view_isometric()
            
        except Exception as e:
            print(f"GLB 로드 실패: {e}")

    def closeEvent(self, event):
        if hasattr(self, 'plotter'):
            try:
                self.plotter.close()
            except Exception:
                pass
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    if os.path.exists(ICON_PATH):
        app.setWindowIcon(QIcon(ICON_PATH))
    viewer = GLBViewerApp("final_game_asset_car.glb")
    viewer.show()
    sys.exit(app.exec())
