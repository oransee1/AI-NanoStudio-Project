import sys
import vtk
import pyvista as pv
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QTreeWidget, QTreeWidgetItem, QPushButton, QSplitter, QLabel
)
from PySide6.QtCore import Qt
from pyvistaqt import QtInteractor

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
        
        title_label = QLabel("<b>부품 계층 구조 (아웃라이너)</b>")
        left_layout.addWidget(title_label)
        
        btn_layout = QHBoxLayout()
        self.btn_all_show = QPushButton("전체 표시")
        self.btn_all_hide = QPushButton("전체 숨김")
        self.btn_all_show.clicked.connect(lambda: self.toggle_all(True))
        self.btn_all_hide.clicked.connect(lambda: self.toggle_all(False))
        btn_layout.addWidget(self.btn_all_show)
        btn_layout.addWidget(self.btn_all_hide)
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
        for i in range(self.tree_widget.topLevelItemCount()):
            item = self.tree_widget.topLevelItem(i)
            item.setCheckState(0, Qt.Checked if visible else Qt.Unchecked)
            name = item.data(0, Qt.UserRole)
            if name in self.actors:
                self.actors[name].SetVisibility(visible)
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
                for i in range(scene.n_blocks):
                    block_name = scene.keys()[i] if scene.keys() else f"Part_{i}"
                    sub_mesh = scene[i]
                    if sub_mesh is None:
                        continue
                    
                    item = QTreeWidgetItem(self.tree_widget)
                    item.setText(0, block_name)
                    item.setText(1, type(sub_mesh).__name__)
                    item.setData(0, Qt.UserRole, block_name)
                    item.setCheckState(0, Qt.Checked)
                    
                    actor = self.plotter.add_mesh(sub_mesh, name=block_name, show_edges=False)
                    self.actors[block_name] = actor
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
    viewer = GLBViewerApp("final_game_asset_car.glb")
    viewer.show()
    sys.exit(app.exec())
