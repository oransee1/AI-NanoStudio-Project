import sys
import os
import random
import warnings
import vtk
from dotenv import load_dotenv

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QFileDialog, QMessageBox, QTreeWidget, QTreeWidgetItem, QLabel, QTabWidget)
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPixmap, QIcon

# PyVista 및 VTK 경고 콘솔 출력 억제
warnings.filterwarnings("ignore", category=FutureWarning)
vtk.vtkObject.GlobalWarningDisplayOff()
vtk.vtkLogger.SetStderrVerbosity(vtk.vtkLogger.VERBOSITY_ERROR)

# Qt 위젯 소멸 시 libshiboken C++ 객체 해제 순서로 인한 잔여 콜백 예외 억제
def _unraisable_hook(unraisable):
    if "already deleted" in str(unraisable.exc_value):
        return
    sys.__unraisablehook__(unraisable)

sys.unraisablehook = _unraisable_hook
from viewport_widget import ViewportWidget
from skp_loader import load_skp_to_pyvista, load_obj_to_pyvista, export_meshes_to_obj

from camera_manager import CameraManager
from uv_unwrapper import UVUnwrapper
from shadow_baker import ShadowBaker
from genai_worker import GenAIWorker
from material_binder import MaterialBinder

# .env 파일 로드
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

from PySide6.QtCore import Qt, QObject, QEvent, QRect, QPoint, QSize
from PySide6.QtWidgets import QRubberBand

class RubberBandFilter(QObject):
    def __init__(self, target_widget, clear_callback=None):
        super().__init__(target_widget)
        self.target = target_widget
        self.clear_callback = clear_callback
        self.rubber_band = QRubberBand(QRubberBand.Rectangle, target_widget)
        self.origin = QPoint()
        self.is_active = False

    def set_active(self, active):
        self.is_active = active
        if not active:
            self.rubber_band.hide()

    def eventFilter(self, obj, event):
        if self.is_active and obj == self.target:
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self.origin = event.pos()
                self.rubber_band.setGeometry(QRect(self.origin, QSize()))
                self.rubber_band.show()
                return False 
            elif event.type() == QEvent.MouseMove and not self.origin.isNull():
                self.rubber_band.setGeometry(QRect(self.origin, event.pos()).normalized())
                return False
            elif event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                self.rubber_band.hide()
                # 드래그 거리가 짧으면(단순 클릭) 피킹 초기화
                if (event.pos() - self.origin).manhattanLength() < 5:
                    if self.clear_callback:
                        from PySide6.QtCore import QTimer
                        QTimer.singleShot(0, self.clear_callback)
                self.origin = QPoint()
                # 토글 버튼과 동기화되도록 자동 비활성화(is_active=False) 라인 삭제
                return False
        return super().eventFilter(obj, event)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI-NanoStudio - Final Integration")
        self.resize(1280, 720)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # 좌측 컨트롤 패널
        left_panel = QWidget()
        left_panel.setFixedWidth(300)
        left_layout = QVBoxLayout(left_panel)
        
        self.btn_load_skp = QPushButton("Load .SKP File")
        self.btn_load_obj = QPushButton("Load .OBJ File (Fallback)")
        self.btn_new_project = QPushButton("New Project")
        self.btn_save_anf = QPushButton("Save Project (.anf)")
        self.btn_load_anf = QPushButton("Load Project (.anf)")
        self.btn_export_obj = QPushButton("Export Project (.obj)")
        self.btn_auto_fix = QPushButton("Auto-Fix Mesh (Clean & Smooth)")
        self.btn_auto_fix.setStyleSheet("font-weight: bold; color: white; background-color: #2196F3; padding: 5px;")
        
        left_layout.addWidget(self.btn_load_skp)
        left_layout.addWidget(self.btn_load_obj)
        left_layout.addWidget(self.btn_new_project)
        left_layout.addWidget(self.btn_save_anf)
        left_layout.addWidget(self.btn_load_anf)
        left_layout.addWidget(self.btn_export_obj)
        left_layout.addWidget(self.btn_auto_fix)
        
        # QTabWidget 추가 (높이를 내용물에 맞게 고정)
        self.tabs = QTabWidget()
        self.tabs.setFixedHeight(300)
        left_layout.addWidget(self.tabs)
        
        # Tab 1: 건축/인테리어 (Part 1)
        self.tab_arch = QWidget()
        self.tab_arch_layout = QVBoxLayout(self.tab_arch)
        self.tab_arch_layout.addWidget(QLabel("건축/인테리어 렌더링"))
        self.btn_render_arch = QPushButton("오프스크린 PBR 실사 렌더링")
        self.btn_render_arch.setStyleSheet("font-weight: bold; color: white; background-color: #d84315; padding: 10px;")
        self.tab_arch_layout.addWidget(self.btn_render_arch)
        
        self.tab_arch_layout.addStretch()
        self.tabs.addTab(self.tab_arch, "Part 1: 렌더링")
        
        # Tab 2: 게임 배경/캐릭터 제작 (Part 2)
        self.tab_game = QWidget()
        self.tab_game_layout = QVBoxLayout(self.tab_game)
        self.btn_unwrap = QPushButton("UV 언랩 및 그림자 베이킹 실행")
        self.tab_game_layout.addWidget(self.btn_unwrap)
        self.btn_ai_texture = QPushButton("AI 텍스처 생성 및 적용")
        self.tab_game_layout.addWidget(self.btn_ai_texture)
        self.btn_load_hdri = QPushButton("외부 HDRI 맵 불러오기")
        self.tab_game_layout.addWidget(self.btn_load_hdri)
        self.btn_generate_hdri = QPushButton("AI HDRI 맵 생성하기")
        self.tab_game_layout.addWidget(self.btn_generate_hdri)
        self.btn_apply_pbr = QPushButton("PBR 머터리얼 적용 (선택 부품)")
        self.tab_game_layout.addWidget(self.btn_apply_pbr)
        self.tab_game_layout.addStretch()
        self.tabs.addTab(self.tab_game, "Part 2: 게임 에셋")
        
        # 아웃라이너(계층 구조) 표시 영역
        self.outliner_label = QLabel("객체 계층 구조 (Outliner):")
        left_layout.addWidget(self.outliner_label)
        
        # 계층 추가/관리 버튼
        self.outliner_btn_layout = QHBoxLayout()
        self.btn_add_node = QPushButton("➕ 계층 추가")
        self.btn_add_node.clicked.connect(self.add_custom_node)
        self.outliner_btn_layout.addWidget(self.btn_add_node)
        self.outliner_btn_layout.addStretch()
        left_layout.addLayout(self.outliner_btn_layout)

        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabels(["부품명"])
        self.tree_widget.itemChanged.connect(self.on_tree_item_changed)
        left_layout.addWidget(self.tree_widget)
        
        # 우측 뷰포트 패널
        self.viewport = ViewportWidget()
        
        # 시네마틱 뷰 상단에 폴리곤 편집용 툴바 추가
        self.cine_toolbar = QHBoxLayout()
        self.btn_pick_p = QPushButton("🖱️ 단일(표면) 면 선택 (P)")
        self.btn_pick_r = QPushButton("🔲 박스(관통) 면 선택 (R)")
        self.btn_clear_pick = QPushButton("❌ 선택 해제")
        self.btn_assign = QPushButton("✅ 선택된 면을 계층으로 할당")
        
        self.btn_pick_p.setCheckable(True)
        self.btn_pick_r.setCheckable(True)
        
        self.btn_pick_p.setToolTip("클릭 또는 작은 드래그로 표면의 면들을 선택합니다. (토글)")
        self.btn_pick_r.setToolTip("드래그하여 박스 영역 내 모든 면들을 관통 선택합니다. (토글)")
        self.btn_clear_pick.setToolTip("현재 선택된 면과 피킹 모드를 해제합니다.")
        self.btn_assign.setToolTip("선택된 핑크색 면들을 왼쪽 아웃라이너의 선택 계층으로 이동시킵니다.")

        from PySide6.QtWidgets import QComboBox
        self.combo_display_mode = QComboBox()
        self.combo_display_mode.addItems(["폴리곤 모드 (Polygon)", "솔리드 모드 (Solid)"])
        self.combo_display_mode.setToolTip("뷰포트의 렌더링 모드를 선택합니다.")

        self.cine_toolbar.addWidget(self.btn_pick_p)
        self.cine_toolbar.addWidget(self.btn_pick_r)
        self.cine_toolbar.addWidget(self.btn_clear_pick)
        self.cine_toolbar.addWidget(self.btn_assign)
        self.cine_toolbar.addSpacing(20)
        self.cine_toolbar.addWidget(QLabel("모드:"))
        self.cine_toolbar.addWidget(self.combo_display_mode)
        self.cine_toolbar.addStretch()
        
        # ViewportWidget 내부의 single_layout 최상단에 툴바 삽입
        self.viewport.single_layout.insertLayout(0, self.cine_toolbar)
        
        main_layout.addWidget(left_panel)
        main_layout.addWidget(self.viewport, stretch=1)
        
        # 매니저 및 데이터 초기화
        self.camera_manager = CameraManager(self.viewport.plotter_single)
        self.uv_unwrapper = UVUnwrapper()
        self.loaded_actors = {}
        self.atlas = None
        
        # 이벤트 연결
        self.btn_load_skp.clicked.connect(self.on_load_skp)
        self.btn_load_obj.clicked.connect(self.on_load_obj)
        self.btn_new_project.clicked.connect(self.on_new_project)
        self.btn_save_anf.clicked.connect(self.on_save_anf)
        self.btn_load_anf.clicked.connect(self.on_load_anf)
        self.btn_export_obj.clicked.connect(self.on_export_obj)
        self.btn_auto_fix.clicked.connect(self.on_btn_auto_fix)
        
        self.btn_render_arch.clicked.connect(self.on_render_arch)
        self.btn_unwrap.clicked.connect(self.on_unwrap_and_bake)
        self.btn_ai_texture.clicked.connect(self.on_ai_texture)
        self.btn_load_hdri.clicked.connect(self.on_load_hdri)
        self.btn_generate_hdri.clicked.connect(self.on_generate_hdri)
        self.btn_apply_pbr.clicked.connect(self.on_apply_pbr)
        
        self.btn_pick_p.clicked.connect(self.on_btn_pick_p)
        self.btn_pick_r.clicked.connect(self.on_btn_pick_r)
        self.btn_clear_pick.clicked.connect(self.on_btn_clear_pick)
        self.btn_assign.clicked.connect(self.on_btn_assign)
        self.combo_display_mode.currentIndexChanged.connect(self.on_display_mode_changed)
        
        # 자동 로드 비활성화 (빈 환경으로 시작)
        # auto_load_path = "3d_model/car.skp"
        # if os.path.exists(auto_load_path):
        #     self.load_skp_file(auto_load_path)

    def on_display_mode_changed(self, index):
        show_edges = (index == 0) # 0: 폴리곤, 1: 솔리드
        self.viewport.set_display_mode(show_edges)
        
    def load_skp_file(self, file_path):
        actors = load_skp_to_pyvista(file_path)
        if not actors:
            QMessageBox.warning(self, "Load Error", "SKP 파일을 로드하지 못했습니다. openskp 설치 상태를 확인하세요.")
            return
        self.loaded_actors = actors
        self.populate_scene(actors)
        
    def on_load_skp(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open SketchUp File", "", "SketchUp Files (*.skp)")
        if file_path:
            self.load_skp_file(file_path)
            
    def on_load_obj(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open OBJ File", "", "OBJ Files (*.obj)")
        if file_path:
            actors = load_obj_to_pyvista(file_path)
            if not actors:
                QMessageBox.warning(self, "Load Error", "OBJ 파일을 로드하지 못했습니다.")
                return
            self.loaded_actors = actors
            self.populate_scene(actors)
            
    def update_dimensions_text(self):
        """현재 씬에 로드된 메시들을 기반으로 전체 크기를 계산하여 뷰포트 텍스트를 업데이트합니다."""
        if not hasattr(self, 'part_meshes') or not self.part_meshes:
            return
            
        overall_bounds = [float('inf'), float('-inf'), float('inf'), float('-inf'), float('inf'), float('-inf')]
        for name, mesh in self.part_meshes.items():
            if not mesh or mesh.n_points == 0:
                continue
            b = mesh.bounds
            overall_bounds[0] = min(overall_bounds[0], b[0])
            overall_bounds[1] = max(overall_bounds[1], b[1])
            overall_bounds[2] = min(overall_bounds[2], b[2])
            overall_bounds[3] = max(overall_bounds[3], b[3])
            overall_bounds[4] = min(overall_bounds[4], b[4])
            overall_bounds[5] = max(overall_bounds[5], b[5])
            
        if overall_bounds[0] != float('inf'):
            size_x = overall_bounds[1] - overall_bounds[0]
            size_y = overall_bounds[3] - overall_bounds[2]
            size_z = overall_bounds[5] - overall_bounds[4]
            dim_text = f"X: {size_x:.3f}m, Y: {size_y:.3f}m, Z: {size_z:.3f}m"
            
            # 기존 Actor 삭제 후 다시 추가 방식이 가장 호환성이 높으므로 이렇게 진행
            try:
                self.viewport.plotter_single.remove_actor("dimensions")
            except:
                pass
            self.viewport.plotter_single.add_text(dim_text, position=(0.02, 0.92), font_size=5, viewport=True, color="#222222", name="dimensions")
            self.viewport.plotter_single.render()

    def populate_scene(self, actors_dict):
        """불러온 데이터를 화면과 아웃라이너에 세팅"""
        self.tree_widget.clear()
        
        # 랜덤 색상 생성용 헬퍼 함수
        def random_color():
            return [random.uniform(0.3, 0.9), random.uniform(0.3, 0.9), random.uniform(0.3, 0.9)]
            
        # 기존 메시 지우기 및 뷰포트 초기 세팅(그리드, 축 등)
        self.viewport.clear_and_setup()
            
        self.part_colors = {}
        self.part_meshes = actors_dict
        
        # 전체 바운딩 박스 크기 계산 (치수 정보 표시용)
        overall_bounds = [float('inf'), float('-inf'), float('inf'), float('-inf'), float('inf'), float('-inf')]
        
        # 각 메시 추가
        for name, mesh in actors_dict.items():
            b = mesh.bounds
            overall_bounds[0] = min(overall_bounds[0], b[0])
            overall_bounds[1] = max(overall_bounds[1], b[1])
            overall_bounds[2] = min(overall_bounds[2], b[2])
            overall_bounds[3] = max(overall_bounds[3], b[3])
            overall_bounds[4] = min(overall_bounds[4], b[4])
            overall_bounds[5] = max(overall_bounds[5], b[5])
            
            # 아웃라이너(트리)에 항목 추가
            item = QTreeWidgetItem(self.tree_widget, [name])
            item.setCheckState(0, Qt.Checked)
            
            # 각 부품별로 구분되는 색상 부여
            part_color = random_color()
            self.part_colors[name] = part_color
            
            # 아이콘 생성 (QPixmap 사용)
            pixmap = QPixmap(16, 16)
            qcolor = QColor(int(part_color[0]*255), int(part_color[1]*255), int(part_color[2]*255))
            pixmap.fill(qcolor)
            item.setIcon(0, QIcon(pixmap))
            
            self.viewport.add_mesh_to_all(mesh, name=name, color=part_color)

        self.update_dimensions_text()
            
        # 이벤트 리스너 연결
        self.tree_widget.itemChanged.connect(self.on_tree_item_changed)
            
        # UI 업데이트
        self.btn_ai_texture.setEnabled(True)
        self.btn_unwrap.setEnabled(True)

        # 폴리곤(면) 분할 기능을 위한 셀 피킹(Cell Picking) 비활성화 (버튼을 눌러야 활성화됨)
        self.picked_mesh = None

        # 기존의 Qt QRubberBand 필터를 복구하여 시각적 윤곽선을 보장합니다.
        if not hasattr(self, 'rubber_band_filter'):
            self.rubber_band_filter = RubberBandFilter(self.viewport.plotter_single.interactor, clear_callback=self.on_btn_clear_pick)
            self.viewport.plotter_single.interactor.installEventFilter(self.rubber_band_filter)

    def on_btn_pick_r(self):
        """박스 면 선택 모드 토글 (R 키 / 관통 선택)"""
        if not self.btn_pick_r.isChecked():
            self.on_btn_clear_pick()
            # 비활성화 시 QRubberBand도 끄기
            if hasattr(self, 'rubber_band_filter'):
                self.rubber_band_filter.set_active(False)
            return
            
        self.btn_pick_p.setChecked(False)
        
        # QRubberBand 시각화 활성화
        if hasattr(self, 'rubber_band_filter'):
            self.rubber_band_filter.set_active(True)

        try:
            self.viewport.plotter_single.disable_picking()
        except Exception:
            pass
            
        try:
            self.viewport.plotter_single.enable_cell_picking(
                callback=self.on_cell_picked, 
                through=True,
                show=True, 
                color='pink',
                show_message="마우스 왼쪽 버튼으로 드래그하여 영역을 선택하세요."
            )
            
            # 버튼 클릭 후 뷰포트에서 마우스 왼쪽 클릭 드래그만으로 바로 박스 선택이 되도록 강제 설정
            style = self.viewport.plotter_single.iren.get_interactor_style()
            if hasattr(style, 'StartSelect'):
                def force_start_select(obj, event):
                    obj.StartSelect()
                style.AddObserver("LeftButtonPressEvent", force_start_select)
        except Exception as e:
            print(f"피킹 활성화 오류: {e}")

    def on_btn_pick_p(self):
        """단일/표면 면 선택 모드 토글 (P 키 / 표면 선택)"""
        if not self.btn_pick_p.isChecked():
            self.on_btn_clear_pick()
            return
            
        self.btn_pick_r.setChecked(False)
        # 단일 면 선택이므로 박스 선택 윤곽선은 끕니다.
        if hasattr(self, 'rubber_band_filter'):
            self.rubber_band_filter.set_active(False)
        try:
            self.viewport.plotter_single.disable_picking()
        except Exception:
            pass
            
        try:
            self.viewport.plotter_single.enable_cell_picking(
                callback=self.on_cell_picked, 
                through=False,
                show=True, 
                color='pink',
                start=True,
                show_message="표면의 폴리곤을 클릭/드래그하여 선택하세요. (P)"
            )
        except Exception as e:
            print(f"피킹 활성화 오류: {e}")

    def on_btn_clear_pick(self):
        """피킹 모드 및 선택된 하이라이트 해제"""
        try:
            self.btn_pick_p.setChecked(False)
            self.btn_pick_r.setChecked(False)
            if hasattr(self, 'rubber_band_filter'):
                self.rubber_band_filter.set_active(False)
            try:
                self.viewport.plotter_single.disable_picking()
            except Exception:
                pass
                
            # 강제로 기본 인터랙터 스타일(TrackballCamera) 복구 (키보드/마우스 락 풀기)
            try:
                self.viewport.plotter_single.enable_trackball_style()
            except Exception:
                pass
                
            # 강제 클리어: 이름이 매칭되는 모든 액터 삭제 (버그 방지)
            for name in ["picked_cells", "my_picked_cells", "_picked_through_selection", "_picked_visible_selection", "_rectangle_selection_frustum"]:
                try:
                    self.viewport.plotter_single.remove_actor(name)
                except Exception:
                    pass
            self.picked_mesh = None
            self.viewport.plotter_single.render()
        except Exception as e:
            print(f"Clear pick error: {e}")

    def on_cell_picked(self, picked_mesh):
        """사용자가 화면에서 면(Cell)을 선택했을 때 호출됨"""
        import pyvista as pv
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import Qt
        
        modifiers = QApplication.keyboardModifiers()
        is_ctrl_pressed = bool(modifiers & Qt.ControlModifier)
        
        def _get_merged(mesh):
            if mesh is None: return None
            if isinstance(mesh, pv.MultiBlock):
                blocks = [b for b in mesh if b is not None and b.n_cells > 0]
                if not blocks: return None
                m = blocks[0]
                if len(blocks) > 1:
                    for b in blocks[1:]: m = m.merge(b)
                return m
            elif mesh.n_cells > 0:
                return mesh
            return None
            
        new_merged = _get_merged(picked_mesh)
        
        if is_ctrl_pressed and getattr(self, 'picked_mesh', None) is not None:
            old_merged = _get_merged(self.picked_mesh)
            if old_merged is not None and new_merged is not None:
                import numpy as np
                old_centers = old_merged.cell_centers().points
                new_centers = new_merged.cell_centers().points
                
                try:
                    from scipy.spatial import cKDTree
                    tree = cKDTree(old_centers)
                    indices_list = tree.query_ball_point(new_centers, r=0.001)
                except ImportError:
                    indices_list = []
                    for nc in new_centers:
                        dists = np.linalg.norm(old_centers - nc, axis=1)
                        matches = np.where(dists < 0.001)[0]
                        indices_list.append(matches)
                
                to_remove_from_old = set()
                to_add_from_new = set(range(new_merged.n_cells))
                
                for i, indices in enumerate(indices_list):
                    if len(indices) > 0:
                        to_remove_from_old.update(indices)
                        to_add_from_new.remove(i)
                
                parts = []
                keep_old = [i for i in range(old_merged.n_cells) if i not in to_remove_from_old]
                if keep_old:
                    parts.append(old_merged.extract_cells(keep_old))
                if to_add_from_new:
                    parts.append(new_merged.extract_cells(list(to_add_from_new)))
                
                if not parts:
                    final_mesh = pv.PolyData()
                elif len(parts) == 1:
                    final_mesh = parts[0]
                else:
                    final_mesh = parts[0].merge(parts[1])
                
                if not isinstance(final_mesh, pv.PolyData):
                    final_mesh = final_mesh.extract_surface(algorithm='dataset_surface')
            elif new_merged is not None:
                final_mesh = new_merged
            else:
                final_mesh = old_merged
        else:
            final_mesh = new_merged
            
        self.picked_mesh = final_mesh
        
        # 이전 하이라이트 지우기
        try:
            self.viewport.plotter_single.remove_actor("my_picked_cells")
        except Exception:
            pass
            
        # PyVista의 자체 하이라이트가 Qt 환경에서 누락될 수 있으므로, 명시적으로 핑크색 액터를 덧그립니다.
        if final_mesh is not None and final_mesh.n_cells > 0:
            self.viewport.plotter_single.add_mesh(
                final_mesh, 
                name="my_picked_cells", 
                color="magenta", 
                show_edges=True, 
                line_width=3, 
                pickable=False,
                reset_camera=False
            )
    def on_btn_assign(self):
        """선택된 폴리곤들을 현재 아웃라이너에서 선택된 계층으로 이동 및 색상 변경"""
        import numpy as np
        import pyvista as pv
        from PySide6.QtWidgets import QMessageBox

        if getattr(self, 'picked_mesh', None) is None:
            QMessageBox.information(self, "선택된 면 없음", "먼저 R 버튼으로 이동할 폴리곤을 선택해주세요.")
            return
            
        # MultiBlock 대응
        if isinstance(self.picked_mesh, pv.MultiBlock):
            blocks = [b for b in self.picked_mesh if b is not None and b.n_cells > 0]
            if not blocks:
                QMessageBox.information(self, "선택된 면 없음", "박스로 선택된 영역 내에 폴리곤이 없습니다.")
                return
            merged_picked = blocks[0].merge(blocks[1:]) if len(blocks) > 1 else blocks[0]
        else:
            merged_picked = self.picked_mesh
            
        if merged_picked.n_cells == 0:
            QMessageBox.information(self, "선택된 면 없음", "박스로 선택된 영역 내에 유효한 폴리곤이 없습니다.")
            return

        selected_items = self.tree_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "계층 미선택", "좌측 아웃라이너에서 할당받을 '계층 명칭'을 먼저 선택해주세요.")
            return

        target_name = selected_items[0].text(0)
        picked_centers = merged_picked.cell_centers().points

        # 이동할 셀 매핑 저장 {source_name: [cell_indices]}
        moves = {name: [] for name in self.part_meshes.keys()}
        
        # 어떤 원본 메시의 어떤 셀인지 매칭
        for name, mesh in self.part_meshes.items():
            if name == target_name or mesh.n_cells == 0: 
                continue
            
            mesh_centers = mesh.cell_centers().points
            
            # SciPy cKDTree로 빠르게 매칭 (스케일업에 의한 오차 및 스케치업 특유의 중복(겹침) 면을 모두 잡기 위해 query_ball_point 사용)
            try:
                from scipy.spatial import cKDTree
                tree = cKDTree(mesh_centers)
                # 반경 0.001(1mm) 이내의 '모든' 중심점 인덱스를 찾음 (Z-fighting 면들 동시 추출)
                indices_list = tree.query_ball_point(picked_centers, r=0.001)
                for indices in indices_list:
                    if indices:
                        moves[name].extend(indices)
            except ImportError:
                # Fallback for small selections
                for pc in picked_centers:
                    dists = np.linalg.norm(mesh_centers - pc, axis=1)
                    matched_indices = np.where(dists < 0.001)[0]
                    if len(matched_indices) > 0:
                        moves[name].extend(matched_indices.tolist())
                        
        changed = False
        if target_name not in self.part_meshes:
            self.part_meshes[target_name] = pv.PolyData()
            
        target_mesh = self.part_meshes[target_name]
        new_target_pieces = [target_mesh] if target_mesh.n_cells > 0 else []
        
        for source_name, cell_indices in moves.items():
            if not cell_indices: continue
            
            source_mesh = self.part_meshes[source_name]
            cell_indices = list(set(cell_indices)) # 중복 제거
            
            # 이동할 셀 추출 (타겟으로 추가)
            extracted = source_mesh.extract_cells(cell_indices)
            if not isinstance(extracted, pv.PolyData):
                extracted = extracted.extract_surface(algorithm='dataset_surface')
            new_target_pieces.append(extracted)
            
            # 남길 셀 추출 (소스에서 제거)
            all_cells = np.arange(source_mesh.n_cells)
            keep_cells = np.setdiff1d(all_cells, cell_indices)
            
            if len(keep_cells) > 0:
                updated_source = source_mesh.extract_cells(keep_cells)
                if not isinstance(updated_source, pv.PolyData):
                    updated_source = updated_source.extract_surface(algorithm='dataset_surface')
            else:
                updated_source = pv.PolyData()
                
            self.part_meshes[source_name] = updated_source
            self.viewport.update_mesh(source_name, updated_source, self.part_colors[source_name])
            changed = True
            
        if changed:
            # 타겟 메시 병합
            if len(new_target_pieces) > 1:
                merged_target = new_target_pieces[0].merge(new_target_pieces[1:])
            elif len(new_target_pieces) == 1:
                merged_target = new_target_pieces[0]
            else:
                merged_target = pv.PolyData()
                
            if not isinstance(merged_target, pv.PolyData):
                merged_target = merged_target.extract_surface(algorithm='dataset_surface')
                
            self.part_meshes[target_name] = merged_target
            self.viewport.update_mesh(target_name, merged_target, self.part_colors[target_name])
            
            # 메시지 창을 띄우기 '전'에 모든 피킹 상태와 하이라이트 레이어를 완벽하게 초기화
            self.on_btn_clear_pick()
            
            QMessageBox.information(self, "할당 완료", f"선택한 면들을 '{target_name}' 계층으로 할당했습니다.")
        else:
            self.on_btn_clear_pick()
            QMessageBox.information(self, "매칭 실패", "선택된 폴리곤과 원본 모델의 위치가 일치하지 않아 분할할 수 없습니다.")

    def add_custom_node(self):
        """사용자가 수동으로 새로운 계층(부품 그룹)을 추가하여 향후 폴리(Poly) 분리 등 가공에 활용"""
        from PySide6.QtWidgets import QInputDialog
        from PySide6.QtGui import QIcon, QPixmap, QColor
        from PySide6.QtCore import Qt
        import random
        
        text, ok = QInputDialog.getText(self, "계층 추가", "새로운 부품(계층) 이름을 입력하세요:")
        if ok and text:
            # 중복 체크
            if hasattr(self, 'part_colors') and text in self.part_colors:
                return
                
            item = QTreeWidgetItem(self.tree_widget)
            item.setText(0, text)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEditable)
            item.setCheckState(0, Qt.Checked)
            
            # 새 계층에 랜덤 색상 부여
            color = [random.uniform(0.3, 0.9), random.uniform(0.3, 0.9), random.uniform(0.3, 0.9)]
            if not hasattr(self, 'part_colors'):
                self.part_colors = {}
            self.part_colors[text] = color
            
            pixmap = QPixmap(16, 16)
            pixmap.fill(QColor(int(color[0]*255), int(color[1]*255), int(color[2]*255)))
            item.setIcon(0, QIcon(pixmap))

    def on_tree_item_changed(self, item, column):
        name = item.text(0)
        is_checked = (item.checkState(0) == Qt.Checked)
        
        # 뷰포트의 모든 플로터에서 해당 액터의 가시성 토글
        for plotter in self.viewport.plotters:
            for actor in plotter.renderer.actors.values():
                if hasattr(actor, 'name') and actor.name == name:
                    actor.SetVisibility(is_checked)
            plotter.render()

    def on_render_arch(self):
        """팝업창(QDialog)을 띄우고 그 안에 인터랙티브 PyVista 뷰어를 넣어 사용자가 시네마틱 구도를 잡을 수 있게 합니다."""
        import pyvista as pv
        import os
        from pyvistaqt import QtInteractor
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton, QMessageBox
        from PySide6.QtCore import Qt
        import datetime, shutil

        try:
            QMessageBox.information(self, "시네마틱 PBR 렌더링", "새 창이 열리면 마우스로 원하는 '시네마틱' 구도를 자유롭게 잡아보세요!\n(배경(HDRI) 로딩에 몇 초 정도 소요될 수 있습니다.)")
            
            dialog = QDialog(self)
            dialog.setWindowTitle("시네마틱 구도 설정 및 렌더링")
            dialog.resize(1280, 720)
            
            layout = QVBoxLayout(dialog)
            
            # 상단 설명 라벨
            lbl_info = QLabel("마우스 좌클릭(회전), 우클릭(줌), 휠클릭(이동)으로 완벽한 구도를 잡으세요.\n구도를 잡은 후 아래 버튼을 누르면 4677x2631 초고해상도로 렌더링되어 저장됩니다.")
            lbl_info.setAlignment(Qt.AlignCenter)
            lbl_info.setStyleSheet("font-weight: bold; font-size: 14px; padding: 5px;")
            layout.addWidget(lbl_info)
            
            # 대화형 PyVista 인터랙터 생성
            plotter = QtInteractor(dialog)
            layout.addWidget(plotter.interactor)
            
            # HDRI 맵 및 환경 설정
            cubemap = pv.examples.download_sky_box_cube_map()
            plotter.set_environment_texture(cubemap)
            
            # 스카이박스의 위쪽 축을 Z축(0,0,1)으로 설정하여 올바른 하늘 방향을 잡습니다
            skybox = cubemap.to_skybox()
            try:
                skybox.SetFloorPlane(0, 0, 1, 0)
                skybox.SetFloorRight(1, 0, 0)
            except:
                pass
            plotter.add_actor(skybox)
            
            try:
                plotter.enable_shadows()
            except Exception:
                pass
            plotter.enable_anti_aliasing("msaa")
            
            # 현재 뷰어의 로드된 액터(메쉬)들 복제
            orig_plotter = self.viewport.plotter_single
            for name, actor in orig_plotter.actors.items():
                if name in ["Axes", "Grid", "Border"] or "axes" in name.lower() or "grid" in name.lower():
                    continue
                    
                if hasattr(actor, 'mapper') and hasattr(actor.mapper, 'dataset'):
                    mesh = actor.mapper.dataset
                    if not isinstance(mesh, (pv.PolyData, pv.UnstructuredGrid)):
                        continue
                        
                    color = actor.prop.color if hasattr(actor.prop, 'color') else 'white'
                    new_actor = plotter.add_mesh(mesh, color=color, show_edges=False)
                    
                    # PBR 재질 부여
                    try:
                        new_actor.prop.interpolation = 'pbr'
                        new_actor.prop.metallic = 0.8
                        new_actor.prop.roughness = 0.15
                    except:
                        pass
                        
            # 초기 카메라 동기화
            orig_camera = orig_plotter.camera
            plotter.camera.position = orig_camera.position
            plotter.camera.focal_point = orig_camera.focal_point
            plotter.camera.up = orig_camera.up
            plotter.camera.view_angle = orig_camera.view_angle
            
            # 저장 버튼
            btn_save = QPushButton("현재 뷰어 구도로 렌더링 이미지 저장하기 (4677x2631)")
            btn_save.setFixedHeight(50)
            btn_save.setStyleSheet("font-size: 16px; font-weight: bold; background-color: #4CAF50; color: white;")
            layout.addWidget(btn_save)
            
            def save_action():
                try:
                    today_str = datetime.datetime.now().strftime("%Y-%m-%d")
                    base_dir = r"C:\Users\OWNER\PycharmProjects\AI-NanoStudio-Project\Output"
                    target_dir = os.path.join(base_dir, today_str)
                    
                    if not os.path.exists(target_dir):
                        os.makedirs(target_dir)
                        
                    timestamp = datetime.datetime.now().strftime("%H%M%S")
                    target_path = os.path.join(target_dir, f"render_{timestamp}.png")
                    
                    # QtInteractor에서 screenshot 호출 시 return_img=False, window_size 인자 사용
                    plotter.screenshot(target_path, transparent_background=False, window_size=[4677, 2631])
                    QMessageBox.information(dialog, "저장 완료", f"초고해상도 렌더링 이미지가 저장되었습니다!\n\n경로: {target_path}")
                except Exception as e:
                    QMessageBox.critical(dialog, "오류", f"저장 중 오류 발생:\n{e}")
                    
            btn_save.clicked.connect(save_action)
            dialog.exec()

        except Exception as e:
            from PySide6.QtWidgets import QMessageBox
            QMessageBox.warning(self, "오류", f"PBR 렌더링 중 오류가 발생했습니다:\n{e}")

    def on_unwrap_and_bake(self):
        if not self.loaded_actors:
            QMessageBox.warning(self, "Error", "먼저 모델을 로드해주세요.")
            return
            
        try:
            # UV 언랩
            self.atlas, uv_img = self.uv_unwrapper.unwrap_and_pack(self.loaded_actors, part_colors=getattr(self, 'part_colors', None))
            
            # 그림자 베이킹
            baker = ShadowBaker(self.loaded_actors)
            self.shadow_map_path = baker.bake_ambient_and_sunlight(self.atlas)
            
            QMessageBox.information(self, "Success", f"UV 언랩 및 베이킹 완료!\n- {uv_img}\n- {self.shadow_map_path}")
        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

    def on_ai_texture(self):
        if not GEMINI_API_KEY:
            QMessageBox.warning(self, "API Key Error", ".env 파일에 GEMINI_API_KEY가 설정되어 있지 않습니다.")
            return
            
        if not getattr(self, "shadow_map_path", None):
            QMessageBox.warning(self, "Error", "먼저 UV 언랩 및 베이킹을 실행해주세요.")
            return
            
        # 게임 텍스처 생성 호출
        prompt = "Hand-painted stylized RPG game texture, highly detailed, vibrant colors"
        style_preset = "game asset, flat diffuse map, clean UV map texture"
        self.worker = GenAIWorker(GEMINI_API_KEY, prompt, self.shadow_map_path, style_preset, aspect_ratio="1:1")
        self.worker.finished_signal.connect(self.on_texture_finished)
        self.worker.error_signal.connect(self.on_worker_error)
        self.worker.start()
        QMessageBox.information(self, "Processing", "AI 텍스처 생성을 요청했습니다...")

    def on_texture_finished(self, status, output_path):
        try:
            # 텍스처 합성은 한 번만 수행 (auto_apply_texture 내에서 합성 이미지 저장됨)
            first_mesh_name = list(self.loaded_actors.keys())[0]
            first_mesh = self.loaded_actors[first_mesh_name]
            final_tex_path = MaterialBinder.auto_apply_texture(first_mesh, output_path, self.shadow_map_path)
            
            # 모든 메시에 PBR(디퓨즈) 바인딩 적용
            for actor_name in self.loaded_actors.keys():
                MaterialBinder.apply_pbr_to_pyvista_actor(
                    plotter=self.viewport.plotter_single,
                    actor_name=actor_name,
                    diffuse_path=final_tex_path
                )
                
            QMessageBox.information(self, "Success", "모든 부품에 텍스처 합성 및 바인딩이 완료되었습니다.")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"바인딩 중 오류 발생: {str(e)}")

    def on_load_hdri(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "HDRI 맵 불러오기", "", "HDR/EXR Files (*.hdr *.exr);;All Images (*.png *.jpg)")
        if file_path:
            success = self.viewport.set_hdri_environment(file_path)
            if success:
                QMessageBox.information(self, "Success", "HDRI 환경 맵이 성공적으로 적용되었습니다.")
            else:
                QMessageBox.warning(self, "Error", "HDRI 환경 맵 적용에 실패했습니다.")

    def on_generate_hdri(self):
        if not GEMINI_API_KEY:
            QMessageBox.warning(self, "API Key Error", ".env 파일에 GEMINI_API_KEY가 설정되어 있지 않습니다.")
            return
            
        prompt = "360 degree panoramic HDRI environment map, equirectangular projection, beautiful natural landscape, photorealistic lighting, seamless"
        style_preset = "HDR environment, seamless 360 panorama"
        
        # 16:9 비율로 파노라마 맵 생성 요청 (나노바나나는 16:9 지원)
        self.hdri_worker = GenAIWorker(GEMINI_API_KEY, prompt, image_path=None, style_preset=style_preset, aspect_ratio="16:9")
        self.hdri_worker.finished_signal.connect(self.on_hdri_finished)
        self.hdri_worker.error_signal.connect(self.on_worker_error)
        self.hdri_worker.start()
        QMessageBox.information(self, "Processing", "AI HDRI 환경 맵 생성을 요청했습니다. (파노라마 이미지)")

    def on_hdri_finished(self, status, output_path):
        success = self.viewport.set_hdri_environment(output_path)
        if success:
            QMessageBox.information(self, "Success", f"생성된 AI HDRI 맵이 적용되었습니다.\n경로: {output_path}")
        else:
            QMessageBox.warning(self, "Error", "생성된 HDRI 맵을 적용하는 데 실패했습니다.")

    def on_worker_error(self, error_msg):
        QMessageBox.warning(self, "Error", f"AI 작업 중 오류 발생:\n{error_msg}")

    def on_apply_pbr(self):
        selected_items = self.tree_widget.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Error", "PBR 맵을 적용할 객체를 계층 구조(Outliner)에서 선택해주세요.")
            return
            
        actor_name = selected_items[0].text(0)
        
        QMessageBox.information(self, "안내", f"'{actor_name}' 객체에 적용할 PBR 텍스처를 순서대로 선택합니다. (취소 시 건너뜁니다)")
        
        diffuse_path, _ = QFileDialog.getOpenFileName(self, "1. Diffuse(Albedo) 맵 선택", "", "Images (*.png *.jpg *.jpeg)")
        normal_path, _ = QFileDialog.getOpenFileName(self, "2. Normal 맵 선택", "", "Images (*.png *.jpg *.jpeg)")
        roughness_path, _ = QFileDialog.getOpenFileName(self, "3. Roughness 맵 선택", "", "Images (*.png *.jpg *.jpeg)")
        metallic_path, _ = QFileDialog.getOpenFileName(self, "4. Metallic 맵 선택", "", "Images (*.png *.jpg *.jpeg)")

        success = MaterialBinder.apply_pbr_to_pyvista_actor(
            plotter=self.viewport.plotter_single,
            actor_name=actor_name,
            diffuse_path=diffuse_path,
            normal_path=normal_path,
            roughness_path=roughness_path,
            metallic_path=metallic_path
        )
        if success:
            QMessageBox.information(self, "Success", f"'{actor_name}'에 PBR 적용이 완료되었습니다.")
        else:
            QMessageBox.warning(self, "Error", "PBR 머티리얼 적용에 실패했습니다.")

    def on_save_anf(self):
        """현재 프로젝트 상태를 .anf 형식으로 저장"""
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Project", "", "AI NanoStudio Format (*.anf)")
        if not file_path: return False
        import json, zipfile, tempfile, shutil, os
        import pyvista as pv
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                metadata = {"parts": [], "colors": {}, "visibility": {}}
                root = self.tree_widget.invisibleRootItem()
                for i in range(root.childCount()):
                    item = root.child(i)
                    name = item.text(0)
                    metadata["parts"].append(name)
                    metadata["visibility"][name] = (item.checkState(0) == Qt.Checked)
                    if name in getattr(self, 'part_colors', {}):
                        metadata["colors"][name] = self.part_colors[name]
                    if name in getattr(self, 'part_meshes', {}):
                        mesh_path = os.path.join(tmpdir, f"{name}.vtp")
                        self.part_meshes[name].save(mesh_path)
                with open(os.path.join(tmpdir, "metadata.json"), "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
                with zipfile.ZipFile(file_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for root_dir, _, files in os.walk(tmpdir):
                        for f in files:
                            abs_path = os.path.join(root_dir, f)
                            zf.write(abs_path, arcname=os.path.relpath(abs_path, tmpdir))
            QMessageBox.information(self, "저장 완료", "프로젝트가 성공적으로 저장되었습니다.")
            return True
        except Exception as e:
            QMessageBox.critical(self, "저장 오류", f"저장 중 오류 발생:\n{e}")
            return False

    def reset_project(self):
        """현재 씬 및 모든 데이터를 초기화합니다."""
        self.loaded_actors = {}
        if hasattr(self, 'part_meshes'):
            self.part_meshes = {}
        if hasattr(self, 'part_colors'):
            self.part_colors = {}
        self.tree_widget.clear()
        self.viewport.clear_and_setup()
        self.btn_ai_texture.setEnabled(False)
        self.btn_unwrap.setEnabled(False)
        # 렌더링 강제 업데이트
        for plotter in getattr(self.viewport, 'plotters', []):
            plotter.render()
        QMessageBox.information(self, "New Project", "새 프로젝트가 시작되었습니다.")

    def on_new_project(self):
        # 작업 중인 데이터가 있는지 확인
        if not hasattr(self, 'part_meshes') or not self.part_meshes:
            # 빈 상태라면 바로 초기화
            self.reset_project()
            return
            
        reply = QMessageBox.question(
            self, 'New Project', 
            '현재 진행 중인 프로젝트를 저장하시겠습니까?\n(저장하지 않으면 모든 변경 사항이 사라집니다.)',
            QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel, 
            QMessageBox.Save
        )
        
        if reply == QMessageBox.Save:
            success = self.on_save_anf()
            if success:
                self.reset_project()
        elif reply == QMessageBox.Discard:
            self.reset_project()
        else:
            # Cancel: 작업 취소
            pass

    def on_load_anf(self):
        """저장된 .anf 프로젝트 불러오기"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Project", "", "AI NanoStudio Format (*.anf)")
        if not file_path: return
        import json, zipfile, tempfile, os
        import pyvista as pv
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                with zipfile.ZipFile(file_path, 'r') as zf:
                    zf.extractall(tmpdir)
                with open(os.path.join(tmpdir, "metadata.json"), "r", encoding="utf-8") as f:
                    metadata = json.load(f)
                    
                self.part_meshes = {}
                self.part_colors = {}
                self.tree_widget.clear()
                
                # 기존 뷰포트 초기화
                for plotter in getattr(self.viewport, 'plotters', []):
                    plotter.clear()
                self.viewport.setup_viewport_aesthetics()
                
                # 데이터 복원
                for name in metadata["parts"]:
                    mesh_path = os.path.join(tmpdir, f"{name}.vtp")
                    if os.path.exists(mesh_path):
                        self.part_meshes[name] = pv.read(mesh_path)
                    if name in metadata.get("colors", {}):
                        self.part_colors[name] = metadata["colors"][name]
                    visibility = metadata.get("visibility", {}).get(name, True)
                    
                    item = QTreeWidgetItem([name])
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    item.setCheckState(0, Qt.Checked if visibility else Qt.Unchecked)
                    self.tree_widget.addTopLevelItem(item)
                    
                    # 체크되어 있으면 화면에 표시
                    if visibility and name in self.part_meshes:
                        self.viewport.update_mesh(name, self.part_meshes[name], self.part_colors.get(name, "gray"))
                        
                for plotter in getattr(self.viewport, 'plotters', []):
                    plotter.reset_camera()
                    plotter.render()
                
                self.update_dimensions_text()
                
            QMessageBox.information(self, "불러오기 완료", "프로젝트를 성공적으로 불러왔습니다.")
        except Exception as e:
            QMessageBox.critical(self, "불러오기 오류", f"불러오기 중 오류 발생:\n{e}")

    def on_btn_auto_fix(self):
        """메시 자동 최적화 및 스무딩 다이얼로그 호출"""
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QCheckBox, QPushButton
        from PySide6.QtCore import Qt
        
        if not hasattr(self, 'part_meshes') or not self.part_meshes:
            QMessageBox.information(self, "데이터 없음", "먼저 3D 모델을 불러와주세요.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Auto-Fix Mesh (형태 최적화)")
        dialog.resize(400, 250)
        layout = QVBoxLayout(dialog)
        
        # 1. Feature Angle (Edge split)
        layout_angle = QHBoxLayout()
        lbl_angle = QLabel("하드 엣지 각도 (Feature Angle): 45도")
        slider_angle = QSlider(Qt.Horizontal)
        slider_angle.setRange(0, 180)
        slider_angle.setValue(45)
        slider_angle.valueChanged.connect(lambda v: lbl_angle.setText(f"하드 엣지 각도 (Feature Angle): {v}도"))
        layout_angle.addWidget(lbl_angle)
        layout_angle.addWidget(slider_angle)
        layout.addLayout(layout_angle)
        
        # 2. Laplacian Smoothing Iterations
        layout_smooth = QHBoxLayout()
        lbl_smooth = QLabel("표면 부드럽게 (Smooth Iterations): 0")
        slider_smooth = QSlider(Qt.Horizontal)
        slider_smooth.setRange(0, 100)
        slider_smooth.setValue(0)
        slider_smooth.valueChanged.connect(lambda v: lbl_smooth.setText(f"표면 부드럽게 (Smooth Iterations): {v}"))
        layout_smooth.addWidget(lbl_smooth)
        layout_smooth.addWidget(slider_smooth)
        layout.addLayout(layout_smooth)
        
        # 3. Subdivision Surface
        chk_subdiv = QCheckBox("표면 세분화 적용 (Subdivision Surface - Linear)")
        layout.addWidget(chk_subdiv)
        
        # Apply & Cancel
        btn_layout = QHBoxLayout()
        btn_apply = QPushButton("적용 (Apply)")
        btn_cancel = QPushButton("취소 (Cancel)")
        btn_layout.addWidget(btn_apply)
        btn_layout.addWidget(btn_cancel)
        layout.addLayout(btn_layout)
        
        def apply_fixes():
            angle = slider_angle.value()
            iters = slider_smooth.value()
            do_subdiv = chk_subdiv.isChecked()
            
            QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                import numpy as np
                import pyvista as pv
                
                # 피킹된 면이 있는지 확인
                has_picked = False
                picked_centers = None
                if getattr(self, 'picked_mesh', None) is not None:
                    if isinstance(self.picked_mesh, pv.MultiBlock):
                        blocks = [b for b in self.picked_mesh if b is not None and b.n_cells > 0]
                        if blocks:
                            merged_picked = blocks[0].merge(blocks[1:]) if len(blocks) > 1 else blocks[0]
                            if merged_picked.n_cells > 0:
                                picked_centers = merged_picked.cell_centers().points
                                has_picked = True
                    elif self.picked_mesh.n_cells > 0:
                        picked_centers = self.picked_mesh.cell_centers().points
                        has_picked = True
                
                # 처리할 메시와 인덱스 수집
                targets = {} # {name: cell_indices_to_process or None for all}
                if has_picked:
                    for name, mesh in self.part_meshes.items():
                        if mesh.n_cells == 0: continue
                        mesh_centers = mesh.cell_centers().points
                        matched_indices = []
                        try:
                            from scipy.spatial import cKDTree
                            tree = cKDTree(mesh_centers)
                            indices_list = tree.query_ball_point(picked_centers, r=0.001)
                            for indices in indices_list:
                                if indices: matched_indices.extend(indices)
                        except ImportError:
                            for pc in picked_centers:
                                dists = np.linalg.norm(mesh_centers - pc, axis=1)
                                matches = np.where(dists < 0.001)[0]
                                if len(matches) > 0: matched_indices.extend(matches.tolist())
                        
                        if matched_indices:
                            targets[name] = list(set(matched_indices))
                else:
                    # 선택된 면이 없으면 전체 메시를 대상으로 함
                    for name, mesh in self.part_meshes.items():
                        if mesh.n_cells > 0:
                            targets[name] = None
                            
                if not targets:
                    QMessageBox.information(dialog, "선택 영역 없음", "처리할 유효한 폴리곤을 찾지 못했습니다.")
                    QApplication.restoreOverrideCursor()
                    return

                changed = False
                for name, indices in targets.items():
                    original_mesh = self.part_meshes[name]
                    
                    if indices is None or len(indices) == original_mesh.n_cells:
                        # 전체 처리
                        mesh_to_process = original_mesh
                        keep_mesh = None
                    else:
                        # 부분 처리
                        mesh_to_process = original_mesh.extract_cells(indices)
                        if not isinstance(mesh_to_process, pv.PolyData):
                            mesh_to_process = mesh_to_process.extract_surface(algorithm='dataset_surface')
                            
                        all_cells = np.arange(original_mesh.n_cells)
                        keep_cells = np.setdiff1d(all_cells, indices)
                        keep_mesh = original_mesh.extract_cells(keep_cells)
                        if not isinstance(keep_mesh, pv.PolyData):
                            keep_mesh = keep_mesh.extract_surface(algorithm='dataset_surface')

                    # 1. Clean
                    fixed_mesh = mesh_to_process.clean()
                    
                    if fixed_mesh.n_faces > 0:
                        # 2. Subdivide
                        if do_subdiv:
                            try:
                                sub = fixed_mesh.subdivide(1, subfilter='linear')
                                if sub.n_faces > 0: fixed_mesh = sub
                            except: pass
                            
                        # 3. Smooth
                        if iters > 0:
                            try:
                                sm = fixed_mesh.smooth(n_iter=iters, relaxation_factor=0.1, feature_smoothing=True)
                                if sm.n_faces > 0: fixed_mesh = sm
                            except: pass
                            
                        # 4. Normals
                        if fixed_mesh.n_faces > 0:
                            try:
                                fixed_mesh = fixed_mesh.compute_normals(split_vertices=True, feature_angle=angle)
                            except: pass
                    
                    # 병합
                    if keep_mesh is not None and keep_mesh.n_cells > 0:
                        final_mesh = keep_mesh.merge(fixed_mesh)
                    else:
                        final_mesh = fixed_mesh
                        
                    self.part_meshes[name] = final_mesh
                    color = self.part_colors.get(name, "#CCCCCC")
                    self.viewport.update_mesh(name, final_mesh, color)
                    changed = True
                    
                if changed:
                    self.on_btn_clear_pick()
                    for plotter in self.viewport.plotters:
                        plotter.render()
                    
                dialog.accept()
                QMessageBox.information(self, "완료", "선택한(또는 전체) 형태 매시가 성공적으로 조정 및 최적화 되었습니다.")
            except Exception as e:
                QMessageBox.critical(self, "오류", f"최적화 중 오류가 발생했습니다:\n{e}")
            finally:
                QApplication.restoreOverrideCursor()
                
        btn_apply.clicked.connect(apply_fixes)
        btn_cancel.clicked.connect(dialog.reject)
        
        dialog.exec()

    def on_export_obj(self):
        if not hasattr(self, 'part_meshes') or not self.part_meshes:
            QMessageBox.warning(self, "Export Error", "내보낼 모델이 없습니다. 먼저 모델을 로드하세요.")
            return
            
        try:
            from datetime import datetime
            import os
            
            # 기본 경로(Export/YYYY-MM-DD) 설정
            base_export_dir = r"C:\Users\OWNER\PycharmProjects\AI-NanoStudio-Project\Export"
            date_str = datetime.now().strftime("%Y-%m-%d")
            export_dir = os.path.join(base_export_dir, date_str)
            
            if not os.path.exists(export_dir):
                os.makedirs(export_dir)
                
            default_path = os.path.join(export_dir, "exported_model.obj")
            
            # 사용자에게 저장 경로 및 파일명 입력받기
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Export OBJ File", default_path, "OBJ Files (*.obj)"
            )
            
            if not file_path:
                return # 취소됨
                
            # skp_loader.py의 export_meshes_to_obj 함수 호출 (선택된 경로로 저장)
            saved_path = export_meshes_to_obj(self.part_meshes, output_path=file_path)
            
            if saved_path:
                QMessageBox.information(self, "Export 완료", f"계층구조가 유지된 OBJ 파일이 저장되었습니다:\n{saved_path}")
            else:
                QMessageBox.warning(self, "Export 실패", "내보내기 과정에서 오류가 발생했습니다.")
        except Exception as e:
            QMessageBox.critical(self, "Export 오류", f"오류 발생:\n{e}")

    def closeEvent(self, event):
        if hasattr(self, 'viewport'):
            for plotter in getattr(self.viewport, 'plotters', []):
                try:
                    plotter.close()
                except Exception:
                    pass
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
