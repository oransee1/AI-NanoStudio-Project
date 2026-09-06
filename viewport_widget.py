from PySide6.QtWidgets import QWidget, QVBoxLayout, QStackedWidget, QGridLayout, QPushButton, QHBoxLayout
from pyvistaqt import QtInteractor

class ViewportWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # 뷰 모드 전환 버튼
        self.btn_layout = QHBoxLayout()
        self.btn_single = QPushButton("시네마틱 뷰 (단일)")
        self.btn_quad = QPushButton("직교 뷰 (2x2)")
        self.btn_layout.addWidget(self.btn_single)
        self.btn_layout.addWidget(self.btn_quad)
        self.layout.addLayout(self.btn_layout)
        
        # 스택 위젯 생성
        self.stacked_widget = QStackedWidget()
        self.layout.addWidget(self.stacked_widget)
        
        # 1. 단일 시네마틱 뷰포트
        self.single_view_widget = QWidget()
        self.single_layout = QVBoxLayout(self.single_view_widget)
        self.single_layout.setContentsMargins(0, 0, 0, 0)
        self.plotter_single = QtInteractor(self.single_view_widget)
        self.single_layout.addWidget(self.plotter_single.interactor)
        
        # 2. 2x2 직교 뷰포트
        self.quad_view_widget = QWidget()
        self.quad_layout = QGridLayout(self.quad_view_widget)
        self.quad_layout.setContentsMargins(0, 0, 0, 0)
        
        self.plotter_tl = QtInteractor(self.quad_view_widget) # Top
        self.plotter_tr = QtInteractor(self.quad_view_widget) # Perspective
        self.plotter_bl = QtInteractor(self.quad_view_widget) # Front
        self.plotter_br = QtInteractor(self.quad_view_widget) # Right
        
        self.quad_layout.addWidget(self.plotter_tl.interactor, 0, 0)
        self.quad_layout.addWidget(self.plotter_tr.interactor, 0, 1)
        self.quad_layout.addWidget(self.plotter_bl.interactor, 1, 0)
        self.quad_layout.addWidget(self.plotter_br.interactor, 1, 1)
        
        # 기본 카메라 설정
        self.plotter_tl.view_xy()
        self.plotter_bl.view_xz()
        self.plotter_br.view_yz()
        self.plotter_tr.view_isometric()
        
        # 직교 뷰 모드
        self.plotter_tl.enable_parallel_projection()
        self.plotter_bl.enable_parallel_projection()
        self.plotter_br.enable_parallel_projection()
        
        self.stacked_widget.addWidget(self.single_view_widget)
        self.stacked_widget.addWidget(self.quad_view_widget)
        
        # 버튼 이벤트
        self.btn_single.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        self.btn_quad.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        
        self.plotters = [
            self.plotter_single,
            self.plotter_tl, self.plotter_tr, self.plotter_bl, self.plotter_br
        ]

        self.setup_quad_cameras()
        self.setup_viewport_aesthetics()

    def setup_viewport_aesthetics(self):
        """Maya 스타일의 옅은 회색 배경과 1m(1000유닛) 간격의 짙은 회색 바닥 그리드 생성"""
        import pyvista as pv
        bg_color = "#D3D3D3" # 옅은 회색
        grid_color = "#555555" # 짙은 회색
        
        for plotter in self.plotters:
            plotter.set_background(bg_color)
            
        # 각 뷰포트 카메라 시선(방향)에 직교하는 2D 배경 그리드를 생성
        # 평면도(Top) 및 시네마틱 뷰 - XY 평면 (Z축 방향에서 바라봄) (크기 320m x 320m, 해상도 320)
        grid_xy = pv.Plane(center=(0,0,0), direction=(0,0,1), i_size=320, j_size=320, i_resolution=320, j_resolution=320)
        # 정면도(Front) - YZ 평면 (X축 방향에서 바라봄)
        grid_yz = pv.Plane(center=(0,0,0), direction=(1,0,0), i_size=320, j_size=320, i_resolution=320, j_resolution=320)
        # 측면도(Right/Left) - XZ 평면 (Y축 방향에서 바라봄)
        grid_xz = pv.Plane(center=(0,0,0), direction=(0,1,0), i_size=320, j_size=320, i_resolution=320, j_resolution=320)
        
        # 각 뷰포트에 백그라운드용 그리드를 투명한 와이어프레임으로 추가
        actors = []
        actors.append(self.plotter_single.add_mesh(grid_xy, style='wireframe', color=grid_color, line_width=1, opacity=0.4, name='bg_grid', reset_camera=False, pickable=False))
        actors.append(self.plotter_tl.add_mesh(grid_xy, style='wireframe', color=grid_color, line_width=1, opacity=0.4, name='bg_grid', reset_camera=False, pickable=False)) # Top
        actors.append(self.plotter_bl.add_mesh(grid_yz, style='wireframe', color=grid_color, line_width=1, opacity=0.4, name='bg_grid', reset_camera=False, pickable=False)) # Front
        actors.append(self.plotter_tr.add_mesh(grid_xz, style='wireframe', color=grid_color, line_width=1, opacity=0.4, name='bg_grid', reset_camera=False, pickable=False)) # Right
        actors.append(self.plotter_br.add_mesh(grid_xz, style='wireframe', color=grid_color, line_width=1, opacity=0.4, name='bg_grid', reset_camera=False, pickable=False)) # Left

        # 십자 축(Axis) 라인 시각화 (선 두께를 굵게 하여 중심 축 강조)
        axis_length = 160 # 그리드 반경에 맞춰 연장
        line_x = pv.Line((-axis_length, 0, 0), (axis_length, 0, 0))
        line_y = pv.Line((0, -axis_length, 0), (0, axis_length, 0))
        line_z = pv.Line((0, 0, -axis_length), (0, 0, axis_length))
        
        axis_color = "#333333" 
        axis_width_ortho = 3 # 직교 뷰용 두께
        axis_width_cinematic = 1 # 시네마틱 뷰용 두께 (3배 축소)
        
        # 시네마틱 뷰 (단일 뷰포트) - 두께 1 (Z축 수직선 제거)
        actors.append(self.plotter_single.add_mesh(line_x, color=axis_color, line_width=axis_width_cinematic, name='axis_x', reset_camera=False, pickable=False))
        actors.append(self.plotter_single.add_mesh(line_y, color=axis_color, line_width=axis_width_cinematic, name='axis_y', reset_camera=False, pickable=False))

        # 직교 뷰 4분할 (Top, Front, Right, Left) - 두께 3
        # 평면도(Top)에는 X, Y 십자축 추가
        actors.append(self.plotter_tl.add_mesh(line_x, color=axis_color, line_width=axis_width_ortho, name='axis_x', reset_camera=False, pickable=False))
        actors.append(self.plotter_tl.add_mesh(line_y, color=axis_color, line_width=axis_width_ortho, name='axis_y', reset_camera=False, pickable=False))
        
        # 측면도(Front, Right, Left)에는 수평 바닥선(해당 뷰의 가로축)만 추가하고 수직축(Z)은 제외하여 깔끔한 지평선 유지
        actors.append(self.plotter_bl.add_mesh(line_y, color=axis_color, line_width=axis_width_ortho, name='axis_y', reset_camera=False, pickable=False)) # Front (Y축이 좌우)
        actors.append(self.plotter_tr.add_mesh(line_x, color=axis_color, line_width=axis_width_ortho, name='axis_x', reset_camera=False, pickable=False)) # Right (X축이 좌우)
        actors.append(self.plotter_br.add_mesh(line_x, color=axis_color, line_width=axis_width_ortho, name='axis_x', reset_camera=False, pickable=False)) # Left (X축이 좌우)

        # 배경 요소들이 카메라 Zoom(bounds)에 영향을 주지 않도록 설정 (오직 모델에만 줌인)
        for act in actors:
            if act:
                act.SetUseBounds(False)
        
        # 뷰포트 명칭(Text) 좌측 상단(upper_left)에 추가하여 축(Axis) 위젯과 겹치지 않게 함
        text_color = "#222222" # 어두운 회색 텍스트
        self.plotter_single.add_text("Cinematic", position=(0.02, 0.95), font_size=12, viewport=True, color=text_color, name="view_name")
        self.plotter_single.add_text("X: 0.000m, Y: 0.000m, Z: 0.000m", position=(0.02, 0.92), font_size=5, viewport=True, color=text_color, name="dimensions")
        
        self.plotter_tl.add_text("Top", position=(0.02, 0.95), font_size=12, viewport=True, color=text_color, name="view_name")
        self.plotter_tr.add_text("Right", position=(0.02, 0.95), font_size=12, viewport=True, color=text_color, name="view_name")
        self.plotter_bl.add_text("Front", position=(0.02, 0.95), font_size=12, viewport=True, color=text_color, name="view_name")
        self.plotter_br.add_text("Left", position=(0.02, 0.95), font_size=12, viewport=True, color=text_color, name="view_name")

    def setup_quad_cameras(self):
        """사분할 뷰의 카메라 각도를 직교 투영 및 정규 시점으로 고정하고 회전(Trackball) 비활성화"""
        # (0, 0) 평면도 (Top View) - 2사분면
        self.plotter_tl.view_xy()
        self.plotter_tl.enable_parallel_projection()
        self.plotter_tl.enable_2d_style()

        # (1, 0) 정면도 (Front View) - 3사분면
        # 차량 앞면이 -X를 향함. 정면에서 보려면 -X 위치에서 +X 방향(원점)을 바라봐야 함.
        self.plotter_bl.view_vector((1, 0, 0), viewup=(0, 0, 1))
        self.plotter_bl.enable_parallel_projection()
        self.plotter_bl.enable_2d_style()

        # (0, 1) 우측면도 (Right View) - 1사분면
        # 차량 우측면은 +Y. 우측면에서 보려면 +Y 위치에서 -Y 방향(원점)을 바라봐야 함.
        self.plotter_tr.view_vector((0, -1, 0), viewup=(0, 0, 1))
        self.plotter_tr.enable_parallel_projection()
        self.plotter_tr.enable_2d_style()

        # (1, 1) 좌측면도 (Left View) - 4사분면
        # 차량 좌측면은 -Y. 좌측면에서 보려면 -Y 위치에서 +Y 방향(원점)을 바라봐야 함.
        self.plotter_br.view_vector((0, 1, 0), viewup=(0, 0, 1))
        self.plotter_br.enable_parallel_projection()
        self.plotter_br.enable_2d_style()

    def set_cinematic_camera(self, mode="aerial"):
        """시네마틱 뷰의 카메라 시점(Eye) 및 주시점(Focal Point) 설정"""
        self.plotter_single.disable_parallel_projection() # 원근 투영
        cam = self.plotter_single.camera

        if mode == "aerial":  # 조감도 (상공 45도 부감 뷰)
            cam.position = (25.0, 25.0, 20.0)
            cam.focal_point = (0.0, 0.0, 2.0)
            cam.viewup = (0.0, 0.0, 1.0)
        elif mode == "perspective":  # 투시도 (인간 눈높이 1.6m 아이 레벨)
            cam.position = (0.0, -15.0, 1.6)
            cam.focal_point = (0.0, 5.0, 1.6)
            cam.viewup = (0.0, 0.0, 1.0)
        self.plotter_single.render()

    def clear_and_setup(self):
        """플로터를 초기화(clear)하고 배경, 그리드, 텍스트를 다시 세팅"""
        for plotter in self.plotters:
            plotter.clear()
        
        # 다시 그리드와 축, 배경색 등 미적 요소 렌더링
        self.setup_viewport_aesthetics()

    def set_display_mode(self, show_edges):
        """씬 내 모든 모델의 엣지(와이어프레임) 표시 여부를 토글합니다."""
        self.show_edges = show_edges
        ignore_names = ['bg_grid', 'axis_x', 'axis_y', 'view_name', 'Cinematic', 'dimensions', 'picked_cells', 'my_picked_cells']
        for plotter in self.plotters:
            for name, actor in plotter.actors.items():
                if name not in ignore_names and hasattr(actor, 'prop'):
                    actor.prop.show_edges = show_edges
            plotter.render()

    def add_mesh_to_all(self, mesh, name=None, color=None):
        """로드된 모델을 모든 플로터에 등록"""
        show_edges = getattr(self, 'show_edges', True)
        for plotter in self.plotters:
            plotter.add_mesh(mesh, name=name, color=color, show_edges=show_edges)
            plotter.show_axes() # 축 방향 표시기
            plotter.reset_camera()
            
    def update_mesh(self, name, new_mesh, color):
        """기존 메시를 갱신하거나 새로 덮어씌움 (폴리곤 분할용)"""
        show_edges = getattr(self, 'show_edges', True)
        for plotter in self.plotters:
            if new_mesh is None or new_mesh.n_cells == 0:
                plotter.remove_actor(name)
            else:
                plotter.add_mesh(new_mesh, name=name, color=color, show_edges=show_edges, reset_camera=False)

    def set_hdri_environment(self, hdri_path):
        """단일 시네마틱 뷰포트에 HDRI 환경 맵을 적용합니다."""
        import pyvista as pv
        try:
            cubemap = pv.read_texture(hdri_path)
            self.plotter_single.set_environment_texture(cubemap)
            # PBR 렌더링에 적합하도록 조명 설정
            self.plotter_single.enable_image_based_lighting()
            self.plotter_single.render()
            return True
        except Exception as e:
            print(f"HDRI 로드 오류: {e}")
            return False

    def closeEvent(self, event):
        for plotter in getattr(self, 'plotters', []):
            try:
                plotter.close()
            except Exception:
                pass
        super().closeEvent(event)
