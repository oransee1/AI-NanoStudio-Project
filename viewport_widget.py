import os
import time
import numpy as np
import pyvista as pv
from PySide6.QtWidgets import QWidget, QVBoxLayout, QStackedWidget, QGridLayout, QPushButton, QHBoxLayout, QLabel
from PySide6.QtCore import QObject, QEvent
from pyvistaqt import QtInteractor
from event_logger import log_action

class SmoothInteractionFilter(QObject):
    """
    고주파 마우스 드래그(1000Hz 게이밍 마우스 등) 시 언리얼 엔진 수준의 60FPS 프레임 스로틀링을 적용하여
    OpenGL 파이프라인 과부하 및 Windows 응답 없음(Hang/0xCFFFFFFF)을 원천 차단합니다.
    """
    def __init__(self, target_widget, min_interval_ms=16):
        super().__init__(target_widget)
        self.target = target_widget
        self.min_interval = min_interval_ms / 1000.0
        self.last_time = 0.0
        self.is_dragging = False

    def eventFilter(self, obj, event):
        t = event.type()
        if t == QEvent.MouseButtonPress:
            self.is_dragging = True
            self.last_time = time.perf_counter()
        elif t == QEvent.MouseButtonRelease:
            self.is_dragging = False
        elif t == QEvent.MouseMove and self.is_dragging:
            now = time.perf_counter()
            if now - self.last_time < self.min_interval:
                return True
            self.last_time = now
        return super().eventFilter(obj, event)

def read_hdri_texture(file_path, max_dim=1024):
    """
    .exr, .hdr, .png, .jpg 등 다양한 포맷의 환경맵 텍스처를 PyVista Texture로 로드합니다.
    - Windows 한글 경로 및 32bit Float EXR/HDR 톤맵핑/감마 보정을 완벽 지원합니다.
    - max_dim=1024 및 하드웨어 밉맵(mipmap=True), 선형 보간(interpolate=True)을
      적용하여 PBR IBL 셰이더 샘플링 시 GPU 멈춤 현상과 TDR을 차단합니다.
    """
    if not file_path or not os.path.exists(file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    ext = os.path.splitext(file_path)[1].lower()

    if ext in ['.exr', '.hdr']:
        img_np = None

        # 1. OpenEXR (.exr 포맷 전용 고속 고화질 로더)
        if ext == '.exr':
            try:
                import OpenEXR
                exr_file = OpenEXR.File(file_path)
                channels = exr_file.channels()
                if channels:
                    if 'RGB' in channels:
                        img_np = channels['RGB'].pixels[..., :3].astype(np.float32)
                    elif 'RGBA' in channels:
                        img_np = channels['RGBA'].pixels[..., :3].astype(np.float32)
                    elif 'R' in channels and 'G' in channels and 'B' in channels:
                        r = channels['R'].pixels
                        g = channels['G'].pixels
                        b = channels['B'].pixels
                        img_np = np.stack([r, g, b], axis=-1).astype(np.float32)
                    else:
                        first_key = list(channels.keys())[0]
                        img_np = channels[first_key].pixels[..., :3].astype(np.float32)
            except Exception as e:
                pass

        # 2. OpenCV fallback
        if img_np is None:
            try:
                import cv2
                buf = np.fromfile(file_path, dtype=np.uint8)
                img_np = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
                if img_np is not None and img_np.ndim == 3 and img_np.shape[2] >= 3:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB).astype(np.float32)
            except Exception:
                pass

        # 3. imageio fallback
        if img_np is None:
            try:
                import imageio.v2 as imageio
                with open(file_path, 'rb') as f:
                    raw_bytes = f.read()
                fmt = 'EXR-FI' if ext == '.exr' else 'HDR-FI'
                img_np = imageio.imread(raw_bytes, format=fmt).astype(np.float32)
            except Exception:
                pass

        if img_np is not None:
            h, w = img_np.shape[:2]

            # 스마트 다운샘플링 (VRAM/TDR 최적화)
            if max_dim and (w > max_dim or h > max_dim):
                step = max(int(np.ceil(w / max_dim)), int(np.ceil(h / max_dim)))
                img_np = img_np[::step, ::step, :3]
                h, w = img_np.shape[:2]
            elif img_np.ndim == 3 and img_np.shape[2] > 3:
                img_np = img_np[..., :3]

            # HDR 톤 맵핑 및 감마 보정 (0..1 범위 정규화하여 VTK PBR 반사 동심원/격자 아티팩트 차단)
            max_val = np.percentile(img_np, 99.5)
            if max_val > 1.0:
                img_norm = np.clip(img_np / max_val, 0.0, 1.0)
            else:
                img_norm = np.clip(img_np, 0.0, 1.0)
            img_norm = np.power(img_norm, 1.0 / 2.2)

            import vtk
            from vtkmodules.util import numpy_support

            img_data = vtk.vtkImageData()
            img_data.SetDimensions(w, h, 1)
            flipped_rgb = np.ascontiguousarray(np.flipud(img_norm), dtype=np.float32).reshape(-1, 3)

            vtk_arr = numpy_support.numpy_to_vtk(flipped_rgb, deep=True, array_type=vtk.VTK_FLOAT)
            vtk_arr.SetNumberOfComponents(3)
            vtk_arr.SetName("ImageScalars")
            img_data.GetPointData().SetScalars(vtk_arr)

            tex = pv.Texture(img_data)
            tex.mipmap = True
            tex.interpolate = True
            return tex

    # PNG, JPG 등 일반 8bit 텍스처 (한글 경로 대응 PIL 사용)
    try:
        import PIL.Image
        with open(file_path, 'rb') as f:
            pil_img = PIL.Image.open(f).convert('RGB')
        tex = pv.Texture(pil_img)
        tex.mipmap = True
        tex.interpolate = True
        return tex
    except Exception:
        tex = pv.read_texture(file_path)
        try:
            tex.mipmap = True
            tex.interpolate = True
        except Exception:
            pass
        return tex

class ViewportWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.layout = QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        
        # 뷰 모드 전환 버튼
        self.btn_layout = QHBoxLayout()
        self.btn_single = QPushButton("시네마틱 뷰 (단일)")
        self.btn_quad = QPushButton("직교 뷰 (2x2)")
        self.btn_single.setStyleSheet("font-weight: bold; padding: 4px 8px;")
        self.btn_quad.setStyleSheet("font-weight: bold; padding: 4px 8px;")
        self.btn_layout.addWidget(self.btn_single)
        self.btn_layout.addWidget(self.btn_quad)
        
        # 4가지 디스플레이 모드 버튼 묶음
        self.btn_layout.addSpacing(15)
        lbl_mode = QLabel("🎨 디스플레이 모드:")
        lbl_mode.setStyleSheet("font-weight: bold; font-size: 12px; margin-left: 5px; color: #222222;")
        self.btn_layout.addWidget(lbl_mode)
        
        self.btn_wireframe = QPushButton("📐 와이어프레임")
        self.btn_solid = QPushButton("🔳 솔리드")
        self.btn_mat_preview = QPushButton("🔮 머티리얼 프리뷰")
        self.btn_rendered = QPushButton("✨ 렌더드")
        
        self.mode_buttons = {
            "wireframe": self.btn_wireframe,
            "solid": self.btn_solid,
            "material_preview": self.btn_mat_preview,
            "rendered": self.btn_rendered
        }
        
        for mode_key, btn in self.mode_buttons.items():
            btn.setCheckable(True)
            self.btn_layout.addWidget(btn)
            btn.clicked.connect(lambda checked=False, k=mode_key: self.set_display_render_mode(k))
            
        self.btn_layout.addStretch()
        self.layout.addLayout(self.btn_layout)
        
        # 스택 위젯 생성
        self.stacked_widget = QStackedWidget()
        self.layout.addWidget(self.stacked_widget)
        
        # 1. 단일 시네마틱 뷰포트
        self.single_view_widget = QWidget()
        self.single_layout = QVBoxLayout(self.single_view_widget)
        self.single_layout.setContentsMargins(0, 0, 0, 0)
        self.single_layout.setSpacing(0)
        self.plotter_single = QtInteractor(self.single_view_widget)
        self.single_layout.addWidget(self.plotter_single.interactor, stretch=1)
        self.smooth_interaction_filter = SmoothInteractionFilter(self.plotter_single.interactor)
        self.plotter_single.interactor.installEventFilter(self.smooth_interaction_filter)
        
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
        self.btn_single.clicked.connect(lambda: (self.stacked_widget.setCurrentIndex(0), log_action("VIEW_LAYOUT_SWITCH", "시네마틱 뷰 (단일) 전환")))
        self.btn_quad.clicked.connect(lambda: (self.stacked_widget.setCurrentIndex(1), self.sync_quad_view(), log_action("VIEW_LAYOUT_SWITCH", "직교 뷰 (2x2) 전환")))
        
        self.plotters = [
            self.plotter_single,
            self.plotter_tl, self.plotter_tr, self.plotter_bl, self.plotter_br
        ]

        self.setup_quad_cameras()
        self.setup_viewport_aesthetics()
        self.setup_pbr_lighting()
        
        # 마우스 드래그/회전 시 30fps 이상 인터랙티브 렌더링으로 멈춤 현실적 완화
        for p in self.plotters:
            try:
                p.interactor.SetDesiredUpdateRate(30.0)
                p.interactor.SetStillUpdateRate(0.0001)
            except Exception:
                pass

        self.current_render_mode = "material_preview"
        self.update_mode_button_styles()

    def render_active(self):
        """현재 화면에 노출 중인 뷰포트만 최적화 렌더링하여 마우스 회전 Lag 및 메모리 멈춤 완벽 방지"""
        try:
            curr_idx = self.stacked_widget.currentIndex()
            if curr_idx == 0:
                self.plotter_single.render()
            else:
                for p in [self.plotter_tl, self.plotter_tr, self.plotter_bl, self.plotter_br]:
                    p.render()
        except Exception as e:
            print(f"렌더링 최적화 예외: {e}")

    def sync_quad_view(self):
        """단일 뷰에서 로드된 메시들을 직교 뷰(2x2) 플로터로 온디맨드 동기화"""
        try:
            main_win = self.window()
            if hasattr(main_win, 'part_meshes') and main_win.part_meshes:
                show_edges = getattr(self, 'show_edges', True)
                quad_plotters = [self.plotter_tl, self.plotter_tr, self.plotter_bl, self.plotter_br]
                for name, mesh in main_win.part_meshes.items():
                    color = getattr(main_win, 'part_colors', {}).get(name, [0.8, 0.8, 0.8])
                    for p in quad_plotters:
                        if name not in p.actors:
                            try:
                                actor = p.add_mesh(mesh, name=name, color=color, show_edges=show_edges, render=False)
                                if hasattr(actor, 'SetBackfaceProperty') and hasattr(actor, 'GetProperty'):
                                    actor.SetBackfaceProperty(actor.GetProperty())
                            except Exception:
                                pass
                self.setup_quad_cameras()
                self.render_active()
        except Exception as e:
            print(f"Quad View 동기화 예외: {e}")

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
        """플로터를 초기화(clear)하고 배경, 그리드, 텍스트 및 PBR 라이팅을 다시 세팅"""
        for plotter in self.plotters:
            plotter.clear()
        
        # 다시 그리드와 축, 배경색 등 미적 요소 렌더링
        self.setup_viewport_aesthetics()
        self.setup_pbr_lighting()

    def set_display_mode(self, show_edges):
        """씬 내 모든 모델의 엣지(와이어프레임) 표시 여부를 토글합니다."""
        self.show_edges = show_edges
        ignore_names = ['bg_grid', 'axis_x', 'axis_y', 'view_name', 'Cinematic', 'dimensions', 'picked_cells', 'my_picked_cells']
        for plotter in self.plotters:
            for name, actor in plotter.actors.items():
                if name not in ignore_names and hasattr(actor, 'prop'):
                    actor.prop.show_edges = show_edges
            plotter.render()

    def update_mode_button_styles(self):
        """현재 선택된 렌더 디스플레이 모드 버튼의 비주얼 스타일 하이라이트"""
        active_style = "background-color: #2563eb; color: white; font-weight: bold; padding: 4px 10px; border-radius: 4px;"
        inactive_style = "background-color: #333333; color: #E0E0E0; padding: 4px 10px; border-radius: 4px;"
        
        for k, btn in getattr(self, 'mode_buttons', {}).items():
            if k == getattr(self, 'current_render_mode', 'material_preview'):
                btn.setStyleSheet(active_style)
                btn.setChecked(True)
            else:
                btn.setStyleSheet(inactive_style)
                btn.setChecked(False)

    def set_display_render_mode(self, mode_key):
        """
        4가지 3D 디스플레이 모드 실시간 전환:
        1. 'wireframe': 와이어프레임 모드 (폴리곤 뼈대 선만 표출)
        2. 'solid': 솔리드 모드 (단순 표면 + 엣지 음영, PBR/HDRI 제외)
        3. 'material_preview': 머티리얼 프리뷰 모드 (PBR 파라미터 + HDRI IBL 광원 표출)
        4. 'rendered': 렌더드 모드 (PBR + HDRI 스카이박스 + 실시간 그림자 + MSAA 안티앨리어싱)
        """
        self.current_render_mode = mode_key
        self.update_mode_button_styles()
        log_action("DISPLAY_MODE_CHANGED", f"3D 디스플레이 모드 전환: {mode_key}")
        
        ignore_names = ['bg_grid', 'axis_x', 'axis_y', 'view_name', 'Cinematic', 'dimensions', 'picked_cells', 'my_picked_cells']

        for plotter in self.plotters:
            # 1. 플로터 환경 효과 (그림자/안티앨리어싱/IBL)
            if mode_key == "rendered":
                try:
                    plotter.enable_shadows()
                except Exception:
                    pass
                try:
                    plotter.enable_anti_aliasing("msaa")
                except Exception:
                    pass
                if hasattr(plotter, 'enable_image_based_lighting'):
                    try:
                        plotter.enable_image_based_lighting()
                    except Exception:
                        pass
            elif mode_key == "material_preview":
                try:
                    plotter.disable_shadows()
                except Exception:
                    pass
                if hasattr(plotter, 'enable_image_based_lighting'):
                    try:
                        plotter.enable_image_based_lighting()
                    except Exception:
                        pass
            else: # wireframe, solid
                try:
                    plotter.disable_shadows()
                except Exception:
                    pass
                if hasattr(plotter, 'disable_image_based_lighting'):
                    try:
                        plotter.disable_image_based_lighting()
                    except Exception:
                        pass

            # 2. 객체(Actor) 렌더링 스타일 및 쉐이딩 인터폴레이션 설정
            for actor_name, actor in plotter.actors.items():
                if actor_name in ignore_names:
                    continue
                if hasattr(actor, 'GetProperty'):
                    prop = actor.GetProperty()
                    if hasattr(prop, 'SetBackfaceCulling'):
                        try:
                            prop.SetBackfaceCulling(False)
                        except Exception:
                            pass
                    if hasattr(prop, 'SetFrontfaceCulling'):
                        try:
                            prop.SetFrontfaceCulling(False)
                        except Exception:
                            pass

                    if mode_key == "wireframe":
                        if hasattr(prop, 'SetRepresentationToWireframe'):
                            prop.SetRepresentationToWireframe()
                        if hasattr(prop, 'SetInterpolationToFlat'):
                            prop.SetInterpolationToFlat()
                    elif mode_key == "solid":
                        if hasattr(prop, 'SetRepresentationToSurface'):
                            prop.SetRepresentationToSurface()
                        if hasattr(prop, 'SetInterpolationToGouraud'):
                            prop.SetInterpolationToGouraud()
                        if hasattr(prop, 'SetEdgeVisibility'):
                            prop.SetEdgeVisibility(True)
                    elif mode_key in ["material_preview", "rendered"]:
                        if hasattr(prop, 'SetRepresentationToSurface'):
                            prop.SetRepresentationToSurface()
                        if hasattr(prop, 'SetInterpolationToPBR'):
                            try:
                                prop.SetInterpolationToPBR()
                            except Exception:
                                pass
                        elif hasattr(prop, 'SetInterpolationToPhysicallyBased'):
                            try:
                                prop.SetInterpolationToPhysicallyBased()
                            except Exception:
                                pass
                        if hasattr(prop, 'SetEdgeVisibility'):
                            prop.SetEdgeVisibility(getattr(self, 'show_edges', False))
                if hasattr(actor, 'SetBackfaceProperty') and hasattr(actor, 'GetProperty'):
                    try:
                        actor.SetBackfaceProperty(actor.GetProperty())
                    except Exception:
                        pass
            plotter.render()
        self.render_active()

    def setup_pbr_lighting(self, plotter=None, bounds=None):
        """Cinema View 및 메인 뷰포트 PBR 반사광/입체감 검수용 directional(방향성) 3점 조명 세팅 (모델 바운딩박스 스케일 동적 반영)"""
        target_plotters = [plotter] if plotter else self.plotters
        for p in target_plotters:
            try:
                p.renderer.RemoveAllLights()
                if hasattr(p.renderer, 'SetTwoSidedLighting'):
                    p.renderer.SetTwoSidedLighting(True)
                
                # 모델 전체의 바운딩 박스 및 중심점/크기 계산 (구체 프리뷰 조명과 100% 동일한 비율 및 입체감 구현)
                target_bounds = bounds if bounds is not None else p.bounds
                cx = (target_bounds[0] + target_bounds[1]) / 2.0
                cy = (target_bounds[2] + target_bounds[3]) / 2.0
                cz = (target_bounds[4] + target_bounds[5]) / 2.0
                diag = max(1.0, ((target_bounds[1]-target_bounds[0])**2 + (target_bounds[3]-target_bounds[2])**2 + (target_bounds[5]-target_bounds[4])**2)**0.5)
                focal = (cx, cy, cz)
                
                # Directional Key Light (주 광원: 우측 사선 상단 정면)
                key = pv.Light(position=(cx + diag * 0.8, cy - diag * 0.8, cz + diag * 1.0), focal_point=focal, color='white', intensity=1.5, light_type='scenelight')
                
                # Directional Fill Light (보조 광원: 좌측 상단 부드러운 광원)
                fill = pv.Light(position=(cx - diag * 0.8, cy - diag * 0.8, cz + diag * 0.7), focal_point=focal, color=(0.85, 0.9, 1.0), intensity=0.8, light_type='scenelight')
                
                # Directional Rim Light (역광/후면 윤곽광)
                rim = pv.Light(position=(cx, cy + diag * 0.8, cz - diag * 0.4), focal_point=focal, color=(1.0, 0.95, 0.85), intensity=1.0, light_type='scenelight')
                
                # Headlight (카메라 시선 방향 보조 조명)
                headlight = pv.Light(light_type='headlight', color='white', intensity=0.6)
                
                p.add_light(key)
                p.add_light(fill)
                p.add_light(rim)
                p.add_light(headlight)
            except Exception as e:
                print(f"조명 세팅 오류: {e}")

    def reset_camera(self, bounds=None):
        """모든 뷰포트(시네마틱 및 4분할 직교 뷰)의 카메라를 현재 로드된 전체 메시에 맞게 리셋"""
        for plotter in self.plotters:
            try:
                if bounds is not None:
                    plotter.reset_camera(bounds=bounds)
                else:
                    plotter.reset_camera()
                if hasattr(plotter, 'renderer') and hasattr(plotter.renderer, 'ResetCameraClippingRange'):
                    plotter.renderer.ResetCameraClippingRange()
            except Exception:
                pass
        self.render_active()

    def add_mesh_to_all(self, mesh, name=None, color=None, reset_camera=False, update_mode=False, show_edges=False):
        """로드된 모델을 활성 플로터에 PBR 렌더링 속성으로 등록 (초고속 배치 최적화)"""
        if not getattr(self, '_pbr_light_setup', False):
            self.setup_pbr_lighting()
            self._pbr_light_setup = True

        # 단일 뷰 상태인 경우 plotter_single만 우선 등록하여 배치 연산량 80% 감축
        if hasattr(self, 'stacked_widget') and self.stacked_widget.currentIndex() == 0:
            target_plotters = [self.plotter_single]
        else:
            target_plotters = self.plotters

        for plotter in target_plotters:
            try:
                actor = plotter.add_mesh(mesh, name=name, color=color, show_edges=show_edges, reset_camera=reset_camera, render=False)
            except Exception:
                actor = plotter.add_mesh(mesh, name=name, color=color, show_edges=show_edges, reset_camera=reset_camera, render=False)
            if hasattr(actor, 'SetBackfaceProperty') and hasattr(actor, 'GetProperty'):
                try:
                    actor.SetBackfaceProperty(actor.GetProperty())
                except Exception:
                    pass
            if reset_camera:
                plotter.reset_camera()
            
        if update_mode:
            self.set_display_render_mode(getattr(self, 'current_render_mode', 'material_preview'))
            
    def update_mesh(self, name, new_mesh, color):
        """기존 메시를 갱신하거나 새로 덮어씌움 (폴리곤 분할용)"""
        show_edges = getattr(self, 'show_edges', True)
        for plotter in self.plotters:
            if new_mesh is None or new_mesh.n_cells == 0:
                plotter.remove_actor(name)
            else:
                try:
                    actor = plotter.add_mesh(new_mesh, name=name, color=color, pbr=True, roughness=0.5, metallic=0.0, show_edges=show_edges, reset_camera=False)
                except Exception:
                    actor = plotter.add_mesh(new_mesh, name=name, color=color, show_edges=show_edges, reset_camera=False)
                if hasattr(actor, 'GetProperty') and actor.GetProperty():
                    try:
                        actor.GetProperty().SetBackfaceCulling(False)
                        actor.GetProperty().SetFrontfaceCulling(False)
                    except Exception:
                        pass
                if hasattr(actor, 'SetBackfaceProperty') and hasattr(actor, 'GetProperty'):
                    try:
                        actor.SetBackfaceProperty(actor.GetProperty())
                    except Exception:
                        pass
        self.render_active()

    def update_mesh_pbr(self, name, color, roughness=0.5, metallic=0.0, normal_scale=1.0, bright=1.0, reflection=1.0, refraction=1.5,
                        emissive=0.0, coat_strength=0.0, coat_roughness=0.0, anisotropy=0.0, occlusion=1.0,
                        use_hdri=True, hdri_ratio=1.0,
                        use_roughness=True, use_metallic=True, use_normal=True, use_bright=True, use_reflection=True, use_refraction=True,
                        use_emissive=False, use_coat_strength=False, use_coat_roughness=False, use_anisotropy=False, use_occlusion=False,
                        render=True):
        """체크박스가 체크된 PBR 파라미터 속성만 실시간 반영 및 HDRI 적용 시 원색 어두워짐 보정, HDRI 적용 비율(30%~100%) 반영"""
        bright_val = bright if use_bright else 1.0
        adj_color = [
            min(1.0, max(0.0, color[0] * bright_val)),
            min(1.0, max(0.0, color[1] * bright_val)),
            min(1.0, max(0.0, color[2] * bright_val))
        ] if color else None
        
        rough_val = roughness if use_roughness else 0.5
        metal_val = metallic if use_metallic else 0.0
        normal_val = normal_scale if use_normal else 0.0
        reflect_val = reflection if use_reflection else 0.0
        refract_val = refraction if use_refraction else 1.5
        emissive_val = emissive if use_emissive else 0.0
        coat_val = coat_strength if use_coat_strength else 0.0
        coat_rough_val = coat_roughness if use_coat_roughness else 0.0
        aniso_val = anisotropy if use_anisotropy else 0.0
        occ_val = occlusion if use_occlusion else 1.0

        if not getattr(self, '_pbr_light_setup', False):
            self.setup_pbr_lighting()
            self._pbr_light_setup = True

        for plotter in self.plotters:
            try:
                actors_list = list(plotter.actors.items())
            except Exception:
                continue

            for actor_name, actor in actors_list:
                if actor_name == name and actor is not None and hasattr(actor, 'GetProperty'):
                    try:
                        prop = actor.GetProperty()
                        if not prop:
                            continue
                        
                        if hasattr(prop, 'SetBackfaceCulling'):
                            try:
                                prop.SetBackfaceCulling(False)
                            except Exception:
                                pass
                        if hasattr(prop, 'SetFrontfaceCulling'):
                            try:
                                prop.SetFrontfaceCulling(False)
                            except Exception:
                                pass
                        
                        # 스케치업/CAD 메쉬의 반전된 면(Backface)에도 앞면과 100% 동일한 PBR 재질 및 조명 반사를 적용
                        if hasattr(actor, 'SetBackfaceProperty'):
                            try:
                                actor.SetBackfaceProperty(prop)
                            except Exception:
                                pass

                        if hasattr(prop, 'SetInterpolationToPBR'):
                            try:
                                prop.SetInterpolationToPBR()
                            except Exception:
                                pass
                        elif hasattr(prop, 'SetInterpolationToPhysicallyBased'):
                            try:
                                prop.SetInterpolationToPhysicallyBased()
                            except Exception:
                                pass
                                
                        if hasattr(prop, 'SetColor') and adj_color:
                            try:
                                prop.SetColor(adj_color[0], adj_color[1], adj_color[2])
                            except Exception:
                                pass
                                
                        # 메쉬의 기본 스칼라(Scalar) 색상 오버라이드를 끄고 PBR 머티리얼 원색 및 IBL 반사 100% 적용
                        if hasattr(actor, 'GetMapper') and actor.GetMapper():
                            try:
                                actor.GetMapper().SetScalarVisibility(False)
                            except Exception:
                                pass

                        # 구체 프리뷰와 동일한 매끄러운 반사 표현을 위해 PBR 모드에서는 와이어프레임 엣지 표출 비활성화
                        if hasattr(prop, 'SetEdgeVisibility'):
                            try:
                                prop.SetEdgeVisibility(False)
                            except Exception:
                                pass

                        # HDRI 적용 시 원색 색상 유지 및 HDRI 비율 반영
                        if use_hdri:
                            try:
                                if hasattr(prop, 'SetAmbient'):
                                    prop.SetAmbient(0.35 * hdri_ratio)
                                if hasattr(prop, 'SetDiffuse'):
                                    prop.SetDiffuse(0.85 + 0.15 * (1.0 - hdri_ratio))
                            except Exception:
                                pass
                        else:
                            try:
                                if hasattr(prop, 'SetAmbient'):
                                    prop.SetAmbient(0.0)
                                if hasattr(prop, 'SetDiffuse'):
                                    prop.SetDiffuse(1.0)
                            except Exception:
                                pass

                        if hasattr(prop, 'SetRoughness'):
                            try:
                                prop.SetRoughness(rough_val)
                            except Exception:
                                pass
                        if hasattr(prop, 'SetMetallic'):
                            try:
                                prop.SetMetallic(metal_val * hdri_ratio if use_hdri else metal_val)
                            except Exception:
                                pass
                        if hasattr(prop, 'SetNormalScale'):
                            try:
                                prop.SetNormalScale(normal_val)
                            except Exception:
                                pass
                        if hasattr(prop, 'SetSpecular'):
                            try:
                                prop.SetSpecular(min(1.0, reflect_val * hdri_ratio if use_hdri else reflect_val))
                            except Exception:
                                pass
                        if hasattr(prop, 'SetSpecularPower'):
                            try:
                                prop.SetSpecularPower(max(1.0, reflect_val * 15.0))
                            except Exception:
                                pass
                        
                        # 신규 PBR 속성 (Emissive, Clearcoat, Anisotropy, Occlusion) 적용
                        if hasattr(prop, 'SetEmissiveFactor'):
                            try:
                                prop.SetEmissiveFactor(emissive_val, emissive_val, emissive_val)
                            except Exception:
                                pass
                        if hasattr(prop, 'SetCoatStrength'):
                            try:
                                prop.SetCoatStrength(coat_val)
                            except Exception:
                                pass
                        if hasattr(prop, 'SetCoatRoughness'):
                            try:
                                prop.SetCoatRoughness(coat_rough_val)
                            except Exception:
                                pass
                        if hasattr(prop, 'SetAnisotropy'):
                            try:
                                prop.SetAnisotropy(aniso_val)
                            except Exception:
                                pass
                        if hasattr(prop, 'SetOcclusionStrength'):
                            try:
                                prop.SetOcclusionStrength(occ_val)
                            except Exception:
                                pass
                        
                        if use_refraction:
                            if hasattr(prop, 'SetIOR'):
                                try:
                                    prop.SetIOR(refract_val)
                                except Exception:
                                    pass
                            if hasattr(prop, 'SetOpacity'):
                                try:
                                    prop.SetOpacity(0.35)
                                except Exception:
                                    pass
                        else:
                            if hasattr(prop, 'SetOpacity'):
                                try:
                                    prop.SetOpacity(1.0)
                                except Exception:
                                    pass

                        # PBR 속성 및 색상이 모두 설정된 최신 prop을 BackfaceProperty에 100% 동기화 전이
                        if hasattr(actor, 'SetBackfaceProperty'):
                            try:
                                actor.SetBackfaceProperty(prop)
                            except Exception:
                                pass
                    except Exception as e:
                        print(f"Actor PBR 속성 적용 오류 ({name}): {e}")

        if render:
            self.render_active()

    def remove_hdri_environment(self):
        """메인 뷰포트에서 HDRI 환경 맵을 제거하고 표준 조명 상태로 복원합니다."""
        for plotter in [self.plotter_single]:
            try:
                if hasattr(plotter, 'remove_environment_texture'):
                    plotter.remove_environment_texture()
                elif hasattr(plotter, 'renderer') and plotter.renderer is not None and hasattr(plotter.renderer, 'SetEnvironmentTexture'):
                    plotter.renderer.SetEnvironmentTexture(None)
            except Exception as e:
                print(f"HDRI 해제 오류: {e}")
        self._current_hdri_path = None
        self.render_active()
        log_action("HDRI_ENVIRONMENT_REMOVED", "HDRI 환경맵 해제 및 기본 조명 복원")

    def set_hdri_environment(self, hdri_path, force=False):
        """메인 시네마틱 뷰포트에 HDRI 환경 맵 및 IBL 조명을 언리얼 엔진 수준으로 최적화 적용합니다."""
        if not hdri_path or not os.path.exists(hdri_path):
            return False

        if not force and getattr(self, '_current_hdri_path', None) == hdri_path:
            return True

        try:
            cubemap = read_hdri_texture(hdri_path, max_dim=1024)
            
            # 메인 시네마틱 플로터(plotter_single)에 단독 최적화 적용 (독립된 OpenGL Context 충돌 원천 방지)
            p = self.plotter_single
            try:
                if hasattr(p, 'renderer') and p.renderer is not None and hasattr(p.renderer, 'SetEnvironmentTexture'):
                    try:
                        p.renderer.SetEnvironmentTexture(None)
                    except Exception:
                        pass

                p.set_environment_texture(cubemap, is_srgb=False, show_background=False)
                if hasattr(p, 'enable_image_based_lighting'):
                    try:
                        p.enable_image_based_lighting()
                    except Exception:
                        pass
                
                if hasattr(p, 'renderer') and p.renderer is not None:
                    try:
                        p.renderer.SetUseSphericalHarmonics(False)
                        p.renderer.SetAutomaticLightCreation(False)
                    except Exception:
                        pass
                
                # 인터랙터 프레임레이트 안정화 (마우스 드래그 시 30~60fps 보장)
                try:
                    p.interactor.SetDesiredUpdateRate(30.0)
                    p.interactor.SetStillUpdateRate(0.0001)
                except Exception:
                    pass

                self.setup_pbr_lighting(p)
            except Exception as pe:
                print(f"시네마틱 뷰포트 HDRI 로드 오류: {pe}")

            self._current_hdri_path = hdri_path
            self.render_active()
            log_action("HDRI_ENVIRONMENT_SET", f"HDRI 환경맵 적용 성공: {os.path.basename(hdri_path)}")
            return True
        except Exception as e:
            print(f"HDRI 로드 오류: {e}")
            log_action("HDRI_ENVIRONMENT_ERROR", f"HDRI 환경맵 적용 실패: {e}")
            return False

    def closeEvent(self, event):
        if not getattr(self, '_is_closed', False):
            self._is_closed = True
            for plotter in getattr(self, 'plotters', []):
                try:
                    if hasattr(plotter, '_closed') and not plotter._closed:
                        plotter.close()
                except Exception:
                    pass
        super().closeEvent(event)
