import sys
import os
import time
import random
import warnings
import vtk
import cv2
import numpy as np
from dotenv import load_dotenv

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                               QHBoxLayout, QPushButton, QFileDialog, QMessageBox, 
                               QTreeWidget, QTreeWidgetItem, QLabel, QTabWidget,
                               QDialog, QDoubleSpinBox, QSlider, QGroupBox, 
                               QFormLayout, QColorDialog, QLineEdit, QCheckBox, QScrollArea, QSizePolicy, QAbstractItemView,
                               QMenu, QInputDialog, QFrame, QGridLayout, QTextEdit, QProgressDialog)
from PySide6.QtCore import Qt, QEventLoop
from PySide6.QtGui import QColor, QPixmap, QIcon, QCursor
import pyvista as pv
from pyvistaqt import QtInteractor

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
from viewport_widget import ViewportWidget, read_hdri_texture
from skp_loader import load_skp_to_pyvista, load_obj_to_pyvista, export_meshes_to_obj

from camera_manager import CameraManager
from uv_unwrapper import UVUnwrapper
from shadow_baker import ShadowBaker
from genai_worker import GenAIWorker, MaterialAIWorker
from material_binder import MaterialBinder
from event_logger import get_logger, log_action, install_activity_logger
from cinematic_director import CinematicDirector, VeoPromptWorker

try:
    from conti_pdf_builder import ContiPDFBuilder
except ImportError:
    ContiPDFBuilder = None

# .env 파일 로드
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ICON_PATH = os.path.join(BASE_DIR, "icon-1.png")

from PySide6.QtCore import Qt, QObject, QEvent, QRect, QPoint, QSize
from PySide6.QtWidgets import QRubberBand, QFrame, QGridLayout

def cv2_imread_utf8(filepath):
    try:
        data = np.fromfile(filepath, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"이미지 로드 실패 ({filepath}): {e}")
        return None

def cv2_imwrite_utf8(filepath, img):
    try:
        ext = os.path.splitext(filepath)[1]
        if not ext:
            ext = ".png"
        result, buf = cv2.imencode(ext, img)
        if result:
            with open(filepath, 'wb') as f:
                f.write(buf)
            return True
        return False
    except Exception as e:
        print(f"이미지 저장 실패 ({filepath}): {e}")
        return False

def generate_pbr_maps(base_img_path, output_dir=None):
    """
    원본 텍스처 파일(.jpg, .tif, .png, .bmp)을 기반으로 PBR 맵 5개 자동 생성:
    - Metallic.png
    - Normal.png
    - ORM.png (Occlusion=R, Roughness=G, Metallic=B)
    - Displacement.png
    - Roughness.png
    """
    if output_dir is None:
        output_dir = os.path.dirname(base_img_path)
    
    img = cv2_imread_utf8(base_img_path)
    if img is None:
        raise ValueError("텍스처 이미지를 읽을 수 없습니다.")

    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Displacement (Height) Map
    disp = cv2.GaussianBlur(gray, (15, 15), 0)
    disp_norm = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX)
    disp_path = os.path.join(output_dir, "Displacement.png")
    cv2_imwrite_utf8(disp_path, disp_norm)

    # 2. Normal Map (Sobel gradient depth)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    strength = 3.0
    dx = -sobelx * strength
    dy = -sobely * strength
    dz = np.ones_like(gray, dtype=np.float64) * 255.0

    norm_mag = np.sqrt(dx**2 + dy**2 + dz**2)
    nx = (dx / norm_mag) * 0.5 + 0.5
    ny = (dy / norm_mag) * 0.5 + 0.5
    nz = (dz / norm_mag) * 0.5 + 0.5

    normal_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    normal_bgr[:, :, 0] = (nz * 255).astype(np.uint8) # B = Z
    normal_bgr[:, :, 1] = (ny * 255).astype(np.uint8) # G = Y
    normal_bgr[:, :, 2] = (nx * 255).astype(np.uint8) # R = X
    normal_path = os.path.join(output_dir, "Normal.png")
    cv2_imwrite_utf8(normal_path, normal_bgr)

    # 3. Roughness Map
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_abs = np.uint8(np.absolute(lap))
    roughness_gray = cv2.addWeighted(255 - gray, 0.4, lap_abs, 0.6, 0)
    roughness_norm = cv2.normalize(roughness_gray, None, 30, 230, cv2.NORM_MINMAX)
    roughness_path = os.path.join(output_dir, "Roughness.png")
    cv2_imwrite_utf8(roughness_path, roughness_norm)

    # 4. Metallic Map
    _, metallic_binary = cv2.threshold(gray, 180, 255, cv2.THRESH_TOZERO)
    metallic_norm = cv2.normalize(metallic_binary, None, 0, 255, cv2.NORM_MINMAX)
    metallic_path = os.path.join(output_dir, "Metallic.png")
    cv2_imwrite_utf8(metallic_path, metallic_norm)

    # 5. ORM Map (R: Occlusion, G: Roughness, B: Metallic)
    occlusion = cv2.GaussianBlur(255 - disp_norm, (31, 31), 0)
    occlusion_norm = cv2.normalize(occlusion, None, 80, 255, cv2.NORM_MINMAX)
    
    orm_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    orm_bgr[:, :, 0] = metallic_norm   # B: Metallic
    orm_bgr[:, :, 1] = roughness_norm  # G: Roughness
    orm_bgr[:, :, 2] = occlusion_norm  # R: Occlusion
    orm_path = os.path.join(output_dir, "ORM.png")
    cv2_imwrite_utf8(orm_path, orm_bgr)

    return {
        "Albedo": base_img_path,
        "Roughness": roughness_path,
        "Metallic": metallic_path,
        "Normal": normal_path,
        "Displacement": disp_path,
        "ORM": orm_path,
    }

class MaterialInspectorDialog(QDialog):
    """독립된 구체(Sphere) 프리뷰 뷰포트 및 PBR 수치 파라미터 수동/AI 조절 패널"""
    def __init__(self, mat_info, apply_callback=None, parent=None):
        super().__init__(parent)
        self.mat_info = dict(mat_info)
        self.apply_callback = apply_callback
        
        self.setWindowTitle(f"🎨 PBR 머티리얼 검수 및 프리뷰 - {self.mat_info.get('name', 'Preset')}")
        self.setFixedWidth(1080)
        self.resize(1080, 880)
        
        if os.path.exists(ICON_PATH):
            self.setWindowIcon(QIcon(ICON_PATH))

        # 대화상자 고대비 가독성 QSS 스타일시트 적용
        self.setStyleSheet("""
            QDialog {
                background-color: #F8F9FA;
                font-family: 'Malgun Gothic', 'Segoe UI', sans-serif;
            }
            QGroupBox {
                font-weight: bold;
                font-size: 13px;
                color: #111111;
                border: 1px solid #B0B0B0;
                border-radius: 6px;
                margin-top: 6px;
                padding-top: 6px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 6px;
                color: #000000;
                font-weight: bold;
            }
            QLabel {
                color: #111111;
                font-size: 12px;
                font-weight: bold;
            }
            QToolTip {
                background-color: #1E1E24;
                color: #FFFFFF;
                border: 1px solid #555555;
                padding: 5px;
                font-size: 12px;
            }
        """)
            
        self.init_ui()

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        
        # ----------------- [좌측] 독립된 구체(Sphere) 머티리얼 전용 프리뷰 뷰포트 -----------------
        preview_box = QGroupBox("🔮 구체(Sphere) 머티리얼 프리뷰 (Material Ball)")
        preview_layout = QVBoxLayout(preview_box)
        
        self.ball_plotter = QtInteractor(self)
        self.ball_plotter.set_background("#1E1E24")
        preview_layout.addWidget(self.ball_plotter.interactor)
        
        # 3점 조명 세팅 (PBR 반사 및 요철 반사광 검수용)
        self.setup_lighting(self.ball_plotter)
        
        # 구체 메쉬 생성
        self.ball_mesh = pv.Sphere(radius=1.0, theta_resolution=80, phi_resolution=80)
        self.ball_mesh.compute_normals(inplace=True)
        
        color = self.mat_info.get("color", [0.8, 0.8, 0.8])
        roughness = self.mat_info.get("roughness", 0.5)
        metallic = self.mat_info.get("metallic", 0.0)
        normal_scale = self.mat_info.get("normal_scale", 1.0)
        
        self.ball_actor = self.ball_plotter.add_mesh(
            self.ball_mesh, color=color, pbr=True, roughness=roughness, metallic=metallic, show_edges=False
        )
        try:
            prop = self.ball_actor.GetProperty()
            if hasattr(prop, 'SetNormalScale'):
                prop.SetNormalScale(normal_scale)
        except Exception:
            pass
            
        self.ball_plotter.reset_camera()
        main_layout.addWidget(preview_box, stretch=1)
        
        # ----------------- [우측] PBR 파라미터 수치 조절 패널 (Inspector) -----------------
        inspector_box = QWidget()
        inspector_layout = QVBoxLayout(inspector_box)
        inspector_layout.setContentsMargins(0, 0, 0, 0)
        inspector_layout.setSpacing(6)
        
        # 1. PBR 속성 파라미터 조절 그룹
        prop_group = QGroupBox("🎛️ PBR 속성 파라미터 검수 및 수동 조절")
        form_layout = QFormLayout(prop_group)
        form_layout.setSpacing(6)

        # (0) 재질 이름 (Material Name) 입력/수정 필드
        self.txt_mat_name = QLineEdit(self.mat_info.get("name", "Preset"))
        self.txt_mat_name.setPlaceholderText("슬롯 재질 이름 입력...")
        self.txt_mat_name.setStyleSheet("font-weight: bold; font-size: 13px; padding: 4px; border: 1px solid #B0B0B0; border-radius: 4px; color: #111111; background-color: #FFFFFF;")
        self.txt_mat_name.textChanged.connect(self.on_name_changed)
        form_layout.addRow("재질 이름 (Name):", self.txt_mat_name)

        # (1) 알베도 컬러 (Albedo Color)
        color_layout = QHBoxLayout()
        self.btn_color = QPushButton()
        self.btn_color.setFixedSize(30, 30)
        self.update_color_button_style(color)
        self.btn_color.clicked.connect(self.choose_color)
        
        self.lbl_color_hex = QLabel(self.mat_info.get("hex", self.rgb_to_hex(color)))
        self.lbl_color_hex.setStyleSheet("font-weight: bold; font-family: monospace; font-size: 14px; color: #111111;")
        color_layout.addWidget(self.btn_color)
        color_layout.addWidget(self.lbl_color_hex)
        color_layout.addStretch()
        form_layout.addRow("알베도 컬러 (Color):", color_layout)
        
        # (1-2) HDRI 환경맵 (HDRI Environment Map) 항목 + 체크박스
        self.chk_hdri = QCheckBox("HDRI 환경맵:")
        self.chk_hdri.setChecked(self.mat_info.get("use_hdri", True))
        self.chk_hdri.toggled.connect(self.on_toggle_hdri)
        
        hdri_layout = QHBoxLayout()
        self.btn_load_hdri = QPushButton("📂 HDRI 로드")
        self.btn_load_hdri.setToolTip("다운로드한 .hdr, .exr 또는 이미지 형식의 HDRI 환경맵을 선택합니다.")
        self.btn_load_hdri.clicked.connect(self.choose_hdri_file)
        
        hdri_name = os.path.basename(self.mat_info.get("hdri_path", "")) if self.mat_info.get("hdri_path") else "기본 조명"
        self.lbl_hdri_name = QLabel()
        self.lbl_hdri_name.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.lbl_hdri_name.setMinimumWidth(0)
        self.lbl_hdri_name.setStyleSheet("font-size: 12px; font-weight: bold; color: #0044CC; background-color: #E8EEF9; padding: 4px 8px; border-radius: 4px; border: 1px solid #B0C4DE;")
        self.update_hdri_label_text(hdri_name, self.mat_info.get("hdri_path", ""))
        
        self.btn_clear_hdri = QPushButton("❌")
        self.btn_clear_hdri.setFixedSize(26, 26)
        self.btn_clear_hdri.setToolTip("HDRI 환경맵 적용 해제")
        self.btn_clear_hdri.setEnabled(bool(self.mat_info.get("hdri_path")) and self.chk_hdri.isChecked())
        self.btn_clear_hdri.clicked.connect(self.clear_hdri_texture)
        
        hdri_layout.addWidget(self.btn_load_hdri)
        hdri_layout.addWidget(self.lbl_hdri_name, stretch=1)
        hdri_layout.addWidget(self.btn_clear_hdri)
        form_layout.addRow(self.chk_hdri, hdri_layout)
        
        # (1-3) HDRI 적용 비율 (HDRI Ratio: 30% ~ 100%)
        hdri_ratio_val = self.mat_info.get("hdri_ratio", 1.0)
        hdri_ratio_layout = QHBoxLayout()
        
        self.spin_hdri_ratio = QDoubleSpinBox()
        self.spin_hdri_ratio.setRange(30.0, 100.0)
        self.spin_hdri_ratio.setSingleStep(5.0)
        self.spin_hdri_ratio.setSuffix("%")
        self.spin_hdri_ratio.setValue(hdri_ratio_val * 100.0)
        self.spin_hdri_ratio.setEnabled(self.chk_hdri.isChecked())
        
        self.slider_hdri_ratio = QSlider(Qt.Horizontal)
        self.slider_hdri_ratio.setRange(30, 100)
        self.slider_hdri_ratio.setValue(int(hdri_ratio_val * 100.0))
        self.slider_hdri_ratio.setEnabled(self.chk_hdri.isChecked())
        
        self.spin_hdri_ratio.valueChanged.connect(self.on_spin_hdri_ratio_changed)
        self.slider_hdri_ratio.valueChanged.connect(self.on_slider_hdri_ratio_changed)
        
        hdri_ratio_layout.addWidget(self.spin_hdri_ratio)
        hdri_ratio_layout.addWidget(self.slider_hdri_ratio)
        
        lbl_hdri_ratio_title = QLabel(" └ HDRI 적용 비율:")
        lbl_hdri_ratio_title.setStyleSheet("color: #333333; font-size: 11px; font-weight: bold;")
        self.lbl_hdri_ratio_title = lbl_hdri_ratio_title
        form_layout.addRow(lbl_hdri_ratio_title, hdri_ratio_layout)
        
        # (2) Bright (선명도) - DoubleSpinBox & Slider 동기화 + 체크박스
        self.chk_bright = QCheckBox("Bright (선명도):")
        self.chk_bright.setChecked(self.mat_info.get("use_bright", True))
        self.chk_bright.toggled.connect(self.on_toggle_bright)
        
        bright_val = self.mat_info.get("bright", 1.0)
        bright_layout = QHBoxLayout()
        self.spin_bright = QDoubleSpinBox()
        self.spin_bright.setRange(0.0, 10.0)
        self.spin_bright.setSingleStep(0.1)
        self.spin_bright.setValue(bright_val)
        
        self.slider_bright = QSlider(Qt.Horizontal)
        self.slider_bright.setRange(0, 1000)
        self.slider_bright.setValue(int(bright_val * 100))
        
        self.spin_bright.valueChanged.connect(self.on_spin_bright_changed)
        self.slider_bright.valueChanged.connect(self.on_slider_bright_changed)
        
        bright_layout.addWidget(self.spin_bright)
        bright_layout.addWidget(self.slider_bright)
        form_layout.addRow(self.chk_bright, bright_layout)
        
        # (3) Roughness (거칠기) - DoubleSpinBox & Slider 동기화 + 체크박스
        self.chk_rough = QCheckBox("Roughness (거칠기):")
        self.chk_rough.setChecked(self.mat_info.get("use_roughness", True))
        self.chk_rough.toggled.connect(self.on_toggle_rough)
        
        rough_layout = QHBoxLayout()
        self.spin_rough = QDoubleSpinBox()
        self.spin_rough.setRange(0.0, 1.0)
        self.spin_rough.setSingleStep(0.05)
        self.spin_rough.setValue(roughness)
        
        self.slider_rough = QSlider(Qt.Horizontal)
        self.slider_rough.setRange(0, 100)
        self.slider_rough.setValue(int(roughness * 100))
        
        self.spin_rough.valueChanged.connect(self.on_spin_rough_changed)
        self.slider_rough.valueChanged.connect(self.on_slider_rough_changed)
        
        rough_layout.addWidget(self.spin_rough)
        rough_layout.addWidget(self.slider_rough)
        form_layout.addRow(self.chk_rough, rough_layout)
        
        # (4) Metallic (금속성) - DoubleSpinBox & Slider 동기화 + 체크박스
        self.chk_metal = QCheckBox("Metallic (금속성):")
        self.chk_metal.setChecked(self.mat_info.get("use_metallic", True))
        self.chk_metal.toggled.connect(self.on_toggle_metal)
        
        metal_layout = QHBoxLayout()
        self.spin_metal = QDoubleSpinBox()
        self.spin_metal.setRange(0.0, 1.0)
        self.spin_metal.setSingleStep(0.05)
        self.spin_metal.setValue(metallic)
        
        self.slider_metal = QSlider(Qt.Horizontal)
        self.slider_metal.setRange(0, 100)
        self.slider_metal.setValue(int(metallic * 100))
        
        self.spin_metal.valueChanged.connect(self.on_spin_metal_changed)
        self.slider_metal.valueChanged.connect(self.on_slider_metal_changed)
        
        metal_layout.addWidget(self.spin_metal)
        metal_layout.addWidget(self.slider_metal)
        form_layout.addRow(self.chk_metal, metal_layout)
        
        # (5) Normal Scale (요철 스케일) + 체크박스
        self.chk_normal = QCheckBox("Normal Scale (요철):")
        self.chk_normal.setChecked(self.mat_info.get("use_normal", True))
        self.chk_normal.toggled.connect(self.on_toggle_normal)
        
        norm_layout = QHBoxLayout()
        self.spin_normal = QDoubleSpinBox()
        self.spin_normal.setRange(0.0, 3.0)
        self.spin_normal.setSingleStep(0.1)
        self.spin_normal.setValue(normal_scale)
        
        self.slider_normal = QSlider(Qt.Horizontal)
        self.slider_normal.setRange(0, 300)
        self.slider_normal.setValue(int(normal_scale * 100))
        
        self.spin_normal.valueChanged.connect(self.on_spin_normal_changed)
        self.slider_normal.valueChanged.connect(self.on_slider_normal_changed)
        
        norm_layout.addWidget(self.spin_normal)
        norm_layout.addWidget(self.slider_normal)
        form_layout.addRow(self.chk_normal, norm_layout)
        
        # (6) Reflection (반사도) - DoubleSpinBox & Slider 동기화 + 체크박스
        self.chk_reflect = QCheckBox("Reflection (반사도):")
        self.chk_reflect.setChecked(self.mat_info.get("use_reflection", True))
        self.chk_reflect.toggled.connect(self.on_toggle_reflect)
        
        reflect_val = self.mat_info.get("reflection", 1.0)
        reflect_layout = QHBoxLayout()
        self.spin_reflect = QDoubleSpinBox()
        self.spin_reflect.setRange(0.0, 10.0)
        self.spin_reflect.setSingleStep(0.1)
        self.spin_reflect.setValue(reflect_val)
        
        self.slider_reflect = QSlider(Qt.Horizontal)
        self.slider_reflect.setRange(0, 1000)
        self.slider_reflect.setValue(int(reflect_val * 100))
        
        self.spin_reflect.valueChanged.connect(self.on_spin_reflect_changed)
        self.slider_reflect.valueChanged.connect(self.on_slider_reflect_changed)
        
        reflect_layout.addWidget(self.spin_reflect)
        reflect_layout.addWidget(self.slider_reflect)
        form_layout.addRow(self.chk_reflect, reflect_layout)
        
        # (7) Refraction (굴절도/IOR) - DoubleSpinBox & Slider 동기화 + 체크박스
        self.chk_refract = QCheckBox("Refraction (굴절도):")
        self.chk_refract.setChecked(self.mat_info.get("use_refraction", True))
        self.chk_refract.toggled.connect(self.on_toggle_refract)
        
        refract_val = self.mat_info.get("refraction", 1.50)
        refract_layout = QHBoxLayout()
        self.spin_refract = QDoubleSpinBox()
        self.spin_refract.setRange(1.00, 3.00)
        self.spin_refract.setSingleStep(0.01)
        self.spin_refract.setDecimals(2)
        self.spin_refract.setValue(refract_val)
        self.spin_refract.setToolTip("굴절률(IOR): 1.0(공기), 1.33(물), 1.50(유리), 2.42(다이아몬드)")
        
        self.slider_refract = QSlider(Qt.Horizontal)
        self.slider_refract.setRange(100, 300)
        self.slider_refract.setValue(int(refract_val * 100))
        
        self.spin_refract.valueChanged.connect(self.on_spin_refract_changed)
        self.slider_refract.valueChanged.connect(self.on_slider_refract_changed)
        
        lbl_ior_tip = QLabel("(ex. IOR 1.5=유리)")
        lbl_ior_tip.setStyleSheet("color: #222222; font-weight: bold; font-size: 12px;")
        self.lbl_ior_tip = lbl_ior_tip
        
        refract_layout.addWidget(self.spin_refract)
        refract_layout.addWidget(self.slider_refract)
        refract_layout.addWidget(lbl_ior_tip)
        form_layout.addRow(self.chk_refract, refract_layout)
        
        # (8) Emissive (자발광) - DoubleSpinBox & Slider 동기화 + 체크박스
        self.chk_emissive = QCheckBox("Emissive (자발광):")
        self.chk_emissive.setChecked(self.mat_info.get("use_emissive", False))
        self.chk_emissive.toggled.connect(self.on_toggle_emissive)
        
        emissive_val = self.mat_info.get("emissive", 0.0)
        emissive_layout = QHBoxLayout()
        self.spin_emissive = QDoubleSpinBox()
        self.spin_emissive.setRange(0.0, 5.0)
        self.spin_emissive.setSingleStep(0.1)
        self.spin_emissive.setValue(emissive_val)
        self.spin_emissive.setToolTip("자가 발광 조명 강도 (디스플레이, 네온, 파이어, 발광 효과)")
        
        self.slider_emissive = QSlider(Qt.Horizontal)
        self.slider_emissive.setRange(0, 500)
        self.slider_emissive.setValue(int(emissive_val * 100))
        
        self.spin_emissive.valueChanged.connect(self.on_spin_emissive_changed)
        self.slider_emissive.valueChanged.connect(self.on_slider_emissive_changed)
        
        emissive_layout.addWidget(self.spin_emissive)
        emissive_layout.addWidget(self.slider_emissive)
        form_layout.addRow(self.chk_emissive, emissive_layout)

        # (9) Coat Strength (코팅 강도) - DoubleSpinBox & Slider 동기화 + 체크박스
        self.chk_coat_strength = QCheckBox("Coat Strength (코팅 강도):")
        self.chk_coat_strength.setChecked(self.mat_info.get("use_coat_strength", False))
        self.chk_coat_strength.toggled.connect(self.on_toggle_coat_strength)
        
        coat_val = self.mat_info.get("coat_strength", 0.0)
        coat_layout = QHBoxLayout()
        self.spin_coat_strength = QDoubleSpinBox()
        self.spin_coat_strength.setRange(0.0, 1.0)
        self.spin_coat_strength.setSingleStep(0.05)
        self.spin_coat_strength.setValue(coat_val)
        self.spin_coat_strength.setToolTip("투명 코팅층 반사 강도 (자동차 도장, 마감 니스, 에폭시/유광 마감)")
        
        self.slider_coat_strength = QSlider(Qt.Horizontal)
        self.slider_coat_strength.setRange(0, 100)
        self.slider_coat_strength.setValue(int(coat_val * 100))
        
        self.spin_coat_strength.valueChanged.connect(self.on_spin_coat_strength_changed)
        self.slider_coat_strength.valueChanged.connect(self.on_slider_coat_strength_changed)
        
        coat_layout.addWidget(self.spin_coat_strength)
        coat_layout.addWidget(self.slider_coat_strength)
        form_layout.addRow(self.chk_coat_strength, coat_layout)

        # (10) Coat Roughness (코팅 거칠기) - DoubleSpinBox & Slider 동기화 + 체크박스
        self.chk_coat_rough = QCheckBox("Coat Roughness (코팅 거칠기):")
        self.chk_coat_rough.setChecked(self.mat_info.get("use_coat_roughness", False))
        self.chk_coat_rough.toggled.connect(self.on_toggle_coat_rough)
        
        coat_rough_val = self.mat_info.get("coat_roughness", 0.0)
        coat_rough_layout = QHBoxLayout()
        self.spin_coat_rough = QDoubleSpinBox()
        self.spin_coat_rough.setRange(0.0, 1.0)
        self.spin_coat_rough.setSingleStep(0.05)
        self.spin_coat_rough.setValue(coat_rough_val)
        self.spin_coat_rough.setToolTip("투명 코팅층 표면 거칠기 (반광 코팅, 무광 니스, 오일 마감)")
        
        self.slider_coat_rough = QSlider(Qt.Horizontal)
        self.slider_coat_rough.setRange(0, 100)
        self.slider_coat_rough.setValue(int(coat_rough_val * 100))
        
        self.spin_coat_rough.valueChanged.connect(self.on_spin_coat_rough_changed)
        self.slider_coat_rough.valueChanged.connect(self.on_slider_coat_rough_changed)
        
        coat_rough_layout.addWidget(self.spin_coat_rough)
        coat_rough_layout.addWidget(self.slider_coat_rough)
        form_layout.addRow(self.chk_coat_rough, coat_rough_layout)

        # (11) Anisotropy (비등방성 반사) - DoubleSpinBox & Slider 동기화 + 체크박스
        self.chk_anisotropy = QCheckBox("Anisotropy (비등방성):")
        self.chk_anisotropy.setChecked(self.mat_info.get("use_anisotropy", False))
        self.chk_anisotropy.toggled.connect(self.on_toggle_anisotropy)
        
        aniso_val = self.mat_info.get("anisotropy", 0.0)
        aniso_layout = QHBoxLayout()
        self.spin_anisotropy = QDoubleSpinBox()
        self.spin_anisotropy.setRange(0.0, 1.0)
        self.spin_anisotropy.setSingleStep(0.05)
        self.spin_anisotropy.setValue(aniso_val)
        self.spin_anisotropy.setToolTip("방향성 결 무늬 반사 (헤어라인 메탈, 새틴 브러시, 레코드판, 탄소섬유)")
        
        self.slider_anisotropy = QSlider(Qt.Horizontal)
        self.slider_anisotropy.setRange(0, 100)
        self.slider_anisotropy.setValue(int(aniso_val * 100))
        
        self.spin_anisotropy.valueChanged.connect(self.on_spin_anisotropy_changed)
        self.slider_anisotropy.valueChanged.connect(self.on_slider_anisotropy_changed)
        
        aniso_layout.addWidget(self.spin_anisotropy)
        aniso_layout.addWidget(self.slider_anisotropy)
        form_layout.addRow(self.chk_anisotropy, aniso_layout)

        # (12) Occlusion (AO 음영 강도) - DoubleSpinBox & Slider 동기화 + 체크박스
        self.chk_occlusion = QCheckBox("Occlusion (AO 음영):")
        self.chk_occlusion.setChecked(self.mat_info.get("use_occlusion", False))
        self.chk_occlusion.toggled.connect(self.on_toggle_occlusion)
        
        occ_val = self.mat_info.get("occlusion", 1.0)
        occ_layout = QHBoxLayout()
        self.spin_occlusion = QDoubleSpinBox()
        self.spin_occlusion.setRange(0.0, 5.0)
        self.spin_occlusion.setSingleStep(0.1)
        self.spin_occlusion.setValue(occ_val)
        self.spin_occlusion.setToolTip("구석진 틈새 주변광 차폐(Ambient Occlusion) 음영 입체감 강도")
        
        self.slider_occlusion = QSlider(Qt.Horizontal)
        self.slider_occlusion.setRange(0, 500)
        self.slider_occlusion.setValue(int(occ_val * 100))
        
        self.spin_occlusion.valueChanged.connect(self.on_spin_occlusion_changed)
        self.slider_occlusion.valueChanged.connect(self.on_slider_occlusion_changed)
        
        occ_layout.addWidget(self.spin_occlusion)
        occ_layout.addWidget(self.slider_occlusion)
        form_layout.addRow(self.chk_occlusion, occ_layout)
        
        # (13) Occlusion(AO 음영) 항목 아래에 텍스처 불러오기 버튼 추가
        self.btn_load_texture = QPushButton("🖼️ 텍스처 불러오기 (PBR 맵 자동 생성)")
        self.btn_load_texture.setCursor(Qt.PointingHandCursor)
        self.btn_load_texture.setStyleSheet("""
            QPushButton {
                background-color: #2E7D32;
                color: #FFFFFF;
                font-weight: bold;
                font-size: 12px;
                padding: 6px 12px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1B5E20;
            }
        """)
        self.btn_load_texture.setToolTip("텍스처 파일(.jpg, .tif, .png, .bmp)을 불러오고 자동으로 Metallic.png, Normal.png, ORM.png, Displacement.png, Roughness.png 5개 파일 및 미리보기를 생성합니다.")
        self.btn_load_texture.clicked.connect(self.choose_and_generate_pbr_textures)
        form_layout.addRow("텍스처 수동/자동 로드:", self.btn_load_texture)

        # 각각의 이미지 보기 (6개) 2by3 (2행 3열) 배치 그룹박스
        tex_group = QGroupBox("🖼️ PBR 텍스처 맵 6종 미리보기 (2by3)")
        grid_tex = QGridLayout(tex_group)
        grid_tex.setSpacing(6)
        grid_tex.setContentsMargins(6, 6, 6, 6)

        self.tex_previews = {}

        maps_layout_spec = [
            ("Albedo", "Original / Albedo", 0, 0),
            ("Roughness", "Roughness.png", 0, 1),
            ("Metallic", "Metallic.png", 0, 2),
            ("Normal", "Normal.png", 1, 0),
            ("Displacement", "Displacement.png", 1, 1),
            ("ORM", "ORM.png", 1, 2),
        ]

        for map_key, map_title, r, c in maps_layout_spec:
            cell_w = QWidget()
            cell_vbox = QVBoxLayout(cell_w)
            cell_vbox.setContentsMargins(2, 2, 2, 2)
            cell_vbox.setSpacing(2)
            cell_vbox.setAlignment(Qt.AlignCenter)

            lbl_t = QLabel(map_title)
            lbl_t.setAlignment(Qt.AlignCenter)
            lbl_t.setStyleSheet("font-size: 11px; font-weight: bold; color: #222222;")

            lbl_img = QLabel("미로드")
            lbl_img.setFixedSize(90, 90)
            lbl_img.setAlignment(Qt.AlignCenter)
            lbl_img.setScaledContents(True)
            lbl_img.setStyleSheet("""
                QLabel {
                    border: 1px solid #B0B0B0;
                    border-radius: 4px;
                    background-color: #1E1E24;
                    color: #AAAAAA;
                    font-size: 10px;
                }
            """)

            cell_vbox.addWidget(lbl_t)
            cell_vbox.addWidget(lbl_img)
            grid_tex.addWidget(cell_w, r, c)

            self.tex_previews[map_key] = lbl_img

        form_layout.addRow(tex_group)
        
        # PBR 스크롤 영역 설정 (수평 스크롤바 원천 차단)
        scroll_area = QScrollArea()
        scroll_area.setWidget(prop_group)
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        
        inspector_layout.addWidget(scroll_area)
        
        # 2. AI Gemini 재질 어시스턴트 그룹
        ai_group = QGroupBox("🤖 Gemini AI 재질 튜너")
        ai_layout = QVBoxLayout(ai_group)
        
        ai_input_layout = QHBoxLayout()
        self.txt_ai_prompt = QLineEdit()
        self.txt_ai_prompt.setPlaceholderText("예: 비에 젖은 아스팔트, 광택 있는 세라믹...")
        self.btn_ai_gen = QPushButton("AI 수치 파싱")
        self.btn_ai_gen.clicked.connect(self.request_ai_material)
        ai_input_layout.addWidget(self.txt_ai_prompt)
        ai_input_layout.addWidget(self.btn_ai_gen)
        ai_layout.addLayout(ai_input_layout)
        
        self.lbl_ai_status = QLabel("AI에 원하는 재질 느낌을 입력하면 PBR 수치가 자동 조절됩니다.")
        self.lbl_ai_status.setStyleSheet("color: #222222; font-weight: bold; font-size: 12px; padding: 2px;")
        ai_layout.addWidget(self.lbl_ai_status)
        
        inspector_layout.addWidget(ai_group)
        
        # 3. 하단 액션 버튼
        btn_action_layout = QHBoxLayout()
        self.btn_apply = QPushButton("✔ 체크된 메인 모델(부품)에 이 머티리얼 적용")
        self.btn_apply.setStyleSheet("background-color: #2563eb; color: white; font-weight: bold; padding: 10px; border-radius: 5px;")
        self.btn_apply.clicked.connect(self.apply_to_model)
        
        self.btn_close = QPushButton("닫기")
        self.btn_close.setStyleSheet("color: #111111; font-weight: bold; padding: 10px; border-radius: 5px;")
        self.btn_close.clicked.connect(self.reject)
        
        btn_action_layout.addWidget(self.btn_apply)
        btn_action_layout.addWidget(self.btn_close)
        
        inspector_layout.addLayout(btn_action_layout)
        main_layout.addWidget(inspector_box, stretch=1)

        # 텍스처 미리보기 썸네일 초기화
        self.update_texture_previews()

        # 만약 전달된 mat_info에 HDRI 경로가 있다면 즉시 로드
        if self.mat_info.get("hdri_path") and os.path.exists(self.mat_info["hdri_path"]):
            self.load_hdri_texture(self.mat_info["hdri_path"])
        else:
            self.update_sphere_preview()

    def choose_and_generate_pbr_textures(self):
        """텍스처 파일(.jpg, .tif, .png, .bmp)을 불러와서 5종 PBR 맵을 자동 생성하고 미리보기에 배치"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, 
            "텍스처 이미지 불러오기", 
            "", 
            "Texture Files (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All Files (*.*)"
        )
        if not file_path or not os.path.exists(file_path):
            return

        try:
            out_dir = os.path.dirname(file_path)
            maps = generate_pbr_maps(file_path, out_dir)
            
            # mat_info에 맵 경로 세팅
            self.mat_info["texture_path"] = maps["Albedo"]
            self.mat_info["roughness_map"] = maps["Roughness"]
            self.mat_info["metallic_map"] = maps["Metallic"]
            self.mat_info["normal_map"] = maps["Normal"]
            self.mat_info["disp_map"] = maps["Displacement"]
            self.mat_info["orm_map"] = maps["ORM"]
            self.mat_info["use_texture"] = True
            
            # 미리보기 6개 이미지 업데이트
            self.update_texture_previews()
            
            # 구체 프리뷰 업데이트
            self.update_sphere_preview()
            
            QMessageBox.information(
                self, 
                "PBR 맵 생성 완료", 
                f"성공적으로 5가지 PBR 맵(Roughness, Metallic, Normal, Displacement, ORM)을 생성하였습니다!\n저장 경로: {out_dir}"
            )
        except Exception as e:
            QMessageBox.critical(self, "PBR 맵 생성 오류", f"텍스처 처리 중 오류가 발생했습니다:\n{e}")

    def update_texture_previews(self):
        """mat_info에 지정된 PBR 텍스처 파일들을 2x3 미리보기 라벨에 표시"""
        map_keys = {
            "Albedo": self.mat_info.get("texture_path"),
            "Roughness": self.mat_info.get("roughness_map"),
            "Metallic": self.mat_info.get("metallic_map"),
            "Normal": self.mat_info.get("normal_map"),
            "Displacement": self.mat_info.get("disp_map"),
            "ORM": self.mat_info.get("orm_map"),
        }

        for key, pth in map_keys.items():
            lbl = getattr(self, 'tex_previews', {}).get(key)
            if not lbl:
                continue
            if pth and os.path.exists(pth):
                pixmap = QPixmap(pth)
                if not pixmap.isNull():
                    lbl.setPixmap(pixmap.scaled(90, 90, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                    lbl.setToolTip(f"{key}: {pth}")
                else:
                    lbl.setText("오류")
            else:
                lbl.setText("미로드")

    def setup_lighting(self, plotter):
        """PBR 반사광 및 HDRI 입체감 검수용 3점 조명 세팅 (메인 뷰포트와 100% 일치)"""
        try:
            plotter.renderer.RemoveAllLights()
            key = pv.Light(position=(1.0, -1.0, 1.5), focal_point=(0.0, 0.0, 0.0), color='white', intensity=1.5, light_type='scenelight')
            fill = pv.Light(position=(-1.0, -1.0, 1.0), focal_point=(0.0, 0.0, 0.0), color=(0.85, 0.9, 1.0), intensity=0.8, light_type='scenelight')
            rim = pv.Light(position=(0.0, 1.0, -0.5), focal_point=(0.0, 0.0, 0.0), color=(1.0, 0.95, 0.85), intensity=1.0, light_type='scenelight')
            headlight = pv.Light(light_type='headlight', color='white', intensity=0.6)
            plotter.add_light(key)
            plotter.add_light(fill)
            plotter.add_light(rim)
            plotter.add_light(headlight)
        except Exception as e:
            print(f"프리뷰 조명 세팅 오류: {e}")

    def choose_hdri_file(self):
        """사용자가 파일 대화상자를 통해 .hdr, .exr 등의 HDRI 파일 선택"""
        default_dir = os.path.join(BASE_DIR, "HDRI_환경맵") if os.path.exists(os.path.join(BASE_DIR, "HDRI_환경맵")) else ""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "HDRI 환경맵 파일 선택", default_dir, "HDRI / Image Files (*.hdr *.exr *.png *.jpg);;All Files (*.*)"
        )
        if file_path:
            self.load_hdri_texture(file_path)

    def load_hdri_texture(self, file_path):
        """구체 프리뷰 뷰포트에 HDRI 큐브맵 환경 및 IBL 조명을 메인 시인과 100% 동일하게 적용합니다."""
        try:
            cubemap = read_hdri_texture(file_path, max_dim=1024)
            self.ball_plotter.set_environment_texture(cubemap, is_srgb=False, show_background=False)
            if hasattr(self.ball_plotter, 'enable_image_based_lighting'):
                try:
                    self.ball_plotter.enable_image_based_lighting()
                except Exception:
                    pass
            if hasattr(self.ball_plotter, 'renderer') and self.ball_plotter.renderer is not None:
                try:
                    self.ball_plotter.renderer.SetUseSphericalHarmonics(False)
                    self.ball_plotter.renderer.SetAutomaticLightCreation(False)
                except Exception:
                    pass
            self.setup_lighting(self.ball_plotter)
            self.mat_info["hdri_path"] = file_path
            self.update_hdri_label_text(os.path.basename(file_path), file_path)
            self.btn_clear_hdri.setEnabled(True)
            self.update_sphere_preview()
            self.ball_plotter.render()
            return True
        except Exception as e:
            print(f"HDRI 맵 적용 오류: {e}")
            return False

    def update_hdri_label_text(self, name, full_path=""):
        """HDRI 파일명이 길 경우 말줄임표(...)를 사용하여 레이아웃 확장 및 수평 스크롤바 생성 방지"""
        display_name = name
        if len(name) > 20:
            ext = os.path.splitext(name)[1]
            base = os.path.splitext(name)[0]
            display_name = base[:14] + "..." + ext
        self.lbl_hdri_name.setText(display_name)
        self.lbl_hdri_name.setToolTip(full_path if full_path else name)

    def clear_hdri_texture(self):
        """HDRI 환경 맵 적용 해제"""
        try:
            self.ball_plotter.remove_environment_texture()
            self.mat_info["hdri_path"] = None
            self.update_hdri_label_text("기본 조명", "")
            self.btn_clear_hdri.setEnabled(False)
            self.setup_lighting(self.ball_plotter)
            self.update_sphere_preview()
            self.ball_plotter.render()
        except Exception:
            pass

    def rgb_to_hex(self, rgb):
        r = int(rgb[0] * 255)
        g = int(rgb[1] * 255)
        b = int(rgb[2] * 255)
        return f"#{r:02X}{g:02X}{b:02X}"

    def update_color_button_style(self, color):
        hex_code = self.rgb_to_hex(color)
        self.btn_color.setStyleSheet(f"background-color: {hex_code}; border: 2px solid #555555; border-radius: 4px;")

    def choose_color(self):
        curr_color = self.mat_info.get("color", [0.8, 0.8, 0.8])
        qcurr = QColor(int(curr_color[0]*255), int(curr_color[1]*255), int(curr_color[2]*255))
        color = QColorDialog.getColor(qcurr, self, "알베도 색상 선택")
        if color.isValid():
            rgb = [color.redF(), color.greenF(), color.blueF()]
            self.mat_info["color"] = rgb
            self.mat_info["hex"] = self.rgb_to_hex(rgb)
            self.update_color_button_style(rgb)
            self.lbl_color_hex.setText(self.mat_info["hex"])
            self.update_sphere_preview()

    def on_spin_bright_changed(self, val):
        self.slider_bright.blockSignals(True)
        self.slider_bright.setValue(int(val * 100))
        self.slider_bright.blockSignals(False)
        self.mat_info["bright"] = val
        self.update_sphere_preview()

    def on_slider_bright_changed(self, val):
        float_val = val / 100.0
        self.spin_bright.blockSignals(True)
        self.spin_bright.setValue(float_val)
        self.spin_bright.blockSignals(False)
        self.mat_info["bright"] = float_val
        self.update_sphere_preview()

    def on_spin_hdri_ratio_changed(self, val):
        self.slider_hdri_ratio.blockSignals(True)
        self.slider_hdri_ratio.setValue(int(val))
        self.slider_hdri_ratio.blockSignals(False)
        self.mat_info["hdri_ratio"] = val / 100.0
        self.update_sphere_preview()

    def on_slider_hdri_ratio_changed(self, val):
        self.spin_hdri_ratio.blockSignals(True)
        self.spin_hdri_ratio.setValue(float(val))
        self.spin_hdri_ratio.blockSignals(False)
        self.mat_info["hdri_ratio"] = val / 100.0
        self.update_sphere_preview()

    def on_toggle_hdri(self, checked):
        self.btn_load_hdri.setEnabled(checked)
        self.lbl_hdri_name.setEnabled(checked)
        self.btn_clear_hdri.setEnabled(checked and bool(self.mat_info.get("hdri_path")))
        if hasattr(self, 'spin_hdri_ratio'):
            self.spin_hdri_ratio.setEnabled(checked)
            self.slider_hdri_ratio.setEnabled(checked)
        self.mat_info["use_hdri"] = checked
        if not checked:
            try:
                self.ball_plotter.remove_environment_texture()
            except Exception:
                pass
        elif self.mat_info.get("hdri_path") and os.path.exists(self.mat_info["hdri_path"]):
            self.load_hdri_texture(self.mat_info["hdri_path"])
        self.update_sphere_preview()

    def on_toggle_bright(self, checked):
        self.spin_bright.setEnabled(checked)
        self.slider_bright.setEnabled(checked)
        self.mat_info["use_bright"] = checked
        self.update_sphere_preview()

    def on_toggle_rough(self, checked):
        self.spin_rough.setEnabled(checked)
        self.slider_rough.setEnabled(checked)
        self.mat_info["use_roughness"] = checked
        self.update_sphere_preview()

    def on_toggle_metal(self, checked):
        self.spin_metal.setEnabled(checked)
        self.slider_metal.setEnabled(checked)
        self.mat_info["use_metallic"] = checked
        self.update_sphere_preview()

    def on_toggle_normal(self, checked):
        self.spin_normal.setEnabled(checked)
        self.slider_normal.setEnabled(checked)
        self.mat_info["use_normal"] = checked
        self.update_sphere_preview()

    def on_toggle_reflect(self, checked):
        self.spin_reflect.setEnabled(checked)
        self.slider_reflect.setEnabled(checked)
        self.mat_info["use_reflection"] = checked
        self.update_sphere_preview()

    def on_toggle_refract(self, checked):
        self.spin_refract.setEnabled(checked)
        self.slider_refract.setEnabled(checked)
        if hasattr(self, 'lbl_ior_tip'):
            self.lbl_ior_tip.setEnabled(checked)
        self.mat_info["use_refraction"] = checked
        self.update_sphere_preview()

    def on_toggle_emissive(self, checked):
        self.spin_emissive.setEnabled(checked)
        self.slider_emissive.setEnabled(checked)
        self.mat_info["use_emissive"] = checked
        self.update_sphere_preview()

    def on_toggle_coat_strength(self, checked):
        self.spin_coat_strength.setEnabled(checked)
        self.slider_coat_strength.setEnabled(checked)
        self.mat_info["use_coat_strength"] = checked
        self.update_sphere_preview()

    def on_toggle_coat_rough(self, checked):
        self.spin_coat_rough.setEnabled(checked)
        self.slider_coat_rough.setEnabled(checked)
        self.mat_info["use_coat_roughness"] = checked
        self.update_sphere_preview()

    def on_toggle_anisotropy(self, checked):
        self.spin_anisotropy.setEnabled(checked)
        self.slider_anisotropy.setEnabled(checked)
        self.mat_info["use_anisotropy"] = checked
        self.update_sphere_preview()

    def on_toggle_occlusion(self, checked):
        self.spin_occlusion.setEnabled(checked)
        self.slider_occlusion.setEnabled(checked)
        self.mat_info["use_occlusion"] = checked
        self.update_sphere_preview()

    def on_spin_bright_changed(self, val):
        self.slider_bright.blockSignals(True)
        self.slider_bright.setValue(int(val * 100))
        self.slider_bright.blockSignals(False)
        self.mat_info["bright"] = val
        self.update_sphere_preview()

    def on_slider_bright_changed(self, val):
        float_val = val / 100.0
        self.spin_bright.blockSignals(True)
        self.spin_bright.setValue(float_val)
        self.spin_bright.blockSignals(False)
        self.mat_info["bright"] = float_val
        self.update_sphere_preview()

    def on_spin_rough_changed(self, val):
        self.slider_rough.blockSignals(True)
        self.slider_rough.setValue(int(val * 100))
        self.slider_rough.blockSignals(False)
        self.mat_info["roughness"] = val
        self.update_sphere_preview()

    def on_slider_rough_changed(self, val):
        float_val = val / 100.0
        self.spin_rough.blockSignals(True)
        self.spin_rough.setValue(float_val)
        self.spin_rough.blockSignals(False)
        self.mat_info["roughness"] = float_val
        self.update_sphere_preview()

    def on_spin_metal_changed(self, val):
        self.slider_metal.blockSignals(True)
        self.slider_metal.setValue(int(val * 100))
        self.slider_metal.blockSignals(False)
        self.mat_info["metallic"] = val
        self.update_sphere_preview()

    def on_slider_metal_changed(self, val):
        float_val = val / 100.0
        self.spin_metal.blockSignals(True)
        self.spin_metal.setValue(float_val)
        self.spin_metal.blockSignals(False)
        self.mat_info["metallic"] = float_val
        self.update_sphere_preview()

    def on_spin_normal_changed(self, val):
        self.slider_normal.blockSignals(True)
        self.slider_normal.setValue(int(val * 100))
        self.slider_normal.blockSignals(False)
        self.mat_info["normal_scale"] = val
        self.update_sphere_preview()

    def on_slider_normal_changed(self, val):
        float_val = val / 100.0
        self.spin_normal.blockSignals(True)
        self.spin_normal.setValue(float_val)
        self.spin_normal.blockSignals(False)
        self.mat_info["normal_scale"] = float_val
        self.update_sphere_preview()

    def on_spin_reflect_changed(self, val):
        self.slider_reflect.blockSignals(True)
        self.slider_reflect.setValue(int(val * 100))
        self.slider_reflect.blockSignals(False)
        self.mat_info["reflection"] = val
        self.update_sphere_preview()

    def on_slider_reflect_changed(self, val):
        float_val = val / 100.0
        self.spin_reflect.blockSignals(True)
        self.spin_reflect.setValue(float_val)
        self.spin_reflect.blockSignals(False)
        self.mat_info["reflection"] = float_val
        self.update_sphere_preview()

    def on_spin_refract_changed(self, val):
        self.slider_refract.blockSignals(True)
        self.slider_refract.setValue(int(val * 100))
        self.slider_refract.blockSignals(False)
        self.mat_info["refraction"] = val
        self.update_sphere_preview()

    def on_slider_refract_changed(self, val):
        float_val = val / 100.0
        self.spin_refract.blockSignals(True)
        self.spin_refract.setValue(float_val)
        self.spin_refract.blockSignals(False)
        self.mat_info["refraction"] = float_val
        self.update_sphere_preview()

    def on_spin_emissive_changed(self, val):
        self.slider_emissive.blockSignals(True)
        self.slider_emissive.setValue(int(val * 100))
        self.slider_emissive.blockSignals(False)
        self.mat_info["emissive"] = val
        self.update_sphere_preview()

    def on_slider_emissive_changed(self, val):
        float_val = val / 100.0
        self.spin_emissive.blockSignals(True)
        self.spin_emissive.setValue(float_val)
        self.spin_emissive.blockSignals(False)
        self.mat_info["emissive"] = float_val
        self.update_sphere_preview()

    def on_spin_coat_strength_changed(self, val):
        self.slider_coat_strength.blockSignals(True)
        self.slider_coat_strength.setValue(int(val * 100))
        self.slider_coat_strength.blockSignals(False)
        self.mat_info["coat_strength"] = val
        self.update_sphere_preview()

    def on_slider_coat_strength_changed(self, val):
        float_val = val / 100.0
        self.spin_coat_strength.blockSignals(True)
        self.spin_coat_strength.setValue(float_val)
        self.spin_coat_strength.blockSignals(False)
        self.mat_info["coat_strength"] = float_val
        self.update_sphere_preview()

    def on_spin_coat_rough_changed(self, val):
        self.slider_coat_rough.blockSignals(True)
        self.slider_coat_rough.setValue(int(val * 100))
        self.slider_coat_rough.blockSignals(False)
        self.mat_info["coat_roughness"] = val
        self.update_sphere_preview()

    def on_slider_coat_rough_changed(self, val):
        float_val = val / 100.0
        self.spin_coat_rough.blockSignals(True)
        self.spin_coat_rough.setValue(float_val)
        self.spin_coat_rough.blockSignals(False)
        self.mat_info["coat_roughness"] = float_val
        self.update_sphere_preview()

    def on_spin_anisotropy_changed(self, val):
        self.slider_anisotropy.blockSignals(True)
        self.slider_anisotropy.setValue(int(val * 100))
        self.slider_anisotropy.blockSignals(False)
        self.mat_info["anisotropy"] = val
        self.update_sphere_preview()

    def on_slider_anisotropy_changed(self, val):
        float_val = val / 100.0
        self.spin_anisotropy.blockSignals(True)
        self.spin_anisotropy.setValue(float_val)
        self.spin_anisotropy.blockSignals(False)
        self.mat_info["anisotropy"] = float_val
        self.update_sphere_preview()

    def on_spin_occlusion_changed(self, val):
        self.slider_occlusion.blockSignals(True)
        self.slider_occlusion.setValue(int(val * 100))
        self.slider_occlusion.blockSignals(False)
        self.mat_info["occlusion"] = val
        self.update_sphere_preview()

    def on_slider_occlusion_changed(self, val):
        float_val = val / 100.0
        self.spin_occlusion.blockSignals(True)
        self.spin_occlusion.setValue(float_val)
        self.spin_occlusion.blockSignals(False)
        self.mat_info["occlusion"] = float_val
        self.update_sphere_preview()

    def update_sphere_preview(self):
        """체크박스가 켜진 PBR 항목만 구체 프리뷰에 실시간 반영하며 원색 어두워짐 보정 및 HDRI 적용 비율(30%~100%) 반영"""
        color = self.mat_info.get("color", [0.8, 0.8, 0.8])
        
        use_hdri = getattr(self, 'chk_hdri', None) and self.chk_hdri.isChecked()
        use_bright = getattr(self, 'chk_bright', None) and self.chk_bright.isChecked()
        use_rough = getattr(self, 'chk_rough', None) and self.chk_rough.isChecked()
        use_metal = getattr(self, 'chk_metal', None) and self.chk_metal.isChecked()
        use_normal = getattr(self, 'chk_normal', None) and self.chk_normal.isChecked()
        use_reflect = getattr(self, 'chk_reflect', None) and self.chk_reflect.isChecked()
        use_refract = getattr(self, 'chk_refract', None) and self.chk_refract.isChecked()
        use_emissive = getattr(self, 'chk_emissive', None) and self.chk_emissive.isChecked()
        use_coat_strength = getattr(self, 'chk_coat_strength', None) and self.chk_coat_strength.isChecked()
        use_coat_rough = getattr(self, 'chk_coat_rough', None) and self.chk_coat_rough.isChecked()
        use_anisotropy = getattr(self, 'chk_anisotropy', None) and self.chk_anisotropy.isChecked()
        use_occlusion = getattr(self, 'chk_occlusion', None) and self.chk_occlusion.isChecked()
        
        hdri_ratio = (self.spin_hdri_ratio.value() / 100.0) if hasattr(self, 'spin_hdri_ratio') else self.mat_info.get("hdri_ratio", 1.0)
        bright = self.spin_bright.value() if use_bright else 1.0
        rough = self.spin_rough.value() if use_rough else 0.5
        metal = self.spin_metal.value() if use_metal else 0.0
        normal_s = self.spin_normal.value() if use_normal else 0.0
        reflect = self.spin_reflect.value() if use_reflect else 0.0
        refract = self.spin_refract.value() if use_refract else 1.5
        emissive = self.spin_emissive.value() if use_emissive else 0.0
        coat_strength = self.spin_coat_strength.value() if use_coat_strength else 0.0
        coat_rough = self.spin_coat_rough.value() if use_coat_rough else 0.0
        anisotropy = self.spin_anisotropy.value() if use_anisotropy else 0.0
        occlusion = self.spin_occlusion.value() if use_occlusion else 1.0
        
        # bright(선명도) 적용 색상
        adj_color = [
            min(1.0, max(0.0, color[0] * bright)),
            min(1.0, max(0.0, color[1] * bright)),
            min(1.0, max(0.0, color[2] * bright))
        ]
        
        prop = getattr(self, 'ball_actor', None)
        if prop is not None and hasattr(prop, 'GetProperty'):
            prop = prop.GetProperty()
        else:
            return

        if not prop:
            return

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

        try:
            prop.SetColor(adj_color[0], adj_color[1], adj_color[2])
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
                prop.SetRoughness(rough)
            except Exception:
                pass
        if hasattr(prop, 'SetMetallic'):
            try:
                prop.SetMetallic(metal * hdri_ratio if use_hdri else metal)
            except Exception:
                pass
        if hasattr(prop, 'SetNormalScale'):
            try:
                prop.SetNormalScale(normal_s)
            except Exception:
                pass
        if hasattr(prop, 'SetSpecular'):
            try:
                prop.SetSpecular(min(1.0, reflect * hdri_ratio if use_hdri else reflect))
            except Exception:
                pass
        if hasattr(prop, 'SetSpecularPower'):
            try:
                prop.SetSpecularPower(max(1.0, reflect * 15.0))
            except Exception:
                pass

        # 신규 PBR 속성 (Emissive, Clearcoat, Anisotropy, Occlusion) 적용
        if hasattr(prop, 'SetEmissiveFactor'):
            try:
                prop.SetEmissiveFactor(emissive, emissive, emissive)
            except Exception:
                pass
        if hasattr(prop, 'SetCoatStrength'):
            try:
                prop.SetCoatStrength(coat_strength)
            except Exception:
                pass
        if hasattr(prop, 'SetCoatRoughness'):
            try:
                prop.SetCoatRoughness(coat_rough)
            except Exception:
                pass
        if hasattr(prop, 'SetAnisotropy'):
            try:
                prop.SetAnisotropy(anisotropy)
            except Exception:
                pass
        if hasattr(prop, 'SetOcclusionStrength'):
            try:
                prop.SetOcclusionStrength(occlusion)
            except Exception:
                pass

        # 굴절(Refraction) 항목 체크 시 투명 재질(Opacity & IOR) 표현
        if use_refract:
            if hasattr(prop, 'SetIOR'):
                try:
                    prop.SetIOR(refract)
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

        # 텍스처(Albedo/Normal 등)가 지정되어 있는 경우 구체 프리뷰 텍스처 입히기
        tex_path = self.mat_info.get("texture_path")
        if tex_path and os.path.exists(tex_path):
            try:
                tex = pv.read_texture(tex_path)
                self.ball_actor.texture = tex
            except Exception:
                pass

        try:
            self.ball_plotter.render()
        except Exception:
            pass

    def request_ai_material(self):
        text = self.txt_ai_prompt.text().strip()
        if not text:
            return
        self.lbl_ai_status.setText("⏳ Gemini AI가 PBR 파라미터를 분석 중입니다...")
        self.btn_ai_gen.setEnabled(False)
        
        self.ai_worker = MaterialAIWorker(text, api_key=GEMINI_API_KEY)
        self.ai_worker.finished.connect(self.on_ai_received)
        self.ai_worker.error.connect(self.on_ai_error)
        self.ai_worker.start()

    def on_ai_received(self, data):
        self.btn_ai_gen.setEnabled(True)
        self.lbl_ai_status.setText(f"✅ AI 수치 수신 완료: {data.get('desc', '')}")
        
        if "color" in data:
            color = data["color"]
            self.mat_info["color"] = color
            self.mat_info["hex"] = self.rgb_to_hex(color)
            self.update_color_button_style(color)
            self.lbl_color_hex.setText(self.mat_info["hex"])
            
        if "bright" in data or "brightness" in data:
            b = float(data.get("bright", data.get("brightness", 1.0)))
            self.spin_bright.setValue(b)
            self.mat_info["bright"] = b

        if "roughness" in data:
            r = float(data["roughness"])
            self.spin_rough.setValue(r)
            self.mat_info["roughness"] = r

        if "reflection" in data:
            rf = float(data["reflection"])
            self.spin_reflect.setValue(rf)
            self.mat_info["reflection"] = rf

        if "refraction" in data or "ior" in data:
            ior = float(data.get("refraction", data.get("ior", 1.5)))
            self.spin_refract.setValue(ior)
            self.mat_info["refraction"] = ior
            
        if "metallic" in data:
            m = float(data["metallic"])
            self.spin_metal.setValue(m)
            self.mat_info["metallic"] = m
            
        if "normal_scale" in data:
            ns = float(data["normal_scale"])
            self.spin_normal.setValue(ns)
            self.mat_info["normal_scale"] = ns

        if "emissive" in data:
            e = float(data["emissive"])
            self.spin_emissive.setValue(e)
            self.mat_info["emissive"] = e

        if "coat_strength" in data:
            cs = float(data["coat_strength"])
            self.spin_coat_strength.setValue(cs)
            self.mat_info["coat_strength"] = cs

        if "coat_roughness" in data:
            cr = float(data["coat_roughness"])
            self.spin_coat_rough.setValue(cr)
            self.mat_info["coat_roughness"] = cr

        if "anisotropy" in data:
            an = float(data["anisotropy"])
            self.spin_anisotropy.setValue(an)
            self.mat_info["anisotropy"] = an

        if "occlusion" in data:
            occ = float(data["occlusion"])
            self.spin_occlusion.setValue(occ)
            self.mat_info["occlusion"] = occ
            
        self.update_sphere_preview()

    def on_ai_error(self, err_msg):
        self.btn_ai_gen.setEnabled(True)
        self.lbl_ai_status.setText(f"❌ AI 분석 오류: {err_msg}")

    def on_name_changed(self, text):
        name = text.strip() or "Preset"
        self.mat_info["name"] = name
        self.setWindowTitle(f"🎨 PBR 머티리얼 검수 및 프리뷰 - {name}")

    def apply_to_model(self):
        # UI 컨트롤의 최신 수치 및 재질 이름을 mat_info에 정확히 동기화
        if hasattr(self, 'txt_mat_name'):
            name_val = self.txt_mat_name.text().strip()
            self.mat_info["name"] = name_val or "Preset"

        if hasattr(self, 'spin_bright'): self.mat_info["bright"] = self.spin_bright.value()
        if hasattr(self, 'spin_rough'): self.mat_info["roughness"] = self.spin_rough.value()
        if hasattr(self, 'spin_metal'): self.mat_info["metallic"] = self.spin_metal.value()
        if hasattr(self, 'spin_normal'): self.mat_info["normal_scale"] = self.spin_normal.value()
        if hasattr(self, 'spin_reflect'): self.mat_info["reflection"] = self.spin_reflect.value()
        if hasattr(self, 'spin_refract'): self.mat_info["refraction"] = self.spin_refract.value()
        if hasattr(self, 'spin_emissive'): self.mat_info["emissive"] = self.spin_emissive.value()
        if hasattr(self, 'spin_coat_strength'): self.mat_info["coat_strength"] = self.spin_coat_strength.value()
        if hasattr(self, 'spin_coat_rough'): self.mat_info["coat_roughness"] = self.spin_coat_rough.value()
        if hasattr(self, 'spin_anisotropy'): self.mat_info["anisotropy"] = self.spin_anisotropy.value()
        if hasattr(self, 'spin_occlusion'): self.mat_info["occlusion"] = self.spin_occlusion.value()

        self.mat_info["use_hdri"] = self.chk_hdri.isChecked()
        self.mat_info["hdri_ratio"] = self.spin_hdri_ratio.value() / 100.0 if hasattr(self, 'spin_hdri_ratio') else 1.0
        self.mat_info["use_bright"] = self.chk_bright.isChecked()
        self.mat_info["use_roughness"] = self.chk_rough.isChecked()
        self.mat_info["use_metallic"] = self.chk_metal.isChecked()
        self.mat_info["use_normal"] = self.chk_normal.isChecked()
        self.mat_info["use_reflection"] = self.chk_reflect.isChecked()
        self.mat_info["use_refraction"] = self.chk_refract.isChecked()
        self.mat_info["use_emissive"] = self.chk_emissive.isChecked()
        self.mat_info["use_coat_strength"] = self.chk_coat_strength.isChecked()
        self.mat_info["use_coat_roughness"] = self.chk_coat_rough.isChecked()
        self.mat_info["use_anisotropy"] = self.chk_anisotropy.isChecked()
        self.mat_info["use_occlusion"] = self.chk_occlusion.isChecked()

        # 적용할 머티리얼 정보를 저장하고 다이얼로그 정상 종료
        self.applied_mat_info = dict(self.mat_info)
        self.accept()

    def cleanup(self):
        """다이얼로그 종료 시 VTK QtInteractor 및 OpenGL 리소스를 언리얼 엔진 수준으로 100% 안전하게 해제"""
        if getattr(self, '_cleaned_up', False):
            return
        self._cleaned_up = True
        
        plotter = getattr(self, 'ball_plotter', None)
        self.ball_plotter = None
        if plotter is not None:
            try:
                # 1. 렌더 타이머 정지
                if hasattr(plotter, 'render_timer') and plotter.render_timer is not None:
                    try:
                        plotter.render_timer.stop()
                    except Exception:
                        pass
                # 2. 렌더러에서 액터, 조명, 환경 텍스처 명시적 분리
                if hasattr(plotter, 'renderer') and plotter.renderer is not None:
                    try:
                        plotter.renderer.RemoveAllViewProps()
                        plotter.renderer.RemoveAllLights()
                        if hasattr(plotter.renderer, 'SetEnvironmentTexture'):
                            plotter.renderer.SetEnvironmentTexture(None)
                    except Exception:
                        pass
                # 3. 플로터 메시 및 리소스 비우기
                if hasattr(plotter, 'clear'):
                    try:
                        plotter.clear()
                    except Exception:
                        pass
                # 4. VTK OpenGL 렌더 윈도우 Finalize (GPU 리소스 안전 언로드)
                if hasattr(plotter, 'render_window') and plotter.render_window is not None:
                    try:
                        plotter.render_window.Finalize()
                    except Exception:
                        pass
                # 5. 플로터 닫기
                if hasattr(plotter, '_closed') and not plotter._closed:
                    try:
                        plotter.close()
                    except Exception:
                        pass
                # 6. 인터랙터 위젯 부모 분리 및 소멸 예약
                if hasattr(plotter, 'interactor') and plotter.interactor is not None:
                    try:
                        plotter.interactor.setParent(None)
                        plotter.interactor.deleteLater()
                    except Exception:
                        pass
            except Exception as e:
                print(f"MaterialInspectorDialog cleanup 예외: {e}")

    def accept(self):
        self.cleanup()
        super().accept()

    def reject(self):
        self.cleanup()
        super().reject()

    def closeEvent(self, event):
        self.cleanup()
        super().closeEvent(event)

class MaterialSlotPanel(QWidget):
    """Cinematic View 하단 동적 원형 재질(Material) 슬롯 패널 위젯 (최대 100개 지원, 우클릭 삭제/이름변경 지원)"""
    def __init__(self, apply_callback=None, parent=None):
        super().__init__(parent)
        self.apply_callback = apply_callback
        
        # 기본 20가지 3D 머티리얼 프리셋 정의
        self.materials = [
            {"name": "Red Metallic", "hex": "#E53935", "color": [0.898, 0.223, 0.208], "roughness": 0.2, "metallic": 0.9, "normal_scale": 1.0},
            {"name": "Glossy Black", "hex": "#1E1E1E", "color": [0.117, 0.117, 0.117], "roughness": 0.05, "metallic": 0.1, "normal_scale": 1.0},
            {"name": "Pearl White", "hex": "#F5F5F5", "color": [0.960, 0.960, 0.960], "roughness": 0.25, "metallic": 0.1, "normal_scale": 1.0},
            {"name": "Ocean Blue", "hex": "#1E88E5", "color": [0.117, 0.533, 0.898], "roughness": 0.3, "metallic": 0.4, "normal_scale": 1.0},
            {"name": "Racing Yellow", "hex": "#FDD835", "color": [0.992, 0.847, 0.208], "roughness": 0.15, "metallic": 0.3, "normal_scale": 1.0},
            {"name": "Matte Gunmetal", "hex": "#424242", "color": [0.258, 0.258, 0.258], "roughness": 0.7, "metallic": 0.8, "normal_scale": 1.2},
            {"name": "Chrome Silver", "hex": "#CFD8DC", "color": [0.811, 0.847, 0.862], "roughness": 0.05, "metallic": 1.0, "normal_scale": 1.0},
            {"name": "Rose Gold", "hex": "#E8B4B8", "color": [0.909, 0.705, 0.721], "roughness": 0.2, "metallic": 0.85, "normal_scale": 1.0},
            {"name": "Lime Green", "hex": "#7CB342", "color": [0.486, 0.701, 0.258], "roughness": 0.4, "metallic": 0.1, "normal_scale": 1.0},
            {"name": "Sunset Orange", "hex": "#FB8C00", "color": [0.984, 0.549, 0.000], "roughness": 0.25, "metallic": 0.2, "normal_scale": 1.0},
            {"name": "Tinted Cyan", "hex": "#00ACC1", "color": [0.000, 0.674, 0.756], "roughness": 0.1, "metallic": 0.5, "normal_scale": 1.0},
            {"name": "Carbon Dark", "hex": "#263238", "color": [0.149, 0.196, 0.219], "roughness": 0.5, "metallic": 0.7, "normal_scale": 1.5},
            {"name": "Leather Brown", "hex": "#8D6E63", "color": [0.552, 0.431, 0.388], "roughness": 0.8, "metallic": 0.0, "normal_scale": 1.8},
            {"name": "Neon Electric", "hex": "#00E5FF", "color": [0.000, 0.898, 1.000], "roughness": 0.1, "metallic": 0.6, "normal_scale": 1.0},
            {"name": "Royal Purple", "hex": "#8E24AA", "color": [0.556, 0.141, 0.666], "roughness": 0.3, "metallic": 0.5, "normal_scale": 1.0},
            {"name": "Matte Satin Grey", "hex": "#78909C", "color": [0.470, 0.564, 0.611], "roughness": 0.6, "metallic": 0.2, "normal_scale": 1.0},
            {"name": "Champagne Gold", "hex": "#D4AF37", "color": [0.831, 0.686, 0.215], "roughness": 0.15, "metallic": 0.95, "normal_scale": 1.0},
            {"name": "Emerald Green", "hex": "#00897B", "color": [0.000, 0.537, 0.482], "roughness": 0.2, "metallic": 0.4, "normal_scale": 1.0},
            {"name": "Ruby Magenta", "hex": "#D81B60", "color": [0.847, 0.105, 0.376], "roughness": 0.1, "metallic": 0.7, "normal_scale": 1.0},
            {"name": "Deep Indigo", "hex": "#3F51B5", "color": [0.247, 0.317, 0.709], "roughness": 0.3, "metallic": 0.3, "normal_scale": 1.0}
        ]
        
        self.init_ui()

    def init_ui(self):
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMaximumHeight(85)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(2, 0, 2, 2)
        main_layout.setSpacing(0)
        
        container = QFrame()
        container.setStyleSheet("""
            QFrame {
                background-color: #3A3F4B;
                border: 1px solid #4D5464;
                border-radius: 6px;
            }
        """)
        
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(6, 2, 6, 2)
        container_layout.setSpacing(1)
        
        header_layout = QHBoxLayout()
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(4)
        
        if os.path.exists(ICON_PATH):
            mat_logo_label = QLabel()
            mat_logo_label.setPixmap(QPixmap(ICON_PATH).scaled(14, 14, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            header_layout.addWidget(mat_logo_label)
            
        self.lbl_title = QLabel(f"🎨 머티리얼 슬롯 ({len(self.materials)}/100)")
        self.lbl_title.setStyleSheet("color: #FFFFFF; font-weight: bold; font-size: 11px; border: none;")
        
        btn_add_slot = QPushButton("➕ 슬롯 추가")
        btn_add_slot.setCursor(Qt.PointingHandCursor)
        btn_add_slot.setStyleSheet("""
            QPushButton {
                background-color: #2563EB;
                color: #FFFFFF;
                font-weight: bold;
                font-size: 10px;
                padding: 1px 5px;
                border-radius: 3px;
                border: none;
            }
            QPushButton:hover {
                background-color: #1D4ED8;
            }
        """)
        btn_add_slot.setToolTip("최대 100개까지 신규 빈 머티리얼 슬롯을 추가합니다.")
        btn_add_slot.clicked.connect(self.on_add_slot)
        
        lbl_tip = QLabel("💡 좌클릭: 수치/이름 수정 및 적용 | 우클릭: 삭제/이름 변경")
        lbl_tip.setStyleSheet("color: #E2E8F0; font-size: 10px; border: none;")
        
        header_layout.addWidget(self.lbl_title)
        header_layout.addWidget(btn_add_slot)
        header_layout.addSpacing(6)
        header_layout.addWidget(lbl_tip)
        header_layout.addStretch()
        container_layout.addLayout(header_layout)
        
        # 슬롯 그리드 스크롤 영역 (하단 밀착 슬림 높이 고정)
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.NoFrame)
        self.scroll_area.setFixedHeight(58)
        self.scroll_area.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        
        self.grid_widget = QWidget()
        self.grid_layout = QGridLayout(self.grid_widget)
        self.grid_layout.setSpacing(3)
        self.grid_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_area.setWidget(self.grid_widget)
        
        container_layout.addWidget(self.scroll_area)
        main_layout.addWidget(container)
        
        self.rebuild_grid_ui()

    def rebuild_grid_ui(self):
        """슬롯 추가/삭제 시 그리드 UI 전체 동적 재구성"""
        for i in reversed(range(self.grid_layout.count())):
            item = self.grid_layout.itemAt(i)
            if item and item.widget():
                item.widget().setParent(None)
                
        self.slot_buttons = []
        if hasattr(self, 'lbl_title'):
            self.lbl_title.setText(f"🎨 머티리얼 슬롯 ({len(self.materials)}/100)")
            
        for idx, mat in enumerate(self.materials):
            row = idx // 10
            col = idx % 10
            
            btn = QPushButton()
            btn.setFixedSize(26, 26)
            btn.setCursor(Qt.PointingHandCursor)
            name = mat.get('name', f"Slot #{idx+1}")
            hex_col = mat.get('hex', '#888888')
            btn.setToolTip(f"{idx+1}. {name}\n(좌클릭: PBR 수치/이름 수정 | 우클릭: 삭제/이름 변경)")
            
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {hex_col};
                    border: 2px solid #555555;
                    border-radius: 13px;
                }}
                QPushButton:hover {{
                    border: 2px solid #FFFFFF;
                    background-color: {hex_col};
                }}
                QPushButton:pressed {{
                    border: 2px solid #00E5FF;
                }}
            """)
            
            # 좌클릭: 머티리얼 검수/수정/적용 및 이름 저장
            btn.clicked.connect(lambda checked=False, slot_i=idx: self.on_slot_clicked(slot_i))
            
            # 우클릭: 삭제 및 이름 변경 컨텍스트 메뉴
            btn.setContextMenuPolicy(Qt.CustomContextMenu)
            btn.customContextMenuRequested.connect(lambda pos, slot_i=idx: self.on_slot_context_menu(slot_i, pos))
            
            self.grid_layout.addWidget(btn, row, col, alignment=Qt.AlignCenter)
            self.slot_buttons.append(btn)

    def on_add_slot(self):
        """신규 빈 머티리얼 슬롯 생성 (최대 100개까지 생성)"""
        if len(self.materials) >= 100:
            QMessageBox.information(self, "슬롯 한도 초과", "머티리얼 슬롯은 최대 100개까지만 생성할 수 있습니다.")
            return
            
        new_idx = len(self.materials) + 1
        new_mat = {
            "name": f"New Material #{new_idx}",
            "hex": "#888888",
            "color": [0.5, 0.5, 0.5],
            "roughness": 0.5,
            "metallic": 0.0,
            "normal_scale": 1.0
        }
        self.materials.append(new_mat)
        self.rebuild_grid_ui()

    def on_slot_context_menu(self, slot_idx, pos):
        """슬롯 우클릭 팝업 메뉴 (슬롯 삭제 & 이름 변경)"""
        if not (0 <= slot_idx < len(self.materials)):
            return
            
        btn = self.slot_buttons[slot_idx]
        mat = self.materials[slot_idx]
        
        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #2D2D30;
                color: #FFFFFF;
                border: 1px solid #555555;
                font-size: 12px;
            }
            QMenu::item {
                padding: 6px 20px;
            }
            QMenu::item:selected {
                background-color: #007ACC;
            }
        """)
        
        act_rename = menu.addAction(f"✏️ '{mat.get('name')}' 이름 변경")
        act_delete = menu.addAction(f"🗑️ '{mat.get('name')}' 슬롯 삭제")
        
        action = menu.exec(btn.mapToGlobal(pos))
        if action == act_rename:
            new_name, ok = QInputDialog.getText(self, "이름 변경", "새 머티리얼 이름을 입력하세요:", text=mat.get("name", ""))
            if ok and new_name.strip():
                mat["name"] = new_name.strip()
                self.rebuild_grid_ui()
                self.material_changed.emit(self.materials)
        elif action == act_delete:
            if len(self.materials) <= 1:
                QMessageBox.warning(self, "삭제 불가", "최소 1개의 머티리얼 슬롯은 존재해야 합니다.")
                return
            reply = QMessageBox.question(self, "슬롯 삭제", f"'{mat.get('name')}' 슬롯을 삭제하시겠습니까?", QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                self.materials.pop(slot_idx)
                if self.active_index >= len(self.materials):
                    self.active_index = max(0, len(self.materials) - 1)
                self.rebuild_grid_ui()
                self.select_slot(self.active_index)
                self.material_changed.emit(self.materials)

class VTK2DMarqueeOverlay:
    """VTK Native Direct 2D Screen Overlay Marquee Box (윤곽선 + 옅은 파란색 반투명 영역 채우기)"""
    def __init__(self, target_widget, plotter=None):
        self.target_widget = target_widget
        self.plotter = plotter
        self.points = vtk.vtkPoints()
        self.points.SetNumberOfPoints(4)
        
        # 1. 외곽 윤곽선 (PolyLine)
        self.lines = vtk.vtkCellArray()
        line_indices = [0, 1, 1, 2, 2, 3, 3, 0]
        for i in range(0, len(line_indices), 2):
            line = vtk.vtkPolyLine()
            line.GetPointIds().SetNumberOfIds(2)
            line.GetPointIds().SetId(0, line_indices[i])
            line.GetPointIds().SetId(1, line_indices[i+1])
            self.lines.InsertNextCell(line)
            
        self.polydata_line = vtk.vtkPolyData()
        self.polydata_line.SetPoints(self.points)
        self.polydata_line.SetLines(self.lines)
        
        # 2. 내부 채우기 면 (Polygon)
        self.polys = vtk.vtkCellArray()
        quad = vtk.vtkPolygon()
        quad.GetPointIds().SetNumberOfIds(4)
        for i in range(4):
            quad.GetPointIds().SetId(i, i)
        self.polys.InsertNextCell(quad)

        self.polydata_fill = vtk.vtkPolyData()
        self.polydata_fill.SetPoints(self.points)
        self.polydata_fill.SetPolys(self.polys)

        self.coordinate = vtk.vtkCoordinate()
        self.coordinate.SetCoordinateSystemToDisplay()
        
        # Mapper & Actor for Line (옅은 파란색 외곽선)
        self.mapper_line = vtk.vtkPolyDataMapper2D()
        self.mapper_line.SetInputData(self.polydata_line)
        self.mapper_line.SetTransformCoordinate(self.coordinate)
        
        self.actor_line = vtk.vtkActor2D()
        self.actor_line.SetMapper(self.mapper_line)
        self.actor_line.GetProperty().SetColor(0.2, 0.6, 1.0) # 옅은 파란색 (#3399FF)
        self.actor_line.GetProperty().SetLineWidth(2.5)
        self.actor_line.SetVisibility(False)

        # Mapper & Actor for Fill (옅은 파란색 반투명 채우기)
        self.mapper_fill = vtk.vtkPolyDataMapper2D()
        self.mapper_fill.SetInputData(self.polydata_fill)
        self.mapper_fill.SetTransformCoordinate(self.coordinate)
        
        self.actor_fill = vtk.vtkActor2D()
        self.actor_fill.SetMapper(self.mapper_fill)
        self.actor_fill.GetProperty().SetColor(0.25, 0.65, 1.0) # 옅은 파란색 (#40A0FF)
        self.actor_fill.GetProperty().SetOpacity(0.30) # 옅은 반투명 채우기
        self.actor_fill.SetVisibility(False)

        self.renderer = None

    def _get_render_window(self):
        try:
            if hasattr(self.target_widget, 'GetRenderWindow'):
                return self.target_widget.GetRenderWindow()
            if self.plotter and hasattr(self.plotter, 'render_window'):
                return self.plotter.render_window
        except Exception:
            pass
        return None

    def _get_renderer(self):
        if self.renderer:
            return self.renderer
        try:
            if self.plotter and hasattr(self.plotter, 'renderer') and self.plotter.renderer:
                self.renderer = self.plotter.renderer
                return self.renderer
            if hasattr(self.target_widget, 'renderer') and self.target_widget.renderer:
                self.renderer = self.target_widget.renderer
                return self.renderer
            rw = self._get_render_window()
            if rw and rw.GetRenderers().GetNumberOfItems() > 0:
                self.renderer = rw.GetRenderers().GetFirstRenderer()
                return self.renderer
        except Exception:
            pass
        return self.renderer

    def update_box(self, origin_p, cur_p):
        ren = self._get_renderer()
        if not ren:
            return
            
        actors2d = ren.GetActors2D()
        if self.actor_fill not in actors2d:
            ren.AddActor2D(self.actor_fill)
        if self.actor_line not in actors2d:
            ren.AddActor2D(self.actor_line)
            
        win_h = self.target_widget.height()
        if win_h <= 0:
            rw = self._get_render_window()
            if rw:
                win_h = rw.GetSize()[1]
        if win_h <= 0:
            win_h = 600

        x0, y0 = origin_p.x(), win_h - origin_p.y()
        x1, y1 = cur_p.x(), win_h - cur_p.y()
        
        self.points.SetPoint(0, x0, y0, 0)
        self.points.SetPoint(1, x1, y0, 0)
        self.points.SetPoint(2, x1, y1, 0)
        self.points.SetPoint(3, x0, y1, 0)
        self.polydata_line.Modified()
        self.polydata_fill.Modified()
        
        # 옅은 파란색으로 일관되게 표시
        self.actor_line.GetProperty().SetColor(0.2, 0.6, 1.0)
        self.actor_fill.GetProperty().SetColor(0.25, 0.65, 1.0)
        self.actor_fill.GetProperty().SetOpacity(0.30)
            
        self.actor_line.SetVisibility(True)
        self.actor_fill.SetVisibility(True)
        
        # 실시간 즉각 렌더링 동기화
        try:
            rw = self._get_render_window()
            if rw:
                rw.Render()
            elif hasattr(self.target_widget, 'render'):
                self.target_widget.render()
        except Exception:
            pass

    def hide(self):
        if self.actor_line and self.actor_line.GetVisibility():
            self.actor_line.SetVisibility(False)
        if self.actor_fill and self.actor_fill.GetVisibility():
            self.actor_fill.SetVisibility(False)

class RubberBandFilter(QObject):
    def __init__(self, target_widget, clear_callback=None, plotter=None):
        super().__init__(target_widget)
        self.target = target_widget
        self.plotter = plotter
        self.clear_callback = clear_callback
        
        self.vtk_overlay = VTK2DMarqueeOverlay(target_widget, plotter=plotter)
        self.origin = QPoint()
        self.release_pos = QPoint()
        self.last_rect = QRect()
        self.is_left_to_right = False
        self.is_active = False
        self.is_dragging = False
        self.install_all_filters()
        self.bind_vtk_observers()

    def install_all_filters(self):
        """target_widget 및 그 하위 모든 뷰포트 자식 위젯들에 eventFilter 전파 설치"""
        try:
            self.target.installEventFilter(self)
            from PySide6.QtWidgets import QWidget
            for child in self.target.findChildren(QWidget):
                child.installEventFilter(self)
        except Exception:
            pass

    def bind_vtk_observers(self):
        """
        VTK 렌더러/인터랙터 직접 옵저버는 피커의 AreaPick 및 HardwareSelector 파이프라인과
        충돌하므로 바인딩하지 않고, 마우스 이벤트는 격리된 Qt eventFilter로만 정밀 처리합니다.
        """
        pass

    def _get_pos_from_vtk(self, iren):
        try:
            if iren:
                vx, vy = iren.GetEventPosition()
                win_h = self.target.height()
                if win_h <= 0 and hasattr(iren, 'GetRenderWindow'):
                    win_h = iren.GetRenderWindow().GetSize()[1]
                return QPoint(vx, max(0, win_h - vy))
        except Exception:
            pass
        return self._get_pos_from_cursor()

    def _get_pos_from_cursor(self):
        try:
            from PySide6.QtGui import QCursor
            gpos = QCursor.pos()
            return self.target.mapFromGlobal(gpos)
        except Exception:
            return QPoint(0, 0)

    def get_event_pos(self, event, obj=None):
        try:
            if hasattr(event, 'position'):
                pos = event.position().toPoint()
            elif hasattr(event, 'pos'):
                pos = event.pos()
            else:
                pos = None
            if pos is not None:
                if obj is not None and obj != self.target and hasattr(obj, 'mapTo'):
                    pos = obj.mapTo(self.target, pos)
                return pos
        except Exception:
            pass
        return self._get_pos_from_cursor()

    def set_active(self, active):
        self.is_active = active
        self.is_dragging = False
        if not active:
            self.vtk_overlay.hide()
            self.origin = QPoint()
        else:
            self.install_all_filters()
            self.bind_vtk_observers()

    def on_press_event(self, pos=None):
        if not self.is_active:
            return
        if pos is None:
            pos = self._get_pos_from_cursor()
        self.origin = pos
        self.is_dragging = True
        self.vtk_overlay.update_box(self.origin, self.origin)

    def on_move_event(self, cur_pos=None):
        if not self.is_active or not self.is_dragging or self.origin.isNull():
            return
        if cur_pos is None:
            cur_pos = self._get_pos_from_cursor()
        dx = abs(cur_pos.x() - self.origin.x())
        dy = abs(cur_pos.y() - self.origin.y())
        if dx >= 2 or dy >= 2:
            self.vtk_overlay.update_box(self.origin, cur_pos)

    def on_release_event(self, cur_pos=None):
        if not self.is_active or not self.is_dragging:
            self.vtk_overlay.hide()
            return
        if cur_pos is None:
            cur_pos = self._get_pos_from_cursor()
        self.release_pos = cur_pos
        self.last_rect = QRect(self.origin, self.release_pos).normalized()
        self.is_left_to_right = (self.origin.x() <= self.release_pos.x())
        self.vtk_overlay.hide()
        self.is_dragging = False
        
        if (self.release_pos - self.origin).manhattanLength() < 5:
            if self.clear_callback:
                from PySide6.QtCore import QTimer
                QTimer.singleShot(0, self.clear_callback)
        self.origin = QPoint()

    def eventFilter(self, obj, event):
        if self.is_active:
            from PySide6.QtGui import QGuiApplication
            t = event.type()
            if t in (QEvent.MouseButtonPress, QEvent.MouseButtonDblClick) and event.button() == Qt.LeftButton:
                pos = self.get_event_pos(event, obj)
                self.on_press_event(pos)
                return False 
            elif t == QEvent.MouseMove:
                if self.is_dragging or (QGuiApplication.mouseButtons() & Qt.LeftButton):
                    pos = self.get_event_pos(event, obj)
                    if not self.is_dragging or self.origin.isNull():
                        self.on_press_event(pos)
                    self.on_move_event(pos)
                    return False
            elif t == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                pos = self.get_event_pos(event, obj)
                self.on_release_event(pos)
                return False
        return super().eventFilter(obj, event)

class SkpLoadLogDialog(QDialog):
    """스케치업 파일 파싱 및 로드 결과를 조금 큰 창에 정밀하게 보여주는 팝업 대화상자"""
    def __init__(self, file_path, parse_info, log_text, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📂 스케치업 파일 로드 리포트 (SketchUp Load Log)")
        self.resize(650, 520)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        
        # 1. 파일명 헤더
        header_lbl = QLabel(f"<b>📁 스케치업 파일:</b> {os.path.basename(file_path)}")
        header_lbl.setStyleSheet("font-size: 13px; color: #2C3E50;")
        layout.addWidget(header_lbl)
        
        # 2. 요약 메타데이터 파싱 정보 그룹박스
        summary_group = QGroupBox("📊 스케치업 파싱 정보 요약")
        summary_layout = QVBoxLayout(summary_group)
        
        ver_str = parse_info.get('skp_version', 'Unknown')
        layers_str = ", ".join(parse_info.get('layers', []))
        total_cnt = parse_info.get('total_parts', 0)
        
        lbl_ver = QLabel(f"<b>• SketchUp 버전:</b> {ver_str}")
        lbl_layers = QLabel(f"<b>• 발견된 레이어(Tags):</b> {layers_str}")
        lbl_parts = QLabel(f"<b>• 총 감지된 독립 그룹/부품:</b> <font color='#27AE60'><b>{total_cnt}개</b></font>")
        
        summary_layout.addWidget(lbl_ver)
        summary_layout.addWidget(lbl_layers)
        summary_layout.addWidget(lbl_parts)
        
        tag_counts = parse_info.get('tag_counts', {})
        if tag_counts:
            dist_str = " | ".join([f"{t}: {c}개" for t, c in tag_counts.items()])
            lbl_dist = QLabel(f"<b>• 레이어별 그룹 분포:</b> {dist_str}")
            lbl_dist.setWordWrap(True)
            summary_layout.addWidget(lbl_dist)
            
        layout.addWidget(summary_group)
        
        # 3. 상세 터미널 로그 뷰어
        log_label = QLabel("<b>📝 스케치업 로딩 로그 상세 (Terminal Log):</b>")
        layout.addWidget(log_label)
        
        self.txt_log = QTextEdit()
        self.txt_log.setReadOnly(True)
        self.txt_log.setFontFamily("Consolas, Courier New, Monospace")
        self.txt_log.setStyleSheet("background-color: #1E1E1E; color: #00FF66; font-size: 12px; border-radius: 4px; padding: 8px;")
        self.txt_log.setText(log_text)
        layout.addWidget(self.txt_log)
        
        # 4. 하단 버튼
        btn_layout = QHBoxLayout()
        btn_copy = QPushButton("📋 로그 복사 (Copy Log)")
        btn_close = QPushButton("확인 (Close)")
        btn_copy.clicked.connect(self.copy_log)
        btn_close.clicked.connect(self.accept)
        
        btn_layout.addWidget(btn_copy)
        btn_layout.addStretch()
        btn_layout.addWidget(btn_close)
        layout.addLayout(btn_layout)
        
    def copy_log(self):
        clipboard = QApplication.clipboard()
        clipboard.setText(self.txt_log.toPlainText())
        QMessageBox.information(self, "복사 완료", "클립보드에 로딩 로그가 복사되었습니다.")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.base_title = "AI-NanoStudio - Final Integration"
        self.init_ui()
        self.update_window_title()
        self.resize(1280, 720)

    def set_titlebar_gray(self):
        """Windows 10/11 네이티브 타이틀바 캡션 색상을 세련된 짙은 회색(#22252B)으로 적용"""
        import sys, ctypes
        if sys.platform == "win32":
            try:
                hwnd = int(self.winId())
                # DWMWA_CAPTION_COLOR = 35, DWMWA_TEXT_COLOR = 36
                DWMWA_CAPTION_COLOR = 35
                DWMWA_TEXT_COLOR = 36
                # COLORREF: 0x00BBGGRR (짙은 회색 #22252B -> R: 0x22, G: 0x25, B: 0x2B -> 0x002B2522)
                color = ctypes.c_uint(0x002B2522)
                text_color = ctypes.c_uint(0x00FFFFFF)
                ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    hwnd, DWMWA_CAPTION_COLOR, ctypes.byref(color), ctypes.sizeof(color)
                )
                ctypes.windll.dwmapi.DwmSetWindowAttribute(
                    hwnd, DWMWA_TEXT_COLOR, ctypes.byref(text_color), ctypes.sizeof(text_color)
                )
            except Exception as e:
                print(f"타이틀바 색상 설정 오류: {e}")

    def showEvent(self, event):
        super().showEvent(event)
        self.set_titlebar_gray()

    def update_window_title(self, file_path=None):
        """윈도우 타이틀바에 프로그램명과 불러오기/저장된 프로젝트 파일명 표시"""
        if file_path:
            file_name = os.path.basename(file_path)
            self.setWindowTitle(f"{self.base_title} - [{file_name}]")
        else:
            self.setWindowTitle(self.base_title)
        
        if os.path.exists(ICON_PATH):
            self.setWindowIcon(QIcon(ICON_PATH))

    def init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # 좌측 컨트롤 패널
        left_panel = QWidget()
        left_panel.setFixedWidth(300)
        left_layout = QVBoxLayout(left_panel)
        
        # 상단 브랜드 로고 패널
        brand_layout = QHBoxLayout()
        if os.path.exists(ICON_PATH):
            brand_logo = QLabel()
            brand_logo.setPixmap(QPixmap(ICON_PATH).scaled(26, 26, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            brand_layout.addWidget(brand_logo)
        brand_title = QLabel("AI-NanoStudio")
        brand_title.setStyleSheet("font-size: 15px; font-weight: bold; color: #E0E0E0;")
        brand_layout.addWidget(brand_title)
        brand_layout.addStretch()
        left_layout.addLayout(brand_layout)
        
        self.btn_load_skp = QPushButton("Load .SKP File")
        self.btn_load_obj = QPushButton("Load .OBJ File (Fallback)")
        self.btn_new_project = QPushButton("New Project")
        self.btn_save_anf = QPushButton("Save Project (.anf)")
        self.btn_load_anf = QPushButton("Load Project (.anf)")
        self.btn_export_obj = QPushButton("Export Project (.obj)")
        self.btn_auto_fix = QPushButton("Auto-Fix Mesh (Clean & Smooth)")
        self.btn_auto_fix.setStyleSheet("font-weight: bold; color: white; background-color: #2196F3; padding: 5px;")
        
        self.btn_orient_normals = QPushButton("📐 모든 면 앞면 통일 (Orient Frontfaces)")
        self.btn_orient_normals.setStyleSheet("font-weight: bold; color: white; background-color: #2E7D32; padding: 6px;")
        self.btn_orient_normals.setToolTip("스케치업/CAD 모델에서 뒤집힌 면(Backface)을 외부 앞면(Frontface) 방향으로 100% 일괄 자동 정렬합니다.")
        
        left_layout.addWidget(self.btn_load_skp)
        left_layout.addWidget(self.btn_load_obj)
        left_layout.addWidget(self.btn_new_project)
        left_layout.addWidget(self.btn_save_anf)
        left_layout.addWidget(self.btn_load_anf)
        left_layout.addWidget(self.btn_export_obj)
        left_layout.addWidget(self.btn_auto_fix)
        left_layout.addWidget(self.btn_orient_normals)
        
        # QTabWidget 추가 (높이를 내용물에 맞게 고정)
        self.tabs = QTabWidget()
        self.tabs.setFixedHeight(300)
        left_layout.addWidget(self.tabs)
        
        tab_icon = QIcon(ICON_PATH) if os.path.exists(ICON_PATH) else QIcon()
        
        # Tab 1: 건축/인테리어 (Part 1)
        self.tab_arch = QWidget()
        self.tab_arch_layout = QVBoxLayout(self.tab_arch)
        self.tab_arch_layout.addWidget(QLabel("건축/인테리어 렌더링"))
        self.btn_render_arch = QPushButton("오프스크린 PBR 실사 렌더링")
        self.btn_render_arch.setStyleSheet("font-weight: bold; color: white; background-color: #d84315; padding: 10px;")
        self.tab_arch_layout.addWidget(self.btn_render_arch)
        
        # ── 시네마틱 디렉터 섹션 (FR-08, FR-09, FR-10) ──
        self.tab_arch_layout.addWidget(QLabel(""))  # 간격
        lbl_director = QLabel("🎬 시네마틱 디렉터 (Veo/Flow)")
        lbl_director.setStyleSheet("font-weight: bold; font-size: 12px; color: #38bdf8;")
        self.tab_arch_layout.addWidget(lbl_director)
        
        # 카메라 A/B 등록 버튼
        cam_btn_layout = QHBoxLayout()
        self.btn_set_cam_a = QPushButton("1️⃣ 시작점(A)")
        self.btn_set_cam_b = QPushButton("2️⃣ 종료점(B)")
        self.btn_set_cam_a.setToolTip("현재 뷰포트 카메라 앵글을 시작점(A)으로 등록합니다.")
        self.btn_set_cam_b.setToolTip("현재 뷰포트 카메라 앵글을 종료점(B)으로 등록합니다.")
        cam_btn_layout.addWidget(self.btn_set_cam_a)
        cam_btn_layout.addWidget(self.btn_set_cam_b)
        self.tab_arch_layout.addLayout(cam_btn_layout)
        
        # 카메라 등록 상태 표시
        self.lbl_cam_status = QLabel("시작점(A): ⬜  |  종료점(B): ⬜")
        self.lbl_cam_status.setStyleSheet("font-size: 11px; color: #94a3b8;")
        self.tab_arch_layout.addWidget(self.lbl_cam_status)
        
        # 씬 콘셉트 입력
        self.input_scene_concept = QLineEdit()
        self.input_scene_concept.setPlaceholderText("씬 콘셉트 (예: 일몰 황금빛 모던 콘크리트 빌라)")
        self.tab_arch_layout.addWidget(self.input_scene_concept)
        
        # Veo 프롬프트 생성 & 콘티 PDF 추출 버튼
        self.btn_generate_veo = QPushButton("🚀 Veo 모션 프롬프트 생성")
        self.btn_generate_veo.setStyleSheet("font-weight: bold; color: white; background-color: #7c3aed; padding: 6px;")
        self.btn_generate_veo.setToolTip("Gemini AI가 카메라 궤적을 분석하여 Google Flow/Veo 전용 프롬프트를 자동 생성합니다.")
        self.tab_arch_layout.addWidget(self.btn_generate_veo)
        
        self.btn_export_conti = QPushButton("📄 콘티 PDF + 프레임 일괄 추출")
        self.btn_export_conti.setStyleSheet("font-weight: bold; color: white; background-color: #0284c7; padding: 6px;")
        self.btn_export_conti.setToolTip("First/Last Frame PNG, 모션 프롬프트, 콘티 기획서 PDF를 한 번에 추출합니다.")
        self.tab_arch_layout.addWidget(self.btn_export_conti)
        
        # Veo 프롬프트 결과 표시
        self.lbl_veo_result = QLabel("")
        self.lbl_veo_result.setWordWrap(True)
        self.lbl_veo_result.setStyleSheet("font-size: 10px; color: #e2e8f0; background-color: #1e1b2e; border: 1px solid #332b4a; border-radius: 4px; padding: 4px;")
        self.lbl_veo_result.setMinimumHeight(40)
        self.tab_arch_layout.addWidget(self.lbl_veo_result)
        
        self.tab_arch_layout.addStretch()
        if not tab_icon.isNull():
            self.tabs.addTab(self.tab_arch, tab_icon, "Part 1: 렌더링")
        else:
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
        self.btn_export_glb = QPushButton("🎮 GLB 게임 에셋 내보내기")
        self.btn_export_glb.setStyleSheet("font-weight: bold; color: white; background-color: #16a34a; padding: 8px;")
        self.btn_export_glb.setToolTip("현재 모델을 텍스처 내장형 .glb 파일로 내보냅니다 (Unity/Unreal 즉시 호환).")
        self.tab_game_layout.addWidget(self.btn_export_glb)
        self.tab_game_layout.addStretch()
        if not tab_icon.isNull():
            self.tabs.addTab(self.tab_game, tab_icon, "Part 2: 게임 에셋")
        else:
            self.tabs.addTab(self.tab_game, "Part 2: 게임 에셋")
        
        # 아웃라이너(계층 구조) 표시 영역
        outliner_header_layout = QHBoxLayout()
        if os.path.exists(ICON_PATH):
            outliner_logo = QLabel()
            outliner_logo.setPixmap(QPixmap(ICON_PATH).scaled(16, 16, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            outliner_header_layout.addWidget(outliner_logo)
        self.outliner_label = QLabel("객체 계층 구조 (Outliner):")
        outliner_header_layout.addWidget(self.outliner_label)
        outliner_header_layout.addStretch()
        left_layout.addLayout(outliner_header_layout)
        
        # 계층 추가/관리 및 전체 선택/해제 버튼
        self.outliner_btn_layout = QHBoxLayout()
        self.btn_add_node = QPushButton("➕ 계층 추가")
        self.btn_delete_node = QPushButton("🗑️ 계층 삭제")
        self.btn_select_all = QPushButton("☑️ 전체 선택")
        self.btn_deselect_all = QPushButton("☐ 전체 해제")
        
        self.btn_add_node.setToolTip("새로운 사용자 정의 계층 노드를 추가합니다.")
        self.btn_delete_node.setToolTip("선택한 계층(Component 또는 부품 그룹)을 아웃라이너 및 3D 뷰포트에서 삭제합니다.")
        self.btn_select_all.setToolTip("아웃라이너의 모든 부품 항목을 선택(체크)합니다.")
        self.btn_deselect_all.setToolTip("아웃라이너의 모든 부품 항목을 해제(체크해제)합니다.")
        
        self.btn_add_node.clicked.connect(self.add_custom_node)
        self.btn_delete_node.clicked.connect(self.delete_selected_hierarchy_node)
        self.btn_select_all.clicked.connect(lambda: self.set_all_tree_items_check_state(True))
        self.btn_deselect_all.clicked.connect(lambda: self.set_all_tree_items_check_state(False))
        
        self.outliner_btn_layout.addWidget(self.btn_add_node)
        self.outliner_btn_layout.addWidget(self.btn_delete_node)
        self.outliner_btn_layout.addWidget(self.btn_select_all)
        self.outliner_btn_layout.addWidget(self.btn_deselect_all)
        left_layout.addLayout(self.outliner_btn_layout)

        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabels(["부품명"])
        self.tree_widget.setEditTriggers(QAbstractItemView.DoubleClicked | QAbstractItemView.EditKeyPressed | QAbstractItemView.SelectedClicked)
        self.tree_widget.itemDoubleClicked.connect(self.on_tree_item_double_clicked)
        self.tree_widget.itemChanged.connect(self.on_tree_item_changed)
        self.tree_widget.itemClicked.connect(self.on_tree_item_clicked)
        self.tree_widget.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree_widget.customContextMenuRequested.connect(self.on_tree_context_menu)
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
        
        # ViewportWidget 내부의 single_layout 하단에 2by10 원형 재질 슬롯 패널 설치
        self.mat_slot_panel = MaterialSlotPanel(apply_callback=self.on_apply_material_slot)
        self.viewport.single_layout.addWidget(self.mat_slot_panel, stretch=0)
        
        main_layout.addWidget(left_panel)
        main_layout.addWidget(self.viewport, stretch=1)
        
        # 매니저 및 데이터 초기화
        self.camera_manager = CameraManager(self.viewport.plotter_single)
        self.cinematic_director = CinematicDirector()
        self.uv_unwrapper = UVUnwrapper()
        self.loaded_actors = {}
        self.atlas = None
        self.picked_mesh = None
        
        # 이벤트 연결
        self.btn_load_skp.clicked.connect(self.on_load_skp)
        self.btn_load_obj.clicked.connect(self.on_load_obj)
        self.btn_new_project.clicked.connect(self.on_new_project)
        self.btn_save_anf.clicked.connect(self.on_save_anf)
        self.btn_load_anf.clicked.connect(self.on_load_anf)
        self.btn_export_obj.clicked.connect(self.on_export_obj)
        self.btn_auto_fix.clicked.connect(self.on_btn_auto_fix)
        self.btn_orient_normals.clicked.connect(lambda: self.orient_all_frontfaces(silent=False))
        
        self.btn_render_arch.clicked.connect(self.on_render_arch)
        self.btn_unwrap.clicked.connect(self.on_unwrap_and_bake)
        self.btn_ai_texture.clicked.connect(self.on_ai_texture)
        self.btn_load_hdri.clicked.connect(self.on_load_hdri)
        self.btn_generate_hdri.clicked.connect(self.on_generate_hdri)
        self.btn_apply_pbr.clicked.connect(self.on_apply_pbr)
        self.btn_export_glb.clicked.connect(self.on_export_glb)
        
        # 시네마틱 디렉터 이벤트 연결 (FR-08, FR-09, FR-10)
        self.btn_set_cam_a.clicked.connect(lambda: self.on_set_camera_keyframe('A'))
        self.btn_set_cam_b.clicked.connect(lambda: self.on_set_camera_keyframe('B'))
        self.btn_generate_veo.clicked.connect(self.on_generate_veo_prompt)
        self.btn_export_conti.clicked.connect(self.on_export_conti_pdf)
        
        self.btn_pick_p.clicked.connect(self.on_btn_pick_p)
        self.btn_pick_r.clicked.connect(self.on_btn_pick_r)
        self.btn_clear_pick.clicked.connect(self.on_btn_clear_pick)
        self.btn_assign.clicked.connect(self.on_btn_assign)
        self.combo_display_mode.currentIndexChanged.connect(self.on_display_mode_changed)
        
        # 마퀴(Marquee) 사각형 영역 선택 필터 사전 준비
        self.ensure_rubber_band_filter()
        
        # 자동 로드 비활성화 (빈 환경으로 시작)
        # auto_load_path = "3d_model/car.skp"
        # if os.path.exists(auto_load_path):
        #     self.load_skp_file(auto_load_path)

    def on_display_mode_changed(self, index):
        show_edges = (index == 0) # 0: 폴리곤, 1: 솔리드
        self.viewport.set_display_mode(show_edges)

    def load_skp_file(self, file_path):
        log_action("LOAD_SKP_START", f"파일 경로: {file_path}")
        start_time = time.time()
        
        # 프로그레스 대화상자 생성 (UI 멈춤/Hang 방지 및 실시간 파싱 진행률 표시)
        progress = None
        try:
            progress = QProgressDialog("SKP 3D 모델 데이터 파싱 및 파이프라인 구성 중...", None, 0, 100, self)
            progress.setWindowTitle("SKP 파일 로딩 중")
            progress.setWindowModality(Qt.NonModal)
            progress.setMinimumDuration(0)
            progress.setValue(5)
            progress.show()
            QApplication.processEvents()
        except Exception:
            pass
        
        def update_progress(val, text=""):
            elapsed_sec = time.time() - start_time
            time_str = f" [소요시간: {elapsed_sec:.1f}초]"
            if progress:
                try:
                    progress.setValue(val)
                    if text:
                        progress.setLabelText(f"{text}{time_str}")
                    else:
                        progress.setLabelText(f"SKP 3D 데이터 처리 중...{time_str}")
                    progress.repaint()
                except Exception:
                    pass
            try:
                QApplication.processEvents()
            except Exception:
                pass
            
        update_progress(10, "openskp 데이터 구조 파싱 시작...")
        
        # openskp 파싱 및 상세 정보 수집
        actors, parse_info = load_skp_to_pyvista(file_path, return_info=True, progress_callback=update_progress)
        
        if not actors:
            if progress:
                try:
                    progress.close()
                except Exception:
                    pass
            log_action("LOAD_SKP_FAIL", f"로드 실패: {file_path}")
            QMessageBox.warning(self, "Load Error", "SKP 파일을 로드하지 못했습니다. openskp 설치 상태를 확인하세요.")
            return
            
        sample_keys = list(actors.keys())[:5]
        log_action("LOAD_SKP_SUCCESS", f"로드 완료 | 총 {len(actors)}개 부품 감지: {sample_keys}...")
        self.loaded_actors = actors
        
        update_progress(75, f"뷰포트 3D 씬 및 아웃라이너 배치 중... (총 {len(actors)}개 부품)")
        self.populate_scene(actors, progress_callback=update_progress)
        
        update_progress(100, "로딩 완료! 3D 뷰포트 화면 갱신 중...")
        if progress:
            try:
                progress.setValue(100)
                progress.close()
                progress.deleteLater()
            except Exception:
                pass
        QApplication.processEvents()
        
        self.update_window_title(file_path)
        
        # 뷰포트 카메라 및 3D 화면 즉시 강제 렌더링
        if hasattr(self, 'part_meshes') and self.part_meshes:
            overall_b = [float('inf'), float('-inf'), float('inf'), float('-inf'), float('inf'), float('-inf')]
            for _, m in self.part_meshes.items():
                if m and m.n_points > 0:
                    b = m.bounds
                    overall_b[0] = min(overall_b[0], b[0])
                    overall_b[1] = max(overall_b[1], b[1])
                    overall_b[2] = min(overall_b[2], b[2])
                    overall_b[3] = max(overall_b[3], b[3])
                    overall_b[4] = min(overall_b[4], b[4])
                    overall_b[5] = max(overall_b[5], b[5])
            if overall_b[0] != float('inf'):
                self.viewport.reset_camera(bounds=overall_b)
            else:
                self.viewport.reset_camera()
        else:
            self.viewport.reset_camera()
            
        self.viewport.render_active()
        QApplication.processEvents()
        
        # 로그 대화상자에 표시할 상세 텍스트 빌드
        from datetime import datetime as dt
        now_str = dt.now().strftime("%H:%M:%S.%f")[:-3]
        
        log_text = (
            f"SketchUp 버전: {{{parse_info.get('skp_version', 'Unknown')}}}\n"
            f"발견된 레이어: {parse_info.get('layers', [])}\n"
            f"[{now_str}] [INFO] [ACTION] LOAD_SKP_SUCCESS | 로드 완료 | 총 {len(actors)}개 부품 감지: {sample_keys}...\n\n"
            f"=== 레이어별 세부 그룹 수 분포 ===\n"
        )
        for tag, count in parse_info.get('tag_counts', {}).items():
            log_text += f"  • {tag}: {count} Groups\n"
            
        log_text += f"\n=== 로드된 샘플 부품 명칭 리스트 (상위 30개) ===\n"
        for name in list(actors.keys())[:30]:
            log_text += f"  - {name}\n"
        if len(actors) > 30:
            log_text += f"  ... 외 {len(actors) - 30}개 부품 추가 로드 완료\n"
            
        # 로드 로그 대화상자 표시 (비모달 show로 메인 뷰포트 3D 렌더링 차단 해제)
        if self.isVisible():
            self.load_log_dlg = SkpLoadLogDialog(file_path, parse_info, log_text, self)
            self.load_log_dlg.show()
        
    def on_load_skp(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open SketchUp File", "", "SketchUp Files (*.skp)")
        if file_path:
            self.load_skp_file(file_path)
            
    def on_load_obj(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open OBJ File", "", "OBJ Files (*.obj)")
        if file_path:
            log_action("LOAD_OBJ_START", f"파일 경로: {file_path}")
            actors = load_obj_to_pyvista(file_path)
            if not actors:
                log_action("LOAD_OBJ_FAIL", f"로드 실패: {file_path}")
                QMessageBox.warning(self, "Load Error", "OBJ 파일을 로드하지 못했습니다.")
                return
            log_action("LOAD_OBJ_SUCCESS", f"로드 완료 | 총 {len(actors)}개 부품")
            self.loaded_actors = actors
            self.populate_scene(actors)
            self.update_window_title(file_path)
            
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

    def populate_scene(self, actors_dict, progress_callback=None):
        """불러온 데이터를 화면과 아웃라이너에 세팅 (고속 최적화 적용)"""
        self.tree_widget.blockSignals(True)
        self.tree_widget.setUpdatesEnabled(False)
        self.tree_widget.clear()
        
        # 뷰포트 rendering 및 업데이트 억제 (배치 속도 극대화 & 응답 없음 방지)
        for p in self.viewport.plotters:
            p.suppress_rendering = True
        self.viewport.setUpdatesEnabled(False)
        
        # 랜덤 색상 생성용 헬퍼 함수
        def random_color():
            return [random.uniform(0.3, 0.9), random.uniform(0.3, 0.9), random.uniform(0.3, 0.9)]
            
        # 기존 메시 지우기 및 뷰포트 초기 세팅(그리드, 축 등)
        self.viewport.clear_and_setup()
            
        self.part_colors = {}
        self.part_materials = {}
        self.last_applied_material = None
        self.frontface_locked_parts = set()
        self.part_meshes = actors_dict
        
        # 전체 바운딩 박스 크기 계산 (치수 정보 표시용)
        overall_bounds = [float('inf'), float('-inf'), float('inf'), float('-inf'), float('inf'), float('-inf')]
        
        # 스케치업 Tags(레이어) 부모 노드 추적용 및 아이콘 캐시
        tag_items = {}
        icon_cache = {}
        
        total_parts = len(actors_dict)
        # 각 메시 추가 및 Tags(레이어) 계층구조 생성
        for idx_count, (name, mesh) in enumerate(actors_dict.items()):
            if progress_callback and total_parts > 0 and idx_count % 100 == 0:
                pct = 75 + int((idx_count / total_parts) * 20) # 75% ~ 95%
                progress_callback(pct, f"3D 메쉬 및 트리 노드 구성 중... ({idx_count}/{total_parts})")

            b = mesh.bounds
            overall_bounds[0] = min(overall_bounds[0], b[0])
            overall_bounds[1] = max(overall_bounds[1], b[1])
            overall_bounds[2] = min(overall_bounds[2], b[2])
            overall_bounds[3] = max(overall_bounds[3], b[3])
            overall_bounds[4] = min(overall_bounds[4], b[4])
            overall_bounds[5] = max(overall_bounds[5], b[5])
            
            # 스케치업 모델 로딩 시 각각의 Component 및 ROOT_MODEL_ROOT 초기 기본 색상을 옅은 회색으로 통일
            default_light_gray = [0.82, 0.82, 0.82]
            part_color = default_light_gray
            self.part_colors[name] = part_color
            
            # 픽스맵 아이콘 생성 및 캐싱
            rgb_key = (int(part_color[0]*255), int(part_color[1]*255), int(part_color[2]*255))
            if rgb_key not in icon_cache:
                pixmap = QPixmap(16, 16)
                pixmap.fill(QColor(*rgb_key))
                icon_cache[rgb_key] = QIcon(pixmap)
            icon = icon_cache[rgb_key]
            
            # 이름에서 Tag(레이어)와 서브부품 분리 (예: "기초부/기둥")
            if "/" in name:
                tag_name, sub_name = name.split("/", 1)
                
                # ROOT_MODEL_ROOT 등 모델 추가 요소는 객체 계층 구조에서 '추가 부품' 그룹으로 분류
                if "ROOT_MODEL" in name or "ROOT_MODEL" in sub_name or sub_name.startswith("ROOT_") or sub_name.startswith("MODEL_") or name.endswith("/ROOT") or name.endswith("/MODEL"):
                    tag_name = "추가 부품"

                # 부모 Tag 노드가 없으면 생성 (스케치업 Tags 패널 항목)
                if tag_name not in tag_items:
                    parent_item = QTreeWidgetItem(self.tree_widget, [tag_name])
                    parent_item.setFlags(parent_item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEditable)
                    parent_item.setCheckState(0, Qt.Checked)
                    parent_item.setData(0, Qt.UserRole, tag_name)
                    parent_item.setData(0, Qt.UserRole + 2, Qt.Checked)
                    parent_item.setIcon(0, icon)
                    tag_items[tag_name] = parent_item
                else:
                    parent_item = tag_items[tag_name]
                    
                # 자식 부품 노드 추가
                item = QTreeWidgetItem(parent_item, [sub_name])
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEditable)
                item.setCheckState(0, Qt.Checked)
                item.setData(0, Qt.UserRole, name)
                item.setData(0, Qt.UserRole + 2, Qt.Checked)
                item.setIcon(0, icon)
            else:
                if "ROOT_MODEL" in name or name.startswith("ROOT_") or name.startswith("MODEL_"):
                    tag_name = "추가 부품"
                    if tag_name not in tag_items:
                        parent_item = QTreeWidgetItem(self.tree_widget, [tag_name])
                        parent_item.setFlags(parent_item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEditable)
                        parent_item.setCheckState(0, Qt.Checked)
                        parent_item.setData(0, Qt.UserRole, tag_name)
                        parent_item.setData(0, Qt.UserRole + 2, Qt.Checked)
                        parent_item.setIcon(0, icon)
                        tag_items[tag_name] = parent_item
                    else:
                        parent_item = tag_items[tag_name]
                    item = QTreeWidgetItem(parent_item, [name])
                else:
                    item = QTreeWidgetItem(self.tree_widget, [name])
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEditable)
                item.setCheckState(0, Qt.Checked)
                item.setData(0, Qt.UserRole, name)
                item.setData(0, Qt.UserRole + 2, Qt.Checked)
                item.setIcon(0, icon)
            
            self.viewport.add_mesh_to_all(mesh, name=name, color=part_color)

        # 각 Tag(레이어) 부모 노드 텍스트에 그룹 개수(예: "기주부 (11 Groups)") 업데이트
        for tag_name, parent_item in tag_items.items():
            child_cnt = parent_item.childCount()
            if child_cnt > 0:
                parent_item.setText(0, f"{tag_name} ({child_cnt} Groups)")

        self.tree_widget.expandAll()
        self.update_dimensions_text()
        
        # 뷰포트 rendering 억제 해제 및 카메라 리셋
        for p in self.viewport.plotters:
            p.suppress_rendering = False
        self.viewport.setUpdatesEnabled(True)
        
        if overall_bounds[0] != float('inf'):
            self.viewport.setup_pbr_lighting(bounds=overall_bounds)
            self.viewport.reset_camera(bounds=overall_bounds)
        else:
            self.viewport.reset_camera()

        self.viewport.set_display_render_mode(getattr(self.viewport, 'current_render_mode', 'material_preview'))
        self.viewport.render_active()
        
        self.tree_widget.setUpdatesEnabled(True)
        self.tree_widget.blockSignals(False)
            
        # UI 업데이트
        self.btn_ai_texture.setEnabled(True)
        self.btn_unwrap.setEnabled(True)

        # 폴리곤(면) 분할 기능을 위한 셀 피킹(Cell Picking) 비활성화 (버튼을 눌러야 활성화됨)
        self.picked_mesh = None

    def ensure_rubber_band_filter(self):
        """마퀴(Marquee) 사각형 영역 선택 필터 인스턴스 보장"""
        if not hasattr(self, 'rubber_band_filter') or self.rubber_band_filter is None:
            self.rubber_band_filter = RubberBandFilter(
                self.viewport.plotter_single.interactor, 
                clear_callback=self.on_btn_clear_pick,
                plotter=self.viewport.plotter_single
            )
        return self.rubber_band_filter

    def on_btn_pick_r(self):
        """박스 면 선택 모드 토글 (R 키 / 관통 선택)"""
        if not self.btn_pick_r.isChecked():
            self.on_btn_clear_pick()
            if hasattr(self, 'rubber_band_filter') and self.rubber_band_filter:
                self.rubber_band_filter.set_active(False)
            return
            
        self.btn_pick_p.setChecked(False)

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
            
            # 마퀴 필터 보장 및 활성화
            rb = self.ensure_rubber_band_filter()
            rb.set_active(True)

            style = self.viewport.plotter_single.iren.get_interactor_style()
            if hasattr(style, 'StartSelect'):
                def force_start_select(obj, event):
                    obj.StartSelect()
                style.AddObserver("LeftButtonPressEvent", force_start_select, 10.0)
        except Exception as e:
            print(f"피킹 활성화 오류: {e}")

    def on_btn_pick_p(self):
        """단일/표면 면 선택 모드 토글 (P 키 / 표면 선택)"""
        if not self.btn_pick_p.isChecked():
            self.on_btn_clear_pick()
            return
            
        self.btn_pick_r.setChecked(False)
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
            
            # 마퀴 필터 보장 및 활성화
            rb = self.ensure_rubber_band_filter()
            rb.set_active(True)

            style = self.viewport.plotter_single.iren.get_interactor_style()
            if hasattr(style, 'StartSelect'):
                def force_start_select(obj, event):
                    obj.StartSelect()
                style.AddObserver("LeftButtonPressEvent", force_start_select, 10.0)
        except Exception as e:
            print(f"피킹 활성화 오류: {e}")

    def on_btn_clear_pick(self):
        """피킹 모드 및 선택된 하이라이트 해제"""
        try:
            self.btn_pick_p.setChecked(False)
            self.btn_pick_r.setChecked(False)
            if hasattr(self, 'rubber_band_filter') and self.rubber_band_filter:
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

    def filter_mesh_enclosed_in_rect(self, mesh, rect):
        """
        Left-to-Right 마퀴 영역 선택 시, 2D 스크린 직사각형(rect) 내부에
        100% 완전히 포함(Enclosed)된 면(Cell)만 추출하여 반환합니다.
        """
        import numpy as np
        import pyvista as pv

        if mesh is None or mesh.n_cells == 0 or rect is None or rect.isEmpty():
            return mesh

        plotter = getattr(self.viewport, 'plotter_single', None)
        if plotter is None:
            return mesh

        try:
            ren = plotter.renderer
            cam = ren.GetActiveCamera()
            aspect = ren.GetTiledAspectRatio()
            mat = cam.GetCompositeProjectionTransformMatrix(aspect, -1, 1)

            M = np.zeros((4, 4))
            for i in range(4):
                for j in range(4):
                    M[i, j] = mat.GetElement(i, j)

            win_size = plotter.window_size
            W, H = win_size[0], win_size[1]
            if W <= 0 or H <= 0:
                return mesh

            # The marquee is stored in Qt widget coordinates while projection
            # results use the render-window pixel coordinates.
            viewport_widget = getattr(plotter, 'interactor', None)
            widget_w = viewport_widget.width() if viewport_widget is not None else W
            widget_h = viewport_widget.height() if viewport_widget is not None else H
            scale_x = W / max(widget_w, 1)
            scale_y = H / max(widget_h, 1)

            pts = mesh.points
            if len(pts) == 0:
                return mesh

            pts4 = np.hstack([pts, np.ones((len(pts), 1))])
            clip = pts4 @ M.T

            w = clip[:, 3]
            w_safe = np.where(np.abs(w) < 1e-7, 1e-7, w)

            x_ndc = clip[:, 0] / w_safe
            y_ndc = clip[:, 1] / w_safe
            z_ndc = clip[:, 2] / w_safe

            screen_x = (x_ndc + 1.0) * 0.5 * W
            screen_y = (1.0 - y_ndc) * 0.5 * H

            margin = 5
            rect_x1 = (min(rect.x(), rect.x() + rect.width()) - margin) * scale_x
            rect_x2 = (max(rect.x(), rect.x() + rect.width()) + margin) * scale_x
            rect_y1 = (min(rect.y(), rect.y() + rect.height()) - margin) * scale_y
            rect_y2 = (max(rect.y(), rect.y() + rect.height()) + margin) * scale_y

            # 2D 스크린 rect 내부 + Z 화면 앞/뒤 클리핑 평면 내부 여부 판단
            is_pt_inside = (
                (screen_x >= rect_x1) & (screen_x <= rect_x2) &
                (screen_y >= rect_y1) & (screen_y <= rect_y2) &
                (z_ndc >= -2.0) & (z_ndc <= 2.0)
            )

            enclosed_cell_indices = []
            for cell_idx in range(mesh.n_cells):
                cell_pt_ids = mesh.get_cell(cell_idx).point_ids
                if len(cell_pt_ids) > 0 and np.any(is_pt_inside[cell_pt_ids]):
                    enclosed_cell_indices.append(cell_idx)

            if not enclosed_cell_indices:
                return pv.PolyData()
            elif len(enclosed_cell_indices) == mesh.n_cells:
                return mesh
            else:
                sub_mesh = mesh.extract_cells(enclosed_cell_indices)
                if not isinstance(sub_mesh, pv.PolyData):
                    sub_mesh = sub_mesh.extract_surface(algorithm='dataset_surface')
                return sub_mesh
        except Exception as e:
            print(f"[Enclosed Selection Error]: {e}")
            return mesh

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

        # 1. VTK 하드웨어 셀렉터가 이미 정확히 검출한 표면 셀(new_merged)이 있다면 최우선 채택
        # 2. VTK 피커가 빈 메쉬를 반환한 경우에만 마퀴 사각형 영역(marquee_rect) 기반 fallback 투영 수행
        if new_merged is None or new_merged.n_cells == 0:
            marquee_rect = None
            if hasattr(self, 'rubber_band_filter') and self.rubber_band_filter.is_active:
                rb = self.rubber_band_filter
                if rb.last_rect is not None and not rb.last_rect.isEmpty() and rb.last_rect.width() >= 5 and rb.last_rect.height() >= 5:
                    marquee_rect = rb.last_rect

            if marquee_rect is not None:
                checked_names = self.get_checked_item_names()
                source_meshes = [
                    part_mesh for name, part_mesh in getattr(self, 'part_meshes', {}).items()
                    if (not checked_names or name in checked_names)
                    and part_mesh is not None and part_mesh.n_cells > 0
                ]
                if source_meshes:
                    fallback_merged = source_meshes[0] if len(source_meshes) == 1 else source_meshes[0].merge(source_meshes[1:])
                    new_merged = self.filter_mesh_enclosed_in_rect(fallback_merged, marquee_rect)

        # 계층구조상 체크된 오브젝트에 속하는 셀만 일차 필터링
        new_merged = self.filter_mesh_by_checked_objects(new_merged)
        
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
            
        # 최종 구성된 피킹 영역을 체크 표시된 계층 오브젝트로 한 번 더 엄격히 필터링
        self.picked_mesh = self.filter_mesh_by_checked_objects(final_mesh)
        self.update_picked_highlight()
    def on_btn_assign(self, target_item=None):
        """선택된 뷰포트 객체의 모든 면(전체)을 아웃라이너에서 선택 또는 우클릭한 계층 그룹으로 이동 및 등록"""
        import numpy as np
        import pyvista as pv
        from PySide6.QtWidgets import QMessageBox, QTreeWidgetItem, QInputDialog
        from PySide6.QtGui import QIcon, QPixmap, QColor
        from PySide6.QtCore import Qt

        if getattr(self, 'picked_mesh', None) is None:
            QMessageBox.information(self, "선택된 객체/면 없음", "먼저 뷰포트에서 이동할 3D 객체(면)를 선택해 주세요.")
            return

        # MultiBlock 대응
        if isinstance(self.picked_mesh, pv.MultiBlock):
            blocks = [b for b in self.picked_mesh if b is not None and b.n_cells > 0]
            if not blocks:
                QMessageBox.information(self, "선택된 객체/면 없음", "선택한 영역 내에 폴리곤이 없습니다.")
                return
            merged_picked = blocks[0].merge(blocks[1:]) if len(blocks) > 1 else blocks[0]
        else:
            merged_picked = self.picked_mesh

        if merged_picked is None or merged_picked.n_cells == 0:
            QMessageBox.information(self, "선택된 객체/면 없음", "선택한 영역 내에 유효한 폴리곤이 없습니다.")
            return

        # 1. 대상 계층 노드 지정 (아웃라이너 미선택 시 편리한 선택 팝업 제공)
        parent_group_item = None
        group_name = ""

        if target_item is not None:
            if target_item.parent() is not None:
                parent_group_item = target_item.parent()
            else:
                parent_group_item = target_item
        else:
            selected_items = self.tree_widget.selectedItems()
            if selected_items:
                sel = selected_items[0]
                if sel.parent() is not None:
                    parent_group_item = sel.parent()
                else:
                    parent_group_item = sel
            else:
                # 아웃라이너에서 사전 선택하지 않은 경우: 존재하는 계층 그룹 목록을 팝업으로 안내하여 즉시 이동 지원
                group_items = []
                root = self.tree_widget.invisibleRootItem()
                if root:
                    for i in range(root.childCount()):
                        top_node = root.child(i)
                        if top_node is not None:
                            g_tag = top_node.data(0, Qt.UserRole) or top_node.text(0).split(' (')[0].strip()
                            if g_tag:
                                group_items.append((g_tag, top_node))

                if not group_items:
                    QMessageBox.warning(self, "계층 미선택", "좌측 아웃라이너에서 할당받을 '계층 명칭(그룹)'을 먼저 선택해주세요.")
                    return

                group_display_names = [f"{g_name} ({node.text(0)})" if node.text(0) != g_name else g_name for g_name, node in group_items]
                chosen_display, ok = QInputDialog.getItem(
                    self,
                    "계층 그룹 선택",
                    "선택한 객체를 이동할 대상 계층 그룹을 선택하세요:",
                    group_display_names,
                    0,
                    False
                )
                if not ok or not chosen_display:
                    return

                chosen_idx = group_display_names.index(chosen_display)
                group_name, parent_group_item = group_items[chosen_idx]

        if parent_group_item is not None and not group_name:
            group_name = parent_group_item.data(0, Qt.UserRole)
            if not group_name:
                raw_text = parent_group_item.text(0)
                group_name = raw_text.split(" (")[0].strip()

        if not group_name or parent_group_item is None:
            QMessageBox.warning(self, "계층 미선택", "할당받을 유효한 계층 그룹을 찾지 못했습니다.")
            return

        picked_centers = merged_picked.cell_centers().points

        # 2. 이동할 객체 매핑 {source_name: [cell_indices]}
        moves = {name: [] for name in self.part_meshes.keys()}

        for name, mesh in self.part_meshes.items():
            if mesh is None or mesh.n_cells == 0:
                continue

            mesh_centers = mesh.cell_centers().points
            try:
                from scipy.spatial import cKDTree
                tree = cKDTree(mesh_centers)
                indices_list = tree.query_ball_point(picked_centers, r=0.001)
                for indices in indices_list:
                    if indices:
                        moves[name].extend(indices)
            except ImportError:
                for pc in picked_centers:
                    dists = np.linalg.norm(mesh_centers - pc, axis=1)
                    matched_indices = np.where(dists < 0.001)[0]
                    if len(matched_indices) > 0:
                        moves[name].extend(matched_indices.tolist())

        # 2-1. 1차 매칭 실패 시 오차 보정을 위해 반경을 확장(r=0.05)하여 재시도
        if not any(moves.values()):
            for name, mesh in self.part_meshes.items():
                if mesh is None or mesh.n_cells == 0:
                    continue
                mesh_centers = mesh.cell_centers().points
                try:
                    from scipy.spatial import cKDTree
                    tree = cKDTree(mesh_centers)
                    indices_list = tree.query_ball_point(picked_centers, r=0.05)
                    for indices in indices_list:
                        if indices:
                            moves[name].extend(indices)
                except Exception:
                    pass

        # 2-2. 최근접 거리(Nearest-Neighbor) fallback 매칭
        if not any(moves.values()):
            best_name = None
            best_dist = float('inf')
            for name, mesh in self.part_meshes.items():
                if mesh is None or mesh.n_cells == 0:
                    continue
                mesh_centers = mesh.cell_centers().points
                try:
                    from scipy.spatial import cKDTree
                    tree = cKDTree(mesh_centers)
                    dists, _ = tree.query(picked_centers, k=1)
                    mean_d = float(np.mean(dists))
                    if mean_d < best_dist and mean_d < 0.2: # 20cm 이내 최근접
                        best_dist = mean_d
                        best_name = name
                except Exception:
                    pass
            if best_name:
                moves[best_name] = [0]

        matched_sources = [name for name, idxs in moves.items() if len(idxs) > 0]
        if not matched_sources:
            self.on_btn_clear_pick()
            QMessageBox.information(self, "매칭 실패", "선택된 3D 객체와 일치하는 원본 부품 데이터를 찾지 못했습니다.")
            return

        # 이미 대상 그룹에 속해 있는지 확인
        already_in_group = [s for s in matched_sources if (s.split('/')[0] if '/' in s else '') == group_name]
        if len(already_in_group) == len(matched_sources):
            self.on_btn_clear_pick()
            QMessageBox.information(
                self,
                "계층 그룹 안내",
                f"선택한 객체는 이미 '{group_name}' 계층 그룹에 등록되어 있습니다."
            )
            return

        changed = False
        completely_moved_sources = []
        new_target_keys = []

        for source_name in matched_sources:
            source_mesh = self.part_meshes.get(source_name)
            if source_mesh is None or source_mesh.n_cells == 0:
                continue

            current_group = source_name.split('/')[0] if '/' in source_name else None
            if current_group == group_name:
                continue

            # [핵심] 선택한 객체의 모든 면(전체)을 이동 대상으로 복사 추출 (100% 모든 셀)
            extracted = source_mesh.copy()
            if not isinstance(extracted, pv.PolyData):
                extracted = extracted.extract_surface(algorithm='dataset_surface')

            # 소스 객체는 모든 면이 통째로 이동하므로 기존 그룹에서 완전 삭제 목록에 추가
            completely_moved_sources.append(source_name)

            # 타겟 key 및 sub_name 결정
            sub_name = source_name.split('/')[-1] if '/' in source_name else source_name
            target_key = f"{group_name}/{sub_name}"

            # 동일 그룹 내 동일 sub_name 중복 시 고유 식별자 부여하여 독립 Component 보존
            if target_key in self.part_meshes and target_key != source_name:
                counter = 1
                base_sub = sub_name
                while target_key in self.part_meshes and target_key != source_name:
                    sub_name = f"{base_sub}_{counter}"
                    target_key = f"{group_name}/{sub_name}"
                    counter += 1

            if extracted.n_cells > 0:
                try:
                    extracted = self.ensure_frontfaces_oriented(extracted)
                except Exception:
                    pass

            self.part_meshes[target_key] = extracted

            # 타겟 그룹 색상 적용
            target_color = self.part_colors.get(group_name, self.part_colors.get(source_name, [0.82, 0.82, 0.82]))
            self.part_colors[target_key] = target_color

            # 머티리얼 정보 인계
            if hasattr(self, 'part_materials') and source_name in self.part_materials:
                self.part_materials[target_key] = self.part_materials.pop(source_name)

            # 앞면 노멀 고정 상태 인계
            if hasattr(self, 'frontface_locked_parts') and source_name in self.frontface_locked_parts:
                self.frontface_locked_parts.discard(source_name)
                self.frontface_locked_parts.add(target_key)

            # 뷰포트에 타겟 메쉬 갱신 등록
            self.viewport.update_mesh(target_key, extracted, target_color)
            new_target_keys.append((target_key, sub_name))
            changed = True

        if changed:
            self.tree_widget.blockSignals(True)
            self.tree_widget.setUpdatesEnabled(False)

            # 1. 완벽히 이동된 소스 메시 및 아이템 정리
            for src_name in completely_moved_sources:
                self.part_meshes.pop(src_name, None)
                self.part_colors.pop(src_name, None)
                if hasattr(self, 'part_materials'):
                    self.part_materials.pop(src_name, None)
                if hasattr(self, 'frontface_locked_parts'):
                    self.frontface_locked_parts.discard(src_name)

                for plotter in getattr(self.viewport, 'plotters', []):
                    try:
                        plotter.remove_actor(src_name)
                    except Exception:
                        pass

                root = self.tree_widget.invisibleRootItem()
                if root:
                    for i in range(root.childCount()):
                        parent_node = root.child(i)
                        if parent_node is None:
                            continue
                        found_child = False
                        for j in range(parent_node.childCount()):
                            child_node = parent_node.child(j)
                            if child_node is not None and child_node.data(0, Qt.UserRole) == src_name:
                                parent_node.removeChild(child_node)
                                p_cnt = parent_node.childCount()
                                p_tag = parent_node.data(0, Qt.UserRole) or parent_node.text(0).split(' (')[0]
                                if p_cnt > 0:
                                    parent_node.setText(0, f"{p_tag} ({p_cnt} Groups)")
                                else:
                                    parent_node.setText(0, f"{p_tag} (0 Groups)")
                                found_child = True
                                break
                        if not found_child:
                            if parent_node.data(0, Qt.UserRole) == src_name and parent_node.childCount() == 0:
                                idx = self.tree_widget.indexOfTopLevelItem(parent_node)
                                if idx != -1:
                                    self.tree_widget.takeTopLevelItem(idx)

            # 2. 이동 대상 부모 그룹 하위에 신규/갱신 Component 등록
            group_color = self.part_colors.get(group_name, [0.82, 0.82, 0.82])
            rgb_key = (int(group_color[0]*255), int(group_color[1]*255), int(group_color[2]*255))
            pixmap = QPixmap(16, 16)
            pixmap.fill(QColor(*rgb_key))
            icon = QIcon(pixmap)

            # [핵심 버그 수정] 대상 부모 그룹의 현재 체크(가시성) 상태 계승
            # 대상 그룹이 체크 해제(Qt.Unchecked / Hide) 상태이면, 새로 이동된 객체도 정확히 체크 해제(Hide) 적용
            p_check_state = parent_group_item.checkState(0)
            target_child_state = Qt.Unchecked if p_check_state == Qt.Unchecked else Qt.Checked

            for t_key, sub_name in new_target_keys:
                item_exists = False
                for i in range(parent_group_item.childCount()):
                    child_node = parent_group_item.child(i)
                    if child_node.data(0, Qt.UserRole) == t_key:
                        item_exists = True
                        child_node.setIcon(0, icon)
                        child_node.setCheckState(0, target_child_state)
                        child_node.setData(0, Qt.UserRole + 2, target_child_state)
                        break

                if not item_exists:
                    new_item = QTreeWidgetItem(parent_group_item, [sub_name])
                    new_item.setFlags(new_item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEditable)
                    new_item.setCheckState(0, target_child_state)
                    new_item.setData(0, Qt.UserRole, t_key)
                    new_item.setData(0, Qt.UserRole + 2, target_child_state)
                    new_item.setIcon(0, icon)

            # 3. 이동 대상 부모 그룹 카운트 텍스트 갱신 및 3-state 체크 상태 동기화
            c_cnt = parent_group_item.childCount()
            if c_cnt > 0:
                parent_group_item.setText(0, f"{group_name} ({c_cnt} Groups)")
                tot = c_cnt
                chk = sum(1 for i in range(tot) if parent_group_item.child(i).checkState(0) == Qt.Checked)
                unchk = sum(1 for i in range(tot) if parent_group_item.child(i).checkState(0) == Qt.Unchecked)
                new_p_state = Qt.Checked if chk == tot else (Qt.Unchecked if unchk == tot else Qt.PartiallyChecked)
                parent_group_item.setCheckState(0, new_p_state)
                parent_group_item.setData(0, Qt.UserRole + 2, new_p_state)

            parent_group_item.setExpanded(True)

            self.tree_widget.setUpdatesEnabled(True)
            self.tree_widget.blockSignals(False)

            self.sync_outliner_visibility()
            self.on_btn_clear_pick()

            QMessageBox.information(
                self,
                "계층 그룹 이동 완료",
                f"선택한 객체의 모든 면(전체)을 '{group_name}' 계층 그룹으로 성공적으로 이동 및 등록했습니다."
            )
        else:
            self.on_btn_clear_pick()
            QMessageBox.information(self, "계층 이동 확인", "선택된 객체가 이미 대상 계층 그룹에 속해 있거나 유효한 대상을 찾지 못했습니다.")

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
            item.setData(0, Qt.UserRole, text)
            item.setData(0, Qt.UserRole + 2, Qt.Checked)
            
            # 새 계층에 랜덤 색상 부여
            color = [random.uniform(0.3, 0.9), random.uniform(0.3, 0.9), random.uniform(0.3, 0.9)]
            if not hasattr(self, 'part_colors'):
                self.part_colors = {}
            self.part_colors[text] = color
            
            pixmap = QPixmap(16, 16)
            pixmap.fill(QColor(int(color[0]*255), int(color[1]*255), int(color[2]*255)))
            item.setIcon(0, QIcon(pixmap))

    def delete_selected_hierarchy_node(self):
        """선택한 아웃라이너 계층(Component 또는 부품 그룹)을 객체 계층 구조 및 3D 뷰포트/데이터에서 완전 삭제"""
        selected_items = self.tree_widget.selectedItems()
        if not selected_items:
            QMessageBox.information(self, "선택 항목 없음", "삭제할 계층(Component)을 아웃라이너에서 먼저 선택해 주세요.")
            return

        confirm = QMessageBox.question(
            self,
            "계층 삭제",
            f"선택한 {len(selected_items)}개 계층/Component 항목을 삭제하시겠습니까?\n"
            f"3D 뷰포트 및 프로젝트 데이터에서 완전히 제거됩니다.",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No
        )
        if confirm != QMessageBox.Yes:
            return

        self.tree_widget.blockSignals(True)
        self.tree_widget.setUpdatesEnabled(False)

        keys_to_delete = set()
        items_to_remove = list(selected_items)

        for item in items_to_remove:
            # 1. 부모 그룹 노드인 경우 하위 자식들의 key 수집
            child_count = item.childCount()
            if child_count > 0:
                for i in range(child_count):
                    child = item.child(i)
                    key = child.data(0, Qt.UserRole)
                    if key:
                        keys_to_delete.add(key)
                tag_key = item.data(0, Qt.UserRole)
                if tag_key and hasattr(self, 'part_meshes') and tag_key in self.part_meshes:
                    keys_to_delete.add(tag_key)
            else:
                # 2. 단일 Component 자식 또는 독립 계층 노드
                key = item.data(0, Qt.UserRole)
                if key:
                    keys_to_delete.add(key)

        # 3. 트리 아이템 제거 및 부모 그룹 갱신
        for item in items_to_remove:
            parent_item = item.parent()
            if parent_item:
                parent_item.removeChild(item)
                c_cnt = parent_item.childCount()
                t_name = parent_item.data(0, Qt.UserRole)
                if c_cnt > 0:
                    parent_item.setText(0, f"{t_name} ({c_cnt} Groups)")
                else:
                    idx = self.tree_widget.indexOfTopLevelItem(parent_item)
                    if idx != -1:
                        self.tree_widget.takeTopLevelItem(idx)
            else:
                idx = self.tree_widget.indexOfTopLevelItem(item)
                if idx != -1:
                    self.tree_widget.takeTopLevelItem(idx)

        # 4. 데이터 딕셔너리 및 뷰포트 Plotter Actor 제거
        for key in keys_to_delete:
            if hasattr(self, 'part_meshes'):
                self.part_meshes.pop(key, None)
            if hasattr(self, 'part_colors'):
                self.part_colors.pop(key, None)
            if hasattr(self, 'part_materials'):
                self.part_materials.pop(key, None)
            if hasattr(self, 'frontface_locked_parts'):
                self.frontface_locked_parts.discard(key)

            if hasattr(self, 'viewport') and hasattr(self.viewport, 'plotters'):
                for plotter in self.viewport.plotters:
                    try:
                        plotter.remove_actor(key)
                    except Exception:
                        pass

        self.tree_widget.setUpdatesEnabled(True)
        self.tree_widget.blockSignals(False)

        # 5. 뷰포트 렌더링 및 치수 텍스트 갱신
        if hasattr(self, 'viewport'):
            self.viewport.render_active()
        self.update_dimensions_text()

        log_action("DELETE_HIERARCHY_NODES", f"{len(keys_to_delete)}개 Component 항목 삭제 완료")

    def on_tree_item_double_clicked(self, item, column):
        """아웃라이너 부품 항목 더블 클릭 시 인라인 명칭 편집(Rename) 모드 실행"""
        if item is not None:
            # 원본 명칭이 UserRole에 기록되어 있지 않은 경우 현재 텍스트로 보장 저장
            if not item.data(0, Qt.UserRole):
                item.setData(0, Qt.UserRole, item.text(0))
            self.tree_widget.editItem(item, column)

    def apply_primary_uv_mapping(self, mesh):
        """Component 및 ROOT_MODEL_ROOT 메쉬에 1차적인 텍스처 UV 좌표(Texture Coordinates)를 자동 생성 및 각인"""
        if mesh is None or not hasattr(mesh, 'n_points') or mesh.n_points == 0:
            return mesh

        try:
            import numpy as np
            # 1. PyVista 내장 texture_map_to_plane 시도
            mapped = mesh.texture_map_to_plane(inplace=False)
            if mapped is not None and mapped.active_texture_coordinates is not None:
                mesh.active_texture_coordinates = mapped.active_texture_coordinates
                return mesh
        except Exception as e:
            print(f"[1차 UV 매핑 Plane 예외 fallback]: {e}")

        try:
            import numpy as np
            pts = np.asarray(mesh.points)
            b = mesh.bounds  # [xmin, xmax, ymin, ymax, zmin, zmax]
            dx = max(b[1] - b[0], 1e-6)
            dy = max(b[3] - b[2], 1e-6)
            dz = max(b[5] - b[4], 1e-6)

            # 가장 얇은 축(두께 축)을 찾아 나머지 2개 축으로 직교 평면 UV 생성
            if dz <= dx and dz <= dy:
                u = (pts[:, 0] - b[0]) / dx
                v = (pts[:, 1] - b[2]) / dy
            elif dy <= dx and dy <= dz:
                u = (pts[:, 0] - b[0]) / dx
                v = (pts[:, 2] - b[4]) / dz
            else:
                u = (pts[:, 1] - b[2]) / dy
                v = (pts[:, 2] - b[4]) / dz

            u = np.clip(u, 0.0, 1.0)
            v = np.clip(v, 0.0, 1.0)
            uvs = np.column_stack([u, v]).astype(np.float32)
            mesh.active_texture_coordinates = uvs
        except Exception as e2:
            print(f"[1차 UV 매핑 BBox 예외]: {e2}")

        return mesh

    def open_material_editor_for_item(self, item):
        """객체 계층 구조에서 특정 Component 또는 ROOT_MODEL_ROOT 색상 창 클릭 시 머티리얼 슬롯을 열고 재질/색상/1차 UV 매핑 적용"""
        if item is None:
            return

        # 1. 적용 대상 부품 목록 수집
        target_items = []
        if item.childCount() > 0:
            def collect_children(p):
                for i in range(p.childCount()):
                    c = p.child(i)
                    if c.childCount() > 0:
                        collect_children(c)
                    else:
                        nm = c.data(0, Qt.UserRole) or c.text(0)
                        if nm:
                            target_items.append((c, nm))
            collect_children(item)
        else:
            nm = item.data(0, Qt.UserRole) or item.text(0)
            if nm:
                target_items.append((item, nm))

        if not target_items:
            return

        # 2. 첫 번째 부품 정보를 기준으로 초기 머티리얼 정보 구성
        first_item, first_name = target_items[0]
        if hasattr(self, 'part_materials') and first_name in self.part_materials:
            initial_mat = dict(self.part_materials[first_name])
        else:
            c = self.part_colors.get(first_name, [0.8, 0.8, 0.8])
            hex_val = f"#{int(c[0]*255):02X}{int(c[1]*255):02X}{int(c[2]*255):02X}"
            raw_title = item.text(0).split(" (")[0]
            initial_mat = {
                "name": raw_title,
                "color": list(c),
                "hex": hex_val,
                "roughness": 0.5,
                "metallic": 0.0,
                "normal_scale": 1.0,
                "bright": 1.0,
                "reflection": 1.0,
                "refraction": 1.5,
                "emissive": 0.0,
                "coat_strength": 0.0,
                "coat_roughness": 0.0,
                "anisotropy": 0.0,
                "occlusion": 1.0,
                "use_hdri": True,
                "hdri_ratio": 1.0,
                "use_bright": True,
                "use_roughness": True,
                "use_metallic": True,
                "use_normal": True,
                "use_reflection": True,
                "use_refraction": True,
                "use_emissive": False,
                "use_coat_strength": False,
                "use_coat_roughness": False,
                "use_anisotropy": False,
                "use_occlusion": False,
            }

        # 3. PBR 머티리얼 인스펙터 다이얼로그 실행
        dlg = MaterialInspectorDialog(initial_mat, apply_callback=None, parent=self)
        res = dlg.exec()
        applied_info = getattr(dlg, 'applied_mat_info', None)
        dlg.cleanup()
        dlg.deleteLater()

        if res != QDialog.Accepted or not applied_info:
            return

        color = applied_info.get('color', [0.8, 0.8, 0.8])
        roughness = applied_info.get('roughness', 0.5)
        metallic = applied_info.get('metallic', 0.0)
        normal_scale = applied_info.get('normal_scale', 1.0)
        bright = applied_info.get('bright', 1.0)
        reflection = applied_info.get('reflection', 1.0)
        refraction = applied_info.get('refraction', 1.5)
        emissive = applied_info.get('emissive', 0.0)
        coat_strength = applied_info.get('coat_strength', 0.0)
        coat_roughness = applied_info.get('coat_roughness', 0.0)
        anisotropy = applied_info.get('anisotropy', 0.0)
        occlusion = applied_info.get('occlusion', 1.0)

        use_hdri = applied_info.get('use_hdri', True)
        hdri_ratio = applied_info.get('hdri_ratio', 1.0)
        use_bright = applied_info.get('use_bright', True)
        use_roughness = applied_info.get('use_roughness', True)
        use_metallic = applied_info.get('use_metallic', True)
        use_normal = applied_info.get('use_normal', True)
        use_reflection = applied_info.get('use_reflection', True)
        use_refraction = applied_info.get('use_refraction', True)
        use_emissive = applied_info.get('use_emissive', False)
        use_coat_strength = applied_info.get('use_coat_strength', False)
        use_coat_roughness = applied_info.get('use_coat_roughness', False)
        use_anisotropy = applied_info.get('use_anisotropy', False)
        use_occlusion = applied_info.get('use_occlusion', False)

        # 뷰포트 디스플레이 모드가 solid나 wireframe이면 머티리얼 프리뷰 모드로 자동 전환
        if getattr(self.viewport, 'current_render_mode', None) in ["solid", "wireframe"]:
            self.viewport.set_display_render_mode("material_preview")

        # HDRI 환경맵 갱신
        hdri_path = applied_info.get('hdri_path')
        if hdri_path and os.path.exists(hdri_path) and use_hdri:
            self.viewport.set_hdri_environment(hdri_path, force=True)

        if not hasattr(self, 'part_materials'):
            self.part_materials = {}

        # 4. 아웃라이너 트리 아이콘용 QPixmap 생성
        rgb_key = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
        pixmap = QPixmap(16, 16)
        pixmap.fill(QColor(*rgb_key))
        new_icon = QIcon(pixmap)

        # 클릭한 노드 아이콘 갱신
        item.setIcon(0, new_icon)

        # 5. 각 부품 메쉬에 1차 UV Mapping 적용, 법선 보정 및 PBR 속성 반영
        count = 0
        for c_item, name in target_items:
            c_item.setIcon(0, new_icon)
            
            # part_meshes 키 매칭 보정
            target_key = name
            if target_key not in getattr(self, 'part_meshes', {}):
                for pm_k in self.part_meshes.keys():
                    if pm_k.endswith("/" + name) or pm_k == name:
                        target_key = pm_k
                        break

            if target_key in getattr(self, 'part_meshes', {}) and self.part_meshes[target_key].n_cells > 0:
                self.part_colors[target_key] = color
                self.part_materials[target_key] = applied_info

                # (1) 1차적인 UV Mapping 적용
                mesh = self.part_meshes[target_key]
                mesh = self.apply_primary_uv_mapping(mesh)

                # (2) 법선 및 앞면 정렬 보정
                fixed_mesh = self.ensure_frontfaces_oriented(mesh, name=target_key, force=False)
                self.part_meshes[target_key] = fixed_mesh

                # (3) 뷰포트 액터 갱신 (UV 좌표가 반영된 메쉬로 갱신)
                self.viewport.update_mesh(target_key, fixed_mesh, color)

                # (4) PBR 속성 반영
                self.viewport.update_mesh_pbr(
                    target_key, color=color, roughness=roughness, metallic=metallic, normal_scale=normal_scale,
                    bright=bright, reflection=reflection, refraction=refraction,
                    emissive=emissive, coat_strength=coat_strength, coat_roughness=coat_roughness,
                    anisotropy=anisotropy, occlusion=occlusion,
                    use_hdri=use_hdri, hdri_ratio=hdri_ratio,
                    use_roughness=use_roughness, use_metallic=use_metallic, use_normal=use_normal,
                    use_bright=use_bright, use_reflection=use_reflection, use_refraction=use_refraction,
                    use_emissive=use_emissive, use_coat_strength=use_coat_strength,
                    use_coat_roughness=use_coat_roughness, use_anisotropy=use_anisotropy, use_occlusion=use_occlusion,
                    render=False
                )
                count += 1

        if count > 0:
            self.viewport.render_active()

        # 6. 하단 기본 머티리얼 슬롯 패널에도 함께 동기화/저장
        if hasattr(self, 'mat_slot_panel') and self.mat_slot_panel is not None:
            slot_idx = getattr(self.mat_slot_panel, 'selected_slot', 0)
            if 0 <= slot_idx < len(self.mat_slot_panel.materials):
                self.mat_slot_panel.materials[slot_idx].update(applied_info)
                self.mat_slot_panel.update_slot_ui(slot_idx)

        # 7. 사용자 안내 알림
        display_name = item.text(0).split(" (")[0]
        mat_name = applied_info.get("name", "Preset")
        QMessageBox.information(
            self,
            "머티리얼 & 1차 UV Mapping 완료",
            f"선택한 '{display_name}' ({count}개 부품)에\n머티리얼 '{mat_name}' 및 1차적인 UV Mapping이 완벽히 적용되었습니다!"
        )

    def on_tree_item_clicked(self, item, column):
        """아웃라이너 항목 클릭 시 색상 창(아이콘 영역) 클릭 여부를 감지하여 머티리얼 슬롯 호출"""
        if item is None or column != 0:
            return

        pos = self.tree_widget.viewport().mapFromGlobal(QCursor.pos())
        rect = self.tree_widget.visualItemRect(item)
        rel_x = pos.x() - rect.left()

        # 체크박스(0~15px) 우측의 색상 창(아이콘) 영역: 약 14px ~ 48px
        if 14 <= rel_x <= 48:
            self.open_material_editor_for_item(item)

    def on_tree_context_menu(self, pos):
        """아웃라이너 항목 우클릭 시 계층 이동, 머티리얼 및 삭제 메뉴 팝업"""
        item = self.tree_widget.itemAt(pos)
        if item is None:
            return

        # 우클릭한 항목을 즉시 아웃라이너 선택 항목으로 지정
        self.tree_widget.setCurrentItem(item)
        item.setSelected(True)

        from PySide6.QtWidgets import QMenu
        from PySide6.QtGui import QAction

        menu = QMenu(self)
        menu.setStyleSheet("""
            QMenu {
                background-color: #23272e;
                color: #ffffff;
                border: 1px solid #4b5563;
                border-radius: 4px;
                padding: 4px;
                font-weight: bold;
            }
            QMenu::item {
                padding: 6px 20px;
                border-radius: 3px;
            }
            QMenu::item:selected {
                background-color: #3b82f6;
                color: #ffffff;
            }
        """)
        # 1. 계층 그룹 이동 메뉴
        act_move = QAction("📦 계층 그룹 이동", self)
        act_move.setToolTip("뷰포트에서 선택된 객체/면을 현재 우클릭한 계층 그룹으로 이동 및 등록합니다.")
        act_move.triggered.connect(lambda: self.on_btn_assign(target_item=item))
        menu.addAction(act_move)

        menu.addSeparator()

        # 2. 머티리얼 및 색상 설정 메뉴
        act_mat = QAction("🎨 머티리얼 및 색상 설정 (PBR / 1차 UV 매핑)", self)
        act_mat.triggered.connect(lambda: self.open_material_editor_for_item(item))
        menu.addAction(act_mat)

        # 3. 계층 삭제 메뉴
        act_del = QAction("🗑️ 선택 계층 삭제", self)
        act_del.triggered.connect(self.delete_selected_hierarchy_node)
        menu.addAction(act_del)

        menu.exec(self.tree_widget.viewport().mapToGlobal(pos))

    def _create_progress_dialog(self, title: str, label_text: str, max_val: int = 100):
        """응답 없음 방지 및 사용자 안내용 모달 프로그레스 다이얼로그 생성"""
        progress = QProgressDialog(label_text, None, 0, max_val, self)
        progress.setWindowTitle(title)
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setMinimumDuration(0)
        progress.setAutoClose(True)
        progress.setAutoReset(True)
        progress.setCancelButton(None)
        progress.setStyleSheet("""
            QProgressDialog {
                background-color: #23272e;
                color: #abb2bf;
                border: 2px solid #3b82f6;
                border-radius: 8px;
            }
            QLabel {
                color: #ffffff;
                font-size: 13px;
                font-weight: bold;
                padding: 6px;
            }
            QProgressBar {
                border: 1px solid #4b5563;
                border-radius: 4px;
                text-align: center;
                color: #ffffff;
                font-weight: bold;
                background-color: #1e2227;
                height: 22px;
            }
            QProgressBar::chunk {
                background-color: #3b82f6;
                border-radius: 3px;
            }
        """)
        progress.setFixedSize(400, 115)
        progress.show()
        QApplication.processEvents()
        return progress

    def on_tree_item_changed(self, item, column):
        if column != 0 or item is None:
            return

        # 재귀 호출 방지 플래그
        if getattr(self, '_is_updating_tree', False):
            return

        user_key = item.data(0, Qt.UserRole)
        display_text = item.text(0)

        # 1. UserRole 데이터가 비어있었던 경우 (초기 생성 직후 등) fallback 보장
        if not user_key:
            user_key = display_text
            self._is_updating_tree = True
            item.setData(0, Qt.UserRole, user_key)
            self._is_updating_tree = False

        # 2. 아웃라이너 부품 명칭 사용자가 직접 수정(Rename) 시 키 동기화
        last_display = item.data(0, Qt.UserRole + 1)
        if last_display is not None and last_display != display_text:
            old_key = user_key
            new_key = display_text
            if hasattr(self, 'part_meshes') and old_key in self.part_meshes:
                self.part_meshes[new_key] = self.part_meshes.pop(old_key)
            if hasattr(self, 'part_colors') and old_key in self.part_colors:
                self.part_colors[new_key] = self.part_colors.pop(old_key)
            if hasattr(self, 'part_materials') and old_key in self.part_materials:
                self.part_materials[new_key] = self.part_materials.pop(old_key)

            for plotter in self.viewport.plotters:
                if old_key in plotter.actors:
                    actor = plotter.actors.pop(old_key)
                    plotter.actors[new_key] = actor

            self._is_updating_tree = True
            item.setData(0, Qt.UserRole, new_key)
            item.setData(0, Qt.UserRole + 1, display_text)
            self._is_updating_tree = False
            return

        # 3. 체크 상태 변경 확인 (CheckState 변경 시에만 동기화 실행)
        current_state = item.checkState(0)
        last_state = item.data(0, Qt.UserRole + 2)

        if last_state == current_state:
            return

        # 완벽한 무한 루프 및 시그널 폭풍 원천 차단
        self._is_updating_tree = True
        self.tree_widget.blockSignals(True)
        self.tree_widget.setUpdatesEnabled(False)

        try:
            item.setData(0, Qt.UserRole + 2, current_state)

            # [경우 A]: 사용자가 부모 그룹 노드를 직접 클릭한 경우 (하위 Component 일괄 전파)
            if item.childCount() > 0:
                target_state = current_state
                # 부모가 부분 선택([■]) 상태에서 클릭된 경우 -> 직관적으로 완전 해제([ ])로 전환하여 모든 Component 일괄 해제
                if target_state == Qt.PartiallyChecked or last_state == Qt.PartiallyChecked:
                    if current_state == Qt.Checked:
                        target_state = Qt.Unchecked
                    else:
                        target_state = current_state
                    item.setCheckState(0, target_state)
                    item.setData(0, Qt.UserRole + 2, target_state)

                def _propagate_to_children(parent_node, t_state):
                    for i in range(parent_node.childCount()):
                        c = parent_node.child(i)
                        c.setCheckState(0, t_state)
                        c.setData(0, Qt.UserRole + 2, t_state)
                        _propagate_to_children(c, t_state)

                _propagate_to_children(item, target_state)

                # 상위 부모가 더 있다면 상위로만 갱신 (자기 자신은 건너뜀)
                p = item.parent()
                while p is not None:
                    tot = p.childCount()
                    chk = sum(1 for i in range(tot) if p.child(i).checkState(0) == Qt.Checked)
                    unchk = sum(1 for i in range(tot) if p.child(i).checkState(0) == Qt.Unchecked)
                    new_p_state = Qt.Checked if chk == tot else (Qt.Unchecked if unchk == tot else Qt.PartiallyChecked)
                    p.setCheckState(0, new_p_state)
                    p.setData(0, Qt.UserRole + 2, new_p_state)
                    p = p.parent()

            # [경우 B]: 사용자가 말단 자식 Component 또는 추가된 단일 계층 노드를 직접 클릭한 경우 (부모 노드 3-state 갱신)
            else:
                p = item.parent()
                while p is not None:
                    tot = p.childCount()
                    chk = sum(1 for i in range(tot) if p.child(i).checkState(0) == Qt.Checked)
                    unchk = sum(1 for i in range(tot) if p.child(i).checkState(0) == Qt.Unchecked)
                    new_p_state = Qt.Checked if chk == tot else (Qt.Unchecked if unchk == tot else Qt.PartiallyChecked)
                    p.setCheckState(0, new_p_state)
                    p.setData(0, Qt.UserRole + 2, new_p_state)
                    p = p.parent()

        finally:
            self.tree_widget.setUpdatesEnabled(True)
            self.tree_widget.blockSignals(False)
            self._is_updating_tree = False
            # 화면 체크박스 그래픽 즉시 갱신 (지연 없는 렌더링)
            self.tree_widget.viewport().update()

        # 4. 뷰포트(Cinematic View 및 Quad View 전체) 가시성 즉각 동기화 (초고속 0.001초 바로 적용)
        self.sync_outliner_visibility()

    def sync_outliner_visibility(self):
        """트리 위젯의 체크 상태에 맞춰 Quad 뷰포트 및 Cinematic View의 모든 액터 가시성을 0.001초 만에 즉시 동기화"""
        checked_names = self.get_checked_item_names()
        
        ignore_names = {'bg_grid', 'axis_x', 'axis_y', 'view_name', 'Cinematic', 'dimensions', 
                        'picked_cells', 'my_picked_cells', '_picked_through_selection', 
                        '_picked_visible_selection', '_rectangle_selection_frustum'}

        # 모든 플로터에 존재하는 부품 액터들의 가시성 일괄 적용 (체크 해제 시 100% 감춤 보장, 초고속 0.001초)
        for plotter in self.viewport.plotters:
            for actor_name, actor in plotter.actors.items():
                if actor_name in ignore_names or actor_name.startswith('axis_') or actor_name.startswith('bg_'):
                    continue
                is_vis = (actor_name in checked_names)
                try:
                    actor.SetVisibility(is_vis)
                    actor.SetPickable(is_vis)
                except Exception:
                    pass

        self.viewport.render_active()
        if hasattr(self.viewport, 'stacked_widget') and self.viewport.stacked_widget.currentIndex() == 1:
            self.viewport.sync_quad_view()

        # 체크된 항목이 아예 없으면 선택 하이라이트도 뷰포트에서 안전하게 제거
        if not checked_names:
            try:
                self.viewport.plotter_single.remove_actor("my_picked_cells")
                self.viewport.plotter_single.render()
            except Exception:
                pass
        elif getattr(self, 'picked_mesh', None) is not None:
            self.update_picked_highlight()

    def get_all_tree_item_names(self):
        """트리 위젯의 모든 아이템 텍스트(오브젝트 이름) 세트 반환"""
        names = set()
        def _traverse(item):
            for i in range(item.childCount()):
                child = item.child(i)
                u_data = child.data(0, Qt.UserRole)
                if u_data:
                    names.add(str(u_data))
                names.add(child.text(0))
                _traverse(child)
        root = self.tree_widget.invisibleRootItem()
        _traverse(root)
        return names

    def set_all_tree_items_check_state(self, checked: bool):
        """트리 위젯의 모든 항목에 대해 체크/체크해제 상태를 일괄 적용하고, 부품 객체의 뷰포트 가시성을 즉시 동기화합니다."""
        target_state = Qt.Checked if checked else Qt.Unchecked

        self._is_updating_tree = True
        self.tree_widget.blockSignals(True)
        self.tree_widget.setUpdatesEnabled(False)
        try:
            def _set_check_recursive(item):
                item.setCheckState(0, target_state)
                item.setData(0, Qt.UserRole + 2, target_state)
                for i in range(item.childCount()):
                    _set_check_recursive(item.child(i))
                    
            root = self.tree_widget.invisibleRootItem()
            for i in range(root.childCount()):
                _set_check_recursive(root.child(i))
        finally:
            self.tree_widget.setUpdatesEnabled(True)
            self.tree_widget.blockSignals(False)
            self._is_updating_tree = False
            self.tree_widget.viewport().update()

        # 전체 해제(checked=False)인 경우: 0.001초 즉각 완료
        if not checked:
            self.picked_mesh = None
            try:
                self.viewport.plotter_single.remove_actor("my_picked_cells")
            except Exception:
                pass
            ignore_names = {'bg_grid', 'axis_x', 'axis_y', 'view_name', 'Cinematic', 'dimensions', 
                            'picked_cells', 'my_picked_cells', '_picked_through_selection', 
                            '_picked_visible_selection', '_rectangle_selection_frustum'}
            for plotter in self.viewport.plotters:
                for actor_name, actor in plotter.actors.items():
                    if actor_name in ignore_names or actor_name.startswith('axis_') or actor_name.startswith('bg_'):
                        continue
                    try:
                        actor.SetVisibility(False)
                        actor.SetPickable(False)
                    except Exception:
                        pass
            self.viewport.render_active()
            return

        # 전체 선택(checked=True)인 경우: 모든 객체 가시화 0.001초 즉각 완료
        ignore_names = {'bg_grid', 'axis_x', 'axis_y', 'view_name', 'Cinematic', 'dimensions', 
                        'picked_cells', 'my_picked_cells', '_picked_through_selection', 
                        '_picked_visible_selection', '_rectangle_selection_frustum'}
        for plotter in self.viewport.plotters:
            for actor_name, actor in plotter.actors.items():
                if actor_name in ignore_names or actor_name.startswith('axis_') or actor_name.startswith('bg_'):
                    continue
                try:
                    actor.SetVisibility(True)
                    actor.SetPickable(True)
                except Exception:
                    pass
        self.viewport.render_active()
        if getattr(self, 'picked_mesh', None) is not None:
            self.update_picked_highlight()

    def get_checked_item_names(self):
        """트리 위젯에서 체크박스가 Checked(체크됨) 상태인 모든 아이템의 텍스트 및 전체 레이어 경로 이름 세트 반환"""
        checked = set()
        def _traverse(item):
            for i in range(item.childCount()):
                child = item.child(i)
                if child.checkState(0) == Qt.Checked:
                    u_data = child.data(0, Qt.UserRole)
                    if u_data:
                        checked.add(str(u_data))
                    checked.add(child.text(0))
                _traverse(child)
        
        root = self.tree_widget.invisibleRootItem()
        _traverse(root)
        return checked

    def filter_mesh_by_checked_objects(self, mesh, checked_names=None):
        """주어진 mesh에서 계층구조상 체크된(Qt.Checked) 오브젝트들에 속하는 cell만 추출하여 반환 (초연산 AABB & 역필터링 최적화)"""
        if mesh is None or not hasattr(mesh, 'n_cells') or mesh.n_cells == 0:
            return None
            
        if checked_names is None:
            checked_names = self.get_checked_item_names()
        if not checked_names:
            return None
            
        part_meshes = getattr(self, 'part_meshes', {})
        if not part_meshes:
            return mesh
            
        total_parts = len(part_meshes)
        if len(checked_names) >= total_parts:
            return mesh

        import numpy as np
        import pyvista as pv

        mb = mesh.bounds  # [xmin, xmax, ymin, ymax, zmin, zmax]
        mesh_centers = mesh.cell_centers().points

        unchecked_names = set(part_meshes.keys()) - checked_names
        if not unchecked_names:
            return mesh

        # 체크 해제된 부품 수가 적은 경우(예: 지붕부 107개만 끄고 750개 켜짐):
        # 해제된 부품의 셀만 찾아서 mesh에서 제거하는 '역필터링(Inverse)' 수행 (속도 수백 배 향상)
        if len(unchecked_names) < len(checked_names):
            candidate_unchecked = []
            for name in unchecked_names:
                p_mesh = part_meshes.get(name)
                if p_mesh is None or p_mesh.n_cells == 0:
                    continue
                pb = p_mesh.bounds
                # AABB 교차 검사 (BBox가 겹치지 않으면 즉시 스킵)
                if (pb[1] < mb[0] - 0.001 or pb[0] > mb[1] + 0.001 or
                    pb[3] < mb[2] - 0.001 or pb[2] > mb[3] + 0.001 or
                    pb[5] < mb[4] - 0.001 or pb[4] > mb[5] + 0.001):
                    continue
                candidate_unchecked.append(p_mesh)

            if not candidate_unchecked:
                return mesh

            remove_indices = set()
            try:
                from scipy.spatial import cKDTree
                tree = cKDTree(mesh_centers)
                for p_mesh in candidate_unchecked:
                    p_centers = p_mesh.cell_centers().points
                    indices_list = tree.query_ball_point(p_centers, r=0.001)
                    for idxs in indices_list:
                        if idxs:
                            remove_indices.update(idxs)
            except ImportError:
                for p_mesh in candidate_unchecked:
                    pb = p_mesh.bounds
                    in_bbox = (
                        (mesh_centers[:, 0] >= pb[0] - 0.001) & (mesh_centers[:, 0] <= pb[1] + 0.001) &
                        (mesh_centers[:, 1] >= pb[2] - 0.001) & (mesh_centers[:, 1] <= pb[3] + 0.001) &
                        (mesh_centers[:, 2] >= pb[4] - 0.001) & (mesh_centers[:, 2] <= pb[5] + 0.001)
                    )
                    cand = np.where(in_bbox)[0]
                    if len(cand) > 0:
                        remove_indices.update(cand.tolist())

            if not remove_indices:
                return mesh
            if len(remove_indices) >= mesh.n_cells:
                return None

            keep_indices = [i for i in range(mesh.n_cells) if i not in remove_indices]
            if not keep_indices:
                return None

            filtered = mesh.extract_cells(keep_indices)
            if not isinstance(filtered, pv.PolyData):
                filtered = filtered.extract_surface(algorithm='dataset_surface')
            return filtered

        else:
            # 체크된 부품 수가 적은 경우: 정방향 필터링 (AABB 컬링 적용)
            candidate_checked = []
            for name in checked_names:
                p_mesh = part_meshes.get(name)
                if p_mesh is None or p_mesh.n_cells == 0:
                    continue
                pb = p_mesh.bounds
                if (pb[1] < mb[0] - 0.001 or pb[0] > mb[1] + 0.001 or
                    pb[3] < mb[2] - 0.001 or pb[2] > mb[3] + 0.001 or
                    pb[5] < mb[4] - 0.001 or pb[4] > mb[5] + 0.001):
                    continue
                candidate_checked.append(p_mesh)

            if not candidate_checked:
                return None

            keep_indices = set()
            try:
                from scipy.spatial import cKDTree
                tree = cKDTree(mesh_centers)
                for p_mesh in candidate_checked:
                    p_centers = p_mesh.cell_centers().points
                    indices_list = tree.query_ball_point(p_centers, r=0.001)
                    for idxs in indices_list:
                        if idxs:
                            keep_indices.update(idxs)
            except ImportError:
                for p_mesh in candidate_checked:
                    pb = p_mesh.bounds
                    in_bbox = (
                        (mesh_centers[:, 0] >= pb[0] - 0.001) & (mesh_centers[:, 0] <= pb[1] + 0.001) &
                        (mesh_centers[:, 1] >= pb[2] - 0.001) & (mesh_centers[:, 1] <= pb[3] + 0.001) &
                        (mesh_centers[:, 2] >= pb[4] - 0.001) & (mesh_centers[:, 2] <= pb[5] + 0.001)
                    )
                    cand = np.where(in_bbox)[0]
                    if len(cand) > 0:
                        keep_indices.update(cand.tolist())

            if not keep_indices:
                return None
            if len(keep_indices) == mesh.n_cells:
                return mesh

            filtered = mesh.extract_cells(list(keep_indices))
            if not isinstance(filtered, pv.PolyData):
                filtered = filtered.extract_surface(algorithm='dataset_surface')
            return filtered

    def update_picked_highlight(self):
        """현재 self.picked_mesh 상태를 바탕으로 마젠타 선택 액터를 갱신"""
        try:
            self.viewport.plotter_single.remove_actor("my_picked_cells")
        except Exception:
            pass
            
        if getattr(self, 'picked_mesh', None) is not None and getattr(self.picked_mesh, 'n_cells', 0) > 0:
            self.viewport.plotter_single.add_mesh(
                self.picked_mesh, 
                name="my_picked_cells", 
                color="magenta", 
                show_edges=True, 
                line_width=3, 
                pickable=False,
                reset_camera=False
            )
        self.viewport.plotter_single.render()

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
            
            # HDRI 환경맵 불러오기 툴바 레이아웃
            hdri_bar_layout = QHBoxLayout()
            btn_load_hdri = QPushButton("📂 HDRI 환경맵 불러오기")
            btn_load_hdri.setCursor(Qt.PointingHandCursor)
            btn_load_hdri.setStyleSheet("font-weight: bold; font-size: 13px; background-color: #2563EB; color: white; padding: 6px 14px; border-radius: 4px;")
            
            lbl_hdri_status = QLabel("현재 배경: 기본 스카이박스 (Skybox)")
            lbl_hdri_status.setStyleSheet("font-weight: bold; font-size: 12px; color: #0044CC; background-color: #E8EEF9; padding: 5px 10px; border-radius: 4px; border: 1px solid #B0C4DE;")
            
            btn_reset_hdri = QPushButton("🔄 기본 배경 복원")
            btn_reset_hdri.setCursor(Qt.PointingHandCursor)
            btn_reset_hdri.setStyleSheet("font-weight: bold; font-size: 13px; background-color: #4B5563; color: white; padding: 6px 14px; border-radius: 4px;")
            
            hdri_bar_layout.addWidget(btn_load_hdri)
            hdri_bar_layout.addWidget(lbl_hdri_status, stretch=1)
            hdri_bar_layout.addWidget(btn_reset_hdri)
            layout.addLayout(hdri_bar_layout)
            
            # 대화형 PyVista 인터랙터 생성
            plotter = QtInteractor(dialog)
            layout.addWidget(plotter.interactor)
            
            # 헬퍼 함수: 스카이박스 및 PBR IBL 환경맵 적용 (하늘이 위로 가도록 X축 90도 회전 보정)
            def apply_skybox_texture(tex_or_cubemap):
                try:
                    plotter.remove_actor("skybox")
                except Exception:
                    pass
                    
                try:
                    plotter.set_environment_texture(tex_or_cubemap)
                except Exception:
                    pass
                    
                try:
                    skybox_actor = tex_or_cubemap.to_skybox()
                    # VTK 스카이박스의 축을 Y-up에서 Z-up(하늘 위쪽)으로 올바르게 X축 90도 회전 보정
                    skybox_actor.RotateX(90)
                    plotter.add_actor(skybox_actor, name="skybox")
                except Exception as e:
                    print(f"Skybox 생성 오류: {e}")

            # 기본 HDRI 맵 및 환경 설정
            try:
                default_cubemap = pv.examples.download_sky_box_cube_map()
                apply_skybox_texture(default_cubemap)
            except Exception as e:
                print(f"기본 스카이박스 로드 예외: {e}")
                
            from PySide6.QtWidgets import QFileDialog
            from viewport_widget import read_hdri_texture

            def choose_hdri_in_dialog():
                file_path, _ = QFileDialog.getOpenFileName(
                    dialog,
                    "HDRI 환경맵 파일 선택",
                    "",
                    "HDRI Image Files (*.hdr *.exr *.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff);;All Files (*.*)"
                )
                if file_path and os.path.exists(file_path):
                    try:
                        tex = read_hdri_texture(file_path)
                        apply_skybox_texture(tex)
                        file_name = os.path.basename(file_path)
                        lbl_hdri_status.setText(f"현재 배경 HDRI: {file_name}")
                        plotter.render()
                    except Exception as err:
                        QMessageBox.warning(dialog, "HDRI 로드 오류", f"HDRI 환경맵 로드 중 오류가 발생했습니다:\n{err}")

            def reset_hdri_in_dialog():
                try:
                    default_cubemap = pv.examples.download_sky_box_cube_map()
                    apply_skybox_texture(default_cubemap)
                    lbl_hdri_status.setText("현재 배경: 기본 스카이박스 (Skybox)")
                    plotter.render()
                except Exception as err:
                    print(f"기본 배경 복원 오류: {err}")

            btn_load_hdri.clicked.connect(choose_hdri_in_dialog)
            btn_reset_hdri.clicked.connect(reset_hdri_in_dialog)
            
            try:
                plotter.enable_shadows()
            except Exception:
                pass
            plotter.enable_anti_aliasing("msaa")
            
            # 현재 뷰어의 로드된 액터(메쉬)들 복제
            orig_plotter = self.viewport.plotter_single
            for name, actor in orig_plotter.actors.items():
                if name in ["Axes", "Grid", "Border", "skybox"] or "axes" in name.lower() or "grid" in name.lower() or "skybox" in name.lower():
                    continue
                    
                if hasattr(actor, 'mapper') and hasattr(actor.mapper, 'dataset'):
                    mesh = actor.mapper.dataset
                    if not isinstance(mesh, (pv.PolyData, pv.UnstructuredGrid)):
                        continue
                        
                    color = actor.prop.color if hasattr(actor.prop, 'color') else 'white'
                    new_actor = plotter.add_mesh(mesh, name=name, color=color, show_edges=False)
                    
                    # PBR 재질 및 양면 렌더링(Two-Sided) 부여
                    try:
                        new_actor.prop.interpolation = 'pbr'
                        if hasattr(actor.prop, 'metallic'): new_actor.prop.metallic = actor.prop.metallic
                        if hasattr(actor.prop, 'roughness'): new_actor.prop.roughness = actor.prop.roughness
                        new_actor.prop.SetBackfaceCulling(False)
                        new_actor.prop.SetFrontfaceCulling(False)
                        new_actor.SetBackfaceProperty(new_actor.prop)
                    except Exception:
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

    def on_apply_material_slot(self, mat_info):
        """선택한 PBR 머티리얼 파라미터(알베도 컬러, Roughness, Metallic 등)를 계층구조에서 체크된 오브젝트들에 즉시 반영"""
        checked_names = self.get_checked_item_names()
        if not checked_names:
            QMessageBox.information(self, "대상 없음", "재질을 적용할 오브젝트를 계층구조(Outliner)에서 먼저 체크해 주세요.")
            return
            
        color = mat_info.get('color', [0.8, 0.8, 0.8])
        name_str = mat_info.get('name', 'Preset')
        bright = mat_info.get('bright', 1.0)
        roughness = mat_info.get('roughness', 0.5)
        metallic = mat_info.get('metallic', 0.0)
        normal_scale = mat_info.get('normal_scale', 1.0)
        reflection = mat_info.get('reflection', 1.0)
        refraction = mat_info.get('refraction', 1.5)
        emissive = mat_info.get('emissive', 0.0)
        coat_strength = mat_info.get('coat_strength', 0.0)
        coat_roughness = mat_info.get('coat_roughness', 0.0)
        anisotropy = mat_info.get('anisotropy', 0.0)
        occlusion = mat_info.get('occlusion', 1.0)
        
        hdri_path = mat_info.get('hdri_path')
        use_hdri = mat_info.get('use_hdri', True)
        hdri_ratio = mat_info.get('hdri_ratio', 1.0)
        use_bright = mat_info.get('use_bright', True)
        use_roughness = mat_info.get('use_roughness', True)
        use_metallic = mat_info.get('use_metallic', True)
        use_normal = mat_info.get('use_normal', True)
        use_reflection = mat_info.get('use_reflection', True)
        use_refraction = mat_info.get('use_refraction', True)
        use_emissive = mat_info.get('use_emissive', False)
        use_coat_strength = mat_info.get('use_coat_strength', False)
        use_coat_roughness = mat_info.get('use_coat_roughness', False)
        use_anisotropy = mat_info.get('use_anisotropy', False)
        use_occlusion = mat_info.get('use_occlusion', False)

        # 머티리얼 적용 시 현재 뷰포트 디스플레이 모드가 solid나 wireframe이면 머티리얼 프리뷰(material_preview) 모드로 자동 전환
        if getattr(self.viewport, 'current_render_mode', None) in ["solid", "wireframe"]:
            self.viewport.set_display_render_mode("material_preview")

        if hdri_path and os.path.exists(hdri_path) and use_hdri:
            self.viewport.set_hdri_environment(hdri_path, force=True)
        else:
            self.viewport.remove_hdri_environment()
        
        self.last_applied_material = mat_info
        if not hasattr(self, 'part_materials'):
            self.part_materials = {}

        count = 0
        for name in checked_names:
            if name in getattr(self, 'part_meshes', {}) and self.part_meshes[name].n_cells > 0:
                self.part_colors[name] = color
                self.part_materials[name] = mat_info
                
                # 머티리얼 적용 대상 부품의 외부 앞면(Frontface) 방향 및 법선(Normals) 재배향 보장 (락이 걸린 메쉬는 보존)
                fixed_mesh = self.ensure_frontfaces_oriented(self.part_meshes[name], name=name, force=False)
                self.part_meshes[name] = fixed_mesh

                # 뷰포트 액터에 정렬 및 락(Lock)이 적용된 최신 PolyData 메쉬 버퍼 100% 갱신 등록
                self.viewport.update_mesh(name, fixed_mesh, color)

                # 개별 오브젝트마다 렌더링하지 않고 render=False로 속성만 일괄 변경
                self.viewport.update_mesh_pbr(
                    name, color=color, roughness=roughness, metallic=metallic, normal_scale=normal_scale,
                    bright=bright, reflection=reflection, refraction=refraction,
                    emissive=emissive, coat_strength=coat_strength, coat_roughness=coat_roughness,
                    anisotropy=anisotropy, occlusion=occlusion,
                    use_hdri=use_hdri, hdri_ratio=hdri_ratio,
                    use_roughness=use_roughness, use_metallic=use_metallic, use_normal=use_normal,
                    use_bright=use_bright, use_reflection=use_reflection, use_refraction=use_refraction,
                    use_emissive=use_emissive, use_coat_strength=use_coat_strength,
                    use_coat_roughness=use_coat_roughness, use_anisotropy=use_anisotropy, use_occlusion=use_occlusion,
                    render=False
                )
                
                # 아웃라이너 트리의 색상 아이콘도 업데이트
                root = self.tree_widget.invisibleRootItem()
                def _update_icon(item):
                    for i in range(item.childCount()):
                        child = item.child(i)
                        if child.text(0) == name:
                            pixmap = QPixmap(16, 16)
                            pixmap.fill(QColor(int(color[0]*255), int(color[1]*255), int(color[2]*255)))
                            child.setIcon(0, QIcon(pixmap))
                        _update_icon(child)
                _update_icon(root)
                count += 1
                
        if count > 0:
            # 모든 오브젝트의 머티리얼 적용이 완료된 후 단 1회만 최적 렌더링
            self.viewport.render_active()
            log_action("APPLY_MATERIAL_SUCCESS", f"머티리얼: '{name_str}', 적용 부품 수: {count}개, Roughness: {roughness:.2f}, Metallic: {metallic:.2f}")
            
            total_parts = len(getattr(self, 'part_meshes', {}))
            msg = f"'{name_str}' 머티리얼 (Roughness: {roughness:.2f}, Metallic: {metallic:.2f})이 아웃라이너에서 체크된 {count}개 오브젝트에 정상 반영되었습니다."
            if count < total_parts:
                msg += f"\n\n💡 [안내] 현재 전체 {total_parts}개 부품 중 {count}개만 체크되어 있어 체크된 부품에만 재질이 변경되었습니다.\n차체 외관을 포함한 전체 오브젝트에 일괄 적용하려면 아웃라이너 좌측 상단의 [☑ 전체 선택] 버튼을 누른 후 머티리얼을 적용하세요!"
            QMessageBox.information(self, "PBR 재질 적용 완료", msg)

    def on_load_hdri(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "HDRI 맵 불러오기", "", "HDR/EXR Files (*.hdr *.exr);;All Images (*.png *.jpg)")
        if file_path:
            log_action("LOAD_HDRI_START", f"파일 경로: {file_path}")
            success = self.viewport.set_hdri_environment(file_path)
            if success:
                log_action("LOAD_HDRI_SUCCESS", f"적용 성공: {os.path.basename(file_path)}")
                QMessageBox.information(self, "Success", "HDRI 환경 맵이 성공적으로 적용되었습니다.")
            else:
                log_action("LOAD_HDRI_FAIL", f"적용 실패: {os.path.basename(file_path)}")
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
        """현재 프로젝트 상태(메시, 아웃라이너, 개별 PBR 재질 수치, 텍스처 파일, 재질 슬롯 목록 등 모든 작업 내용)를 .anf 형식으로 저장"""
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Project", "", "AI NanoStudio Format (*.anf)")
        if not file_path: return False
        import json, zipfile, tempfile, shutil, os
        import pyvista as pv
        from PySide6.QtWidgets import QApplication
        from PySide6.QtCore import Qt

        part_items = list(self.part_meshes.items()) if hasattr(self, 'part_meshes') and self.part_meshes else []
        total_mesh_count = len(part_items)
        max_progress_val = total_mesh_count + 10

        progress = self._create_progress_dialog(
            "💾 ANF 프로젝트 저장 중",
            "프로젝트 패키징 준비 중...",
            max_val=max_progress_val
        )
        progress.setValue(1)
        QApplication.processEvents()
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                textures_dir = os.path.join(tmpdir, "textures")
                meshes_dir = os.path.join(tmpdir, "meshes")
                os.makedirs(textures_dir, exist_ok=True)
                os.makedirs(meshes_dir, exist_ok=True)
                
                saved_file_map = {}
                
                def bundle_file(src_path):
                    if not src_path or not os.path.exists(src_path):
                        return None
                    abs_src = os.path.abspath(src_path)
                    if abs_src in saved_file_map:
                        return saved_file_map[abs_src]
                    
                    filename = os.path.basename(abs_src)
                    dest_name = filename
                    counter = 1
                    dest_path = os.path.join(textures_dir, dest_name)
                    while os.path.exists(dest_path):
                        name_no_ext, ext_str = os.path.splitext(filename)
                        dest_name = f"{name_no_ext}_{counter}{ext_str}"
                        dest_path = os.path.join(textures_dir, dest_name)
                        counter += 1
                    try:
                        shutil.copy2(abs_src, dest_path)
                        rel_path = f"textures/{dest_name}"
                        saved_file_map[abs_src] = rel_path
                        return rel_path
                    except Exception as err:
                        print(f"파일 패키징 실패 ({src_path}): {err}")
                        return None

                def process_mat_dict(m_dict):
                    if not m_dict or not isinstance(m_dict, dict):
                        return m_dict
                    m_copy = dict(m_dict)
                    path_keys = [
                        ("hdri_path", "hdri_rel_path"),
                        ("texture_path", "texture_rel_path"),
                        ("roughness_map", "roughness_rel_map"),
                        ("metallic_map", "metallic_rel_map"),
                        ("normal_map", "normal_rel_map"),
                        ("disp_map", "disp_rel_map"),
                        ("orm_map", "orm_rel_map"),
                    ]
                    for abs_k, rel_k in path_keys:
                        if m_copy.get(abs_k):
                            rel_p = bundle_file(m_copy[abs_k])
                            if rel_p:
                                m_copy[rel_k] = rel_p
                    return m_copy

                metadata = {
                    "version": "2.1",
                    "parts": [],
                    "mesh_files": {}, # part_name -> rel_vtp_path
                    "colors": {},
                    "visibility": {},
                    "part_materials": {},
                    "slot_materials": [],
                    "viewport_hdri_rel_path": None,
                    "last_applied_material": None
                }

                # 1. 아웃라이너 부품 계층 및 가시성 수집 (재귀적 탐색)
                progress.setLabelText("계층 구조 및 부품 정보 수집 중...")
                progress.setValue(2)
                QApplication.processEvents()

                metadata["tree_hierarchy"] = {}
                def collect_tree_items(parent_item, parent_tag=None):
                    for i in range(parent_item.childCount()):
                        item = parent_item.child(i)
                        key = item.data(0, Qt.UserRole) or item.text(0)
                        
                        if item.childCount() > 0:
                            tag_clean = key.split(" (")[0].strip()
                            if tag_clean not in metadata["tree_hierarchy"]:
                                metadata["tree_hierarchy"][tag_clean] = []
                            collect_tree_items(item, parent_tag=tag_clean)
                        else:
                            if key not in metadata["parts"]:
                                metadata["parts"].append(key)
                            metadata["visibility"][key] = (item.checkState(0) == Qt.Checked)
                            if parent_tag:
                                if parent_tag not in metadata["tree_hierarchy"]:
                                    metadata["tree_hierarchy"][parent_tag] = []
                                metadata["tree_hierarchy"][parent_tag].append(key)

                collect_tree_items(self.tree_widget.invisibleRootItem())

                # 2. part_meshes에 등록된 모든 메시 100% 완전 저장 (실시간 Progress & processEvents)
                step_val = 3
                for idx, (name, mesh) in enumerate(part_items):
                    disp_name = (name[:30] + '...') if len(name) > 30 else name
                    progress.setLabelText(f"3D 메시 데이터 보존 중 ({idx+1}/{total_mesh_count}):\n{disp_name}")
                    progress.setValue(min(step_val, total_mesh_count + 3))
                    QApplication.processEvents()
                    step_val += 1

                    if name not in metadata["parts"]:
                        metadata["parts"].append(name)
                        metadata["visibility"][name] = True
                    if hasattr(self, 'part_colors') and name in self.part_colors:
                        metadata["colors"][name] = self.part_colors[name]
                    
                    vtp_filename = f"part_{idx}.vtp"
                    vtp_rel_path = f"meshes/{vtp_filename}"
                    vtp_abs_path = os.path.join(meshes_dir, vtp_filename)
                    try:
                        mesh.save(vtp_abs_path)
                        metadata["mesh_files"][name] = vtp_rel_path
                    except Exception as e_vtp:
                        print(f"메시 VTP 저장 실패 ({name}): {e_vtp}")

                # 3. 개별 부품별 PBR 재질 수치 및 HDRI / 텍스처 경로 저장
                progress.setLabelText("PBR 재질 및 텍스처 맵 패키징 중...")
                progress.setValue(total_mesh_count + 4)
                QApplication.processEvents()

                if hasattr(self, 'part_materials') and self.part_materials:
                    for p_name, m_info in self.part_materials.items():
                        metadata["part_materials"][p_name] = process_mat_dict(m_info)

                # 4. 하단 재질 슬롯 목록 (최대 100개 슬롯의 재질 이름, 수치, HDRI/텍스처 패키징)
                if hasattr(self, 'mat_slot_panel') and hasattr(self.mat_slot_panel, 'materials'):
                    for slot_info in self.mat_slot_panel.materials:
                        metadata["slot_materials"].append(process_mat_dict(slot_info))

                # 5. 메인 뷰포트 배경 HDRI 경로 패키징
                if hasattr(self.viewport, 'hdri_path') and self.viewport.hdri_path:
                    metadata["viewport_hdri_rel_path"] = bundle_file(self.viewport.hdri_path)

                # 6. 최근 적용된 재질 정보 저장
                if getattr(self, 'last_applied_material', None) is not None:
                    metadata["last_applied_material"] = process_mat_dict(self.last_applied_material)

                # 메타데이터 JSON 파일 기록
                progress.setLabelText("프로젝트 메타데이터 기록 중...")
                progress.setValue(total_mesh_count + 6)
                QApplication.processEvents()

                with open(os.path.join(tmpdir, "metadata.json"), "w", encoding="utf-8") as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)

                # 전체 tempdir 압축 (.anf)
                progress.setLabelText("ANF 아카이브 압축 파일 빌드 중...")
                progress.setValue(total_mesh_count + 8)
                QApplication.processEvents()

                file_list = []
                for root_dir, _, files in os.walk(tmpdir):
                    for f in files:
                        file_list.append(os.path.join(root_dir, f))

                with zipfile.ZipFile(file_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                    for i, abs_path in enumerate(file_list):
                        zf.write(abs_path, arcname=os.path.relpath(abs_path, tmpdir))
                        if i % 10 == 0:
                            QApplication.processEvents()

                progress.setValue(max_progress_val)
                progress.setLabelText("프로젝트 저장 완료!")
                QApplication.processEvents()

            progress.close()
            QMessageBox.information(self, "저장 완료", "모든 메시, 텍스처 파일, PBR 재질 수치 및 슬롯 정보가 성공적으로 저장되었습니다.")
            self.update_window_title(file_path)
            return True
        except Exception as e:
            if 'progress' in locals(): progress.close()
            QMessageBox.critical(self, "저장 오류", f"프로젝트 저장 중 오류 발생:\n{e}")
            return False

    def reset_project(self):
        """현재 씬 및 모든 데이터를 초기화합니다."""
        self.loaded_actors = {}
        if hasattr(self, 'part_meshes'):
            self.part_meshes = {}
        if hasattr(self, 'part_colors'):
            self.part_colors = {}
        if hasattr(self, 'part_materials'):
            self.part_materials = {}
        self.tree_widget.clear()
        self.viewport.clear_and_setup()
        self.btn_ai_texture.setEnabled(False)
        self.btn_unwrap.setEnabled(False)
        self.update_window_title()
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
        """저장된 .anf 프로젝트 불러오기 (메시, PBR 재질 수치, 텍스처 파일, 재질 슬롯 목록 100% 복원)"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Project", "", "AI NanoStudio Format (*.anf)")
        if not file_path: return
        import json, zipfile, os
        import pyvista as pv

        progress = self._create_progress_dialog(
            "ANF 프로젝트 불러오기",
            f"프로젝트 패키지 압축 해제 중...\n{os.path.basename(file_path)}",
            max_val=100
        )
        progress.setValue(5)
        QApplication.processEvents()
        
        try:
            log_action("LOAD_ANF_START", f"프로젝트 불러오기 시작: {file_path}")
            cache_dir = os.path.join(BASE_DIR, "logs", "anf_cache")
            os.makedirs(cache_dir, exist_ok=True)

            progress.setValue(10)
            progress.setLabelText("프로젝트 파일 패키지 및 텍스처 압축 해제 중...")
            QApplication.processEvents()

            with zipfile.ZipFile(file_path, 'r') as zf:
                zf.extractall(cache_dir)
            with open(os.path.join(cache_dir, "metadata.json"), "r", encoding="utf-8") as f:
                metadata = json.load(f)

            progress.setValue(20)
            progress.setLabelText("메타데이터 분석 및 3D 부품 리스트 구성 중...")
            QApplication.processEvents()

            def resolve_bundled_file(rel_path):
                if not rel_path:
                    return None
                abs_p = os.path.join(cache_dir, rel_path)
                if os.path.exists(abs_p):
                    return abs_p
                return None

            def restore_mat_dict(m_dict):
                if not m_dict or not isinstance(m_dict, dict):
                    return m_dict
                m_copy = dict(m_dict)
                path_keys = [
                    ("hdri_path", "hdri_rel_path"),
                    ("texture_path", "texture_rel_path"),
                    ("roughness_map", "roughness_rel_map"),
                    ("metallic_map", "metallic_rel_map"),
                    ("normal_map", "normal_rel_map"),
                    ("disp_map", "disp_rel_map"),
                    ("orm_map", "orm_rel_map"),
                ]
                for abs_k, rel_k in path_keys:
                    if m_copy.get(rel_k):
                        abs_p = resolve_bundled_file(m_copy[rel_k])
                        if abs_p:
                            m_copy[abs_k] = abs_p
                return m_copy

            self.part_meshes = {}
            self.part_colors = {}
            self.part_materials = {}
            self.tree_widget.clear()
            
            # 기존 뷰포트 초기화 및 조명/그리드 완벽 재설정
            self.viewport.clear_and_setup()

            # 1. 하단 재질 슬롯 목록 (100개 슬롯) 복원
            if metadata.get("slot_materials") and hasattr(self, 'mat_slot_panel'):
                restored_slots = []
                for slot_info in metadata["slot_materials"]:
                    restored_slots.append(restore_mat_dict(slot_info))
                self.mat_slot_panel.materials = restored_slots
                self.mat_slot_panel.rebuild_grid_ui()

            # 2. 메인 뷰포트 HDRI 배경 복원
            v_hdri_rel = metadata.get("viewport_hdri_rel_path")
            if v_hdri_rel:
                abs_v_hdri = resolve_bundled_file(v_hdri_rel)
                if abs_v_hdri and os.path.exists(abs_v_hdri):
                    self.viewport.set_hdri_environment(abs_v_hdri, force=True)

            # 3. 부품 메시, 색상, 가시성 및 개별 PBR 재질 수치 복원
            mesh_files = metadata.get("mesh_files") or {}
            parts_meta = metadata.get("parts", [])
            tree_hier_meta = metadata.get("tree_hierarchy") or {}

            # mesh_files short-name to full-key lookup table 구성
            short_to_full_map = {}
            for k in mesh_files.keys():
                if "/" in k:
                    sub_part = k.split("/", 1)[1]
                    short_to_full_map[sub_part] = k
                    short_to_full_map[k] = k
                else:
                    short_to_full_map[k] = k

            # 복원 대상 전체 메시 키 목록 수집 (중복 제거)
            all_part_keys = []
            seen_keys = set()

            def register_key(raw_k):
                if not raw_k or raw_k.endswith(" Groups)") or (" (" in raw_k and "Groups" in raw_k):
                    return
                resolved_k = short_to_full_map.get(raw_k, raw_k)
                if resolved_k not in seen_keys:
                    seen_keys.add(resolved_k)
                    all_part_keys.append(resolved_k)

            if mesh_files:
                for k in mesh_files.keys():
                    register_key(k)
            for k in parts_meta:
                register_key(k)

            self.tree_widget.blockSignals(True)
            self.tree_widget.setUpdatesEnabled(False)
            self.tree_widget.clear()

            tag_items = {}
            icon_cache = {}
            total_parts = len(all_part_keys)

            for idx_k, name in enumerate(all_part_keys):
                if total_parts > 0 and (idx_k % 10 == 0 or idx_k == total_parts - 1):
                    pct = 20 + int((idx_k / total_parts) * 70) # 20% ~ 90%
                    progress.setValue(pct)
                    progress.setLabelText(f"3D 메쉬 및 계층 구조 복원 중... ({idx_k + 1}/{total_parts})")
                    QApplication.processEvents()

                # VTP 파일 경로 읽기
                mesh_rel_path = mesh_files.get(name)
                if not mesh_rel_path:
                    sub_k = name.split("/")[-1] if "/" in name else name
                    mesh_rel_path = mesh_files.get(sub_k)

                if mesh_rel_path:
                    mesh_path = os.path.join(cache_dir, mesh_rel_path)
                else:
                    clean_name = name.replace('/', '_')
                    mesh_path = os.path.join(cache_dir, f"{name}.vtp")
                    if not os.path.exists(mesh_path):
                        mesh_path = os.path.join(cache_dir, f"{clean_name}.vtp")
                        if not os.path.exists(mesh_path):
                            mesh_path = os.path.join(cache_dir, f"meshes/{clean_name}.vtp")

                if os.path.exists(mesh_path):
                    try:
                        self.part_meshes[name] = pv.read(mesh_path)
                    except Exception as err_m:
                        print(f"메시 로드 실패 ({name}): {err_m}")

                sub_k = name.split("/")[-1] if "/" in name else name
                color = metadata.get("colors", {}).get(name, metadata.get("colors", {}).get(sub_k, [0.82, 0.82, 0.82]))
                self.part_colors[name] = color

                mat_dict = metadata.get("part_materials", {})
                if mat_dict and (name in mat_dict or sub_k in mat_dict):
                    m_data = mat_dict.get(name) or mat_dict.get(sub_k)
                    self.part_materials[name] = restore_mat_dict(m_data)

                vis_dict = metadata.get("visibility", {})
                visibility = vis_dict.get(name, vis_dict.get(sub_k, True))

                # 색상 아이콘 캐싱
                rgb_key = (int(color[0]*255), int(color[1]*255), int(color[2]*255))
                if rgb_key not in icon_cache:
                    pixmap = QPixmap(16, 16)
                    pixmap.fill(QColor(*rgb_key))
                    icon_cache[rgb_key] = QIcon(pixmap)
                icon = icon_cache[rgb_key]

                # 트리 그룹 및 자식 노드 등록
                tag_name = ""
                sub_name = name

                if "/" in name:
                    tag_name, sub_name = name.split("/", 1)
                else:
                    for p_tag, c_keys in tree_hier_meta.items():
                        if name in c_keys or any(c.endswith("/" + name) for c in c_keys):
                            tag_name = p_tag
                            break

                if tag_name:
                    if tag_name not in tag_items:
                        parent_item = QTreeWidgetItem(self.tree_widget, [tag_name])
                        parent_item.setFlags(parent_item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEditable)
                        parent_item.setCheckState(0, Qt.Checked)
                        parent_item.setData(0, Qt.UserRole, tag_name)
                        parent_item.setIcon(0, icon)
                        tag_items[tag_name] = parent_item
                    else:
                        parent_item = tag_items[tag_name]

                    item = QTreeWidgetItem(parent_item, [sub_name])
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEditable)
                    item.setCheckState(0, Qt.Checked if visibility else Qt.Unchecked)
                    item.setData(0, Qt.UserRole, name)
                    item.setData(0, Qt.UserRole + 2, Qt.Checked if visibility else Qt.Unchecked)
                    item.setIcon(0, icon)
                else:
                    item = QTreeWidgetItem(self.tree_widget, [name])
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsEditable)
                    item.setCheckState(0, Qt.Checked if visibility else Qt.Unchecked)
                    item.setData(0, Qt.UserRole, name)
                    item.setData(0, Qt.UserRole + 2, Qt.Checked if visibility else Qt.Unchecked)
                    item.setIcon(0, icon)

                # 메시가 존재하면 모든 플로터에 등록
                if name in self.part_meshes:
                    mesh = self.part_meshes[name]
                    self.viewport.add_mesh_to_all(mesh, name=name, color=color)

                    if not visibility:
                        for p in self.viewport.plotters:
                            try:
                                p.remove_actor(name)
                            except Exception:
                                pass

                    if name in self.part_materials:
                        mat_info = self.part_materials[name]
                        self.viewport.update_mesh_pbr(
                            name,
                            color=mat_info.get('color', color),
                            roughness=mat_info.get('roughness', 0.5),
                            metallic=mat_info.get('metallic', 0.0),
                            normal_scale=mat_info.get('normal_scale', 1.0),
                            bright=mat_info.get('bright', 1.0),
                            reflection=mat_info.get('reflection', 1.0),
                            refraction=mat_info.get('refraction', 1.5),
                            emissive=mat_info.get('emissive', 0.0),
                            coat_strength=mat_info.get('coat_strength', 0.0),
                            coat_roughness=mat_info.get('coat_roughness', 0.0),
                            anisotropy=mat_info.get('anisotropy', 0.0),
                            occlusion=mat_info.get('occlusion', 1.0),
                            use_hdri=mat_info.get('use_hdri', True),
                            hdri_ratio=mat_info.get('hdri_ratio', 1.0),
                            use_roughness=mat_info.get('use_roughness', True),
                            use_metallic=mat_info.get('use_metallic', True),
                            use_normal=mat_info.get('use_normal', True),
                            use_bright=mat_info.get('use_bright', True),
                            use_reflection=mat_info.get('use_reflection', True),
                            use_refraction=mat_info.get('use_refraction', True),
                            use_emissive=mat_info.get('use_emissive', False),
                            use_coat_strength=mat_info.get('use_coat_strength', False),
                            use_coat_roughness=mat_info.get('use_coat_roughness', False),
                            use_anisotropy=mat_info.get('use_anisotropy', False),
                            use_occlusion=mat_info.get('use_occlusion', False),
                            render=False
                        )

            progress.setValue(92)
            progress.setLabelText("객체 계층 구조 트리 및 헤더 갱신 중...")
            QApplication.processEvents()

            # 각 부모 그룹 항목의 텍스트를 카운트 헤더로 업데이트 (예: "지줄가구부 (189 Groups)")
            for tag_name, parent_item in tag_items.items():
                c_cnt = parent_item.childCount()
                if c_cnt > 0:
                    parent_item.setText(0, f"{tag_name} ({c_cnt} Groups)")

            self.tree_widget.expandAll()
            self.tree_widget.setUpdatesEnabled(True)
            self.tree_widget.blockSignals(False)

            # 4. 최근 적용된 재질 복원
            if metadata.get("last_applied_material"):
                self.last_applied_material = restore_mat_dict(metadata["last_applied_material"])

            # 5. 디스플레이 렌더 모드 동기화 (기본: material_preview)
            curr_mode = getattr(self.viewport, 'current_render_mode', 'material_preview')
            self.viewport.set_display_render_mode(curr_mode)

            progress.setValue(98)
            progress.setLabelText("뷰포트 렌더링 및 카메라 최종 세팅 중...")
            QApplication.processEvents()

            # 6. 카메라 및 뷰포트 전체 재설정 (복원된 모든 객체가 한눈에 꽉 차게 보이도록)
            self.viewport.reset_camera()
            self.update_dimensions_text()
            self.update_window_title(file_path)

            progress.setValue(100)
            progress.close()

            log_action("LOAD_ANF_SUCCESS", f"프로젝트 복원 성공: {os.path.basename(file_path)} | 부품 수: {len(self.part_meshes)}개")
            QMessageBox.information(self, "불러오기 완료", f"프로젝트({len(self.part_meshes)}개 부품, 색상 및 PBR 설정)를 성공적으로 불러왔습니다.")
        except Exception as e:
            if 'progress' in locals() and progress:
                progress.close()
            log_action("LOAD_ANF_ERROR", f"불러오기 실패: {e}")
            QMessageBox.critical(self, "불러오기 오류", f"프로젝트 불러오기 중 오류 발생:\n{e}")
        finally:
            if 'progress' in locals() and progress:
                try:
                    progress.close()
                except Exception:
                    pass

    def get_model_center(self):
        """전체 모델의 통합 중심점(Center) 반환 (개별 로컬 부품 오목함 반전 버그 방지)"""
        if not hasattr(self, 'part_meshes') or not self.part_meshes:
            return [0.0, 0.0, 0.0]
        try:
            bounds = [float('inf'), float('-inf'), float('inf'), float('-inf'), float('inf'), float('-inf')]
            for m in self.part_meshes.values():
                if m is not None and hasattr(m, 'bounds'):
                    b = m.bounds
                    bounds[0] = min(bounds[0], b[0])
                    bounds[1] = max(bounds[1], b[1])
                    bounds[2] = min(bounds[2], b[2])
                    bounds[3] = max(bounds[3], b[3])
                    bounds[4] = min(bounds[4], b[4])
                    bounds[5] = max(bounds[5], b[5])
            if bounds[0] != float('inf'):
                return [(bounds[0]+bounds[1])/2.0, (bounds[2]+bounds[3])/2.0, (bounds[4]+bounds[5])/2.0]
        except Exception:
            pass
        return [0.0, 0.0, 0.0]

    def ensure_frontfaces_oriented(self, mesh, name=None, force=False):
        """3D 메쉬의 법선(Normal)을 매끈하고 일관되게 정렬하며, 타이어/내부 면 등 오목한 유기적 입체가 뒤집히거나 구멍이 뚫리지 않도록 100% 매니폴드 정방향 법선 보정
        """
        if mesh is None or not hasattr(mesh, 'n_cells') or mesh.n_cells == 0:
            return mesh
            
        # 이미 면 방향 락(Lock)이 걸려있는 메쉬이고 강제(force=True)가 아니면 재반전을 차단하고 그대로 유지
        is_locked = False
        if not force:
            if hasattr(mesh, 'user_dict') and isinstance(mesh.user_dict, dict) and mesh.user_dict.get('frontface_locked', False):
                is_locked = True
            elif name and hasattr(self, 'frontface_locked_parts') and name in self.frontface_locked_parts:
                is_locked = True
                
        if is_locked:
            return mesh

        try:
            import numpy as np
            
            # 1. 메시 전반의 위상 일관성(Manifold Consistency) 보장 법선 계산 (개별 삼각형을 찢지 않음)
            oriented = mesh.compute_normals(
                cell_normals=True,
                point_normals=True,
                consistent_normals=True,
                auto_orient_normals=True,
                split_vertices=False,
                feature_angle=60,
                inplace=False
            )
            
            # 2. 오브젝트 자체의 중심(part_center) 대비 외곽 표면의 대다수가 반전(Inverted)된 메쉬인지 검사
            # (개별 삼각셀 단위로 쪼개서 뒤집는 대신 전체 메쉬 위상을 일관되게 유지)
            part_center = oriented.center
            cell_centers = oriented.cell_centers().points
            cell_normals = oriented.cell_data.get('Normals')
            
            if cell_normals is not None and len(cell_normals) == len(cell_centers):
                vecs = cell_centers - np.array(part_center)
                norms = np.linalg.norm(vecs, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                unit_vecs = vecs / norms
                
                dots = np.sum(unit_vecs * cell_normals, axis=1)
                
                # 외각 표면의 65% 이상이 파트 중심을 향하고 있다면 전체 메쉬 순서를 통으로 반전
                if np.mean(dots < 0) > 0.65:
                    faces = oriented.faces.copy()
                    offset = 0
                    while offset < len(faces):
                        n_verts = faces[offset]
                        if n_verts >= 3:
                            faces[offset + 1], faces[offset + 2] = faces[offset + 2], faces[offset + 1]
                        offset += (n_verts + 1)
                    oriented.faces = faces
                    oriented = oriented.compute_normals(
                        cell_normals=True,
                        point_normals=True,
                        consistent_normals=True,
                        auto_orient_normals=True,
                        split_vertices=False,
                        feature_angle=60,
                        inplace=False
                    )
            
            # 3. 정렬 완료 후 락(Lock) 상태 등록
            if not hasattr(oriented, 'user_dict') or oriented.user_dict is None:
                oriented.user_dict = {}
            oriented.user_dict['frontface_locked'] = True
            if name and hasattr(self, 'frontface_locked_parts'):
                self.frontface_locked_parts.add(name)
                
            return oriented
        except Exception as e:
            print(f"Normal orient error: {e}")
            return mesh

    def orient_all_frontfaces(self, silent=False):
        """스케치업/CAD 모델의 모든 뒤집힌 면(Backface)을 외부 앞면(Frontface) 방향으로 100% 자동 통일 정렬 및 락(Lock) 고정"""
        if not hasattr(self, 'part_meshes') or not self.part_meshes:
            if not silent:
                QMessageBox.information(self, "데이터 없음", "먼저 3D 모델을 불러와주세요.")
            return

        # 아웃라이너 체크 여부와 관계없이 모델의 '전체 부품 메시'에 대해 일괄 면 방향 통일 처리
        target_names = list(self.part_meshes.keys())

        if not target_names:
            if not silent:
                QMessageBox.information(self, "대상 없음", "면 방향을 정렬할 오브젝트가 선택되지 않았습니다.")
            return

        count = 0
        for name in target_names:
            if name in self.part_meshes and self.part_meshes[name].n_cells > 0:
                try:
                    mesh = self.part_meshes[name]
                    # 수동 통일 실행 시 force=True로 면 방향 재배향 후 락(Lock) 설정
                    fixed_mesh = self.ensure_frontfaces_oriented(mesh, name=name, force=True)
                    self.part_meshes[name] = fixed_mesh
                    color = self.part_colors.get(name, [0.8, 0.8, 0.8])
                    self.viewport.update_mesh(name, fixed_mesh, color)
                    
                    mat_info = None
                    if hasattr(self, 'part_materials') and name in self.part_materials:
                        mat_info = self.part_materials[name]
                    elif getattr(self, 'last_applied_material', None) is not None:
                        mat_info = self.last_applied_material

                    if mat_info:
                        self.viewport.update_mesh_pbr(
                            name,
                            color=mat_info.get('color', color),
                            roughness=mat_info.get('roughness', 0.5),
                            metallic=mat_info.get('metallic', 0.0),
                            normal_scale=mat_info.get('normal_scale', 1.0),
                            bright=mat_info.get('bright', 1.0),
                            reflection=mat_info.get('reflection', 1.0),
                            refraction=mat_info.get('refraction', 1.5),
                            emissive=mat_info.get('emissive', 0.0),
                            coat_strength=mat_info.get('coat_strength', 0.0),
                            coat_roughness=mat_info.get('coat_roughness', 0.0),
                            anisotropy=mat_info.get('anisotropy', 0.0),
                            occlusion=mat_info.get('occlusion', 1.0),
                            use_hdri=mat_info.get('use_hdri', True),
                            hdri_ratio=mat_info.get('hdri_ratio', 1.0),
                            use_roughness=mat_info.get('use_roughness', True),
                            use_metallic=mat_info.get('use_metallic', True),
                            use_normal=mat_info.get('use_normal', True),
                            use_bright=mat_info.get('use_bright', True),
                            use_reflection=mat_info.get('use_reflection', True),
                            use_refraction=mat_info.get('use_refraction', True),
                            use_emissive=mat_info.get('use_emissive', False),
                            use_coat_strength=mat_info.get('use_coat_strength', False),
                            use_coat_roughness=mat_info.get('use_coat_roughness', False),
                            use_anisotropy=mat_info.get('use_anisotropy', False),
                            use_occlusion=mat_info.get('use_occlusion', False),
                            render=False
                        )
                    count += 1
                except Exception as e:
                    print(f"[{name}] 면 앞면 통일 처리 실패: {e}")

        if count > 0:
            self.sync_outliner_visibility()
            log_action("ORIENT_FRONTFACES_SUCCESS", f"면 방향 통일 완료: {count}개 부품")
            if not silent:
                QMessageBox.information(
                    self, "면 앞면 통일 완료", 
                    f"전체 {count}개 부품 오브젝트의 모든 반전된 면(Backface)을 외부 앞면(Frontface) 방향으로 100% 자동 통일 정렬하였습니다.\n\n"
                    f"이제 개별 부품을 체크/체크해제하거나 PBR 머티리얼을 적용해도 면 뚫림 현상 없이 표현됩니다."
                )

    def on_btn_auto_fix(self):
        """메시 자동 최적화 및 스무딩 다이얼로그 호출 (계층구조에서 체크된 오브젝트 대상)"""
        from PySide6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QCheckBox, QPushButton
        from PySide6.QtCore import Qt
        
        if not hasattr(self, 'part_meshes') or not self.part_meshes:
            QMessageBox.information(self, "데이터 없음", "먼저 3D 모델을 불러와주세요.")
            return

        checked_names = self.get_checked_item_names()
        if not checked_names:
            QMessageBox.information(self, "대상 없음", "계층구조(Outliner)에서 Auto-Fix를 적용할 오브젝트의 체크박스를 1개 이상 선택해 주세요.")
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Auto-Fix Mesh (형태 최적화)")
        dialog.resize(400, 260)
        layout = QVBoxLayout(dialog)
        
        # 0. 적용 대상 안내 라벨
        lbl_target = QLabel(f"📌 적용 대상: 계층구조에서 체크된 {len(checked_names)}개 오브젝트")
        lbl_target.setStyleSheet("font-weight: bold; color: #1E88E5; padding-bottom: 5px;")
        layout.addWidget(lbl_target)
        
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
                
                # 실행 시점의 최신 체크 상태 반영
                current_checked_names = self.get_checked_item_names()
                
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
                
                # 처리할 메시와 인덱스 수집 (체크 표시된 오브젝트로만 제한)
                targets = {} # {name: cell_indices_to_process or None for all}
                if has_picked:
                    for name, mesh in self.part_meshes.items():
                        if name not in current_checked_names:
                            continue
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
                    # 선택된 면이 없으면 체크된 메시 전체를 대상으로 함
                    for name, mesh in self.part_meshes.items():
                        if name in current_checked_names and mesh.n_cells > 0:
                            targets[name] = None
                            
                if not targets:
                    QMessageBox.information(dialog, "선택 영역 없음", "체크된 오브젝트 중 처리할 유효한 폴리곤을 찾지 못했습니다.")
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
                QMessageBox.information(self, "완료", f"체크된 {len(targets)}개 오브젝트의 메쉬가 성공적으로 최적화 되었습니다.")
            except Exception as e:
                QMessageBox.critical(self, "오류", f"최적화 중 오류가 발생했습니다:\n{e}")
            finally:
                QApplication.restoreOverrideCursor()
                
        btn_apply.clicked.connect(apply_fixes)
        btn_cancel.clicked.connect(dialog.reject)
        
        dialog.exec()

    # ── 시네마틱 디렉터 핸들러 (FR-08, FR-09, FR-10) ──────────────────

    def on_set_camera_keyframe(self, slot):
        """FR-08: 현재 뷰포트 카메라 앵글을 시작점(A) 또는 종료점(B)으로 등록"""
        try:
            plotter = self.viewport.plotter_single
            pose = self.cinematic_director.set_keyframe(slot, plotter)
            self.lbl_cam_status.setText(self.cinematic_director.get_registration_status())
            
            pos = pose["position"]
            slot_name = "시작점(A)" if slot == 'A' else "종료점(B)"
            QMessageBox.information(
                self, "카메라 등록 완료",
                f"{slot_name} 카메라 앵글이 등록되었습니다.\n"
                f"위치: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
            )
            log_action("CAMERA_KEYFRAME", f"{slot_name} 등록: {pos}")
        except Exception as e:
            QMessageBox.warning(self, "오류", f"카메라 등록 중 오류 발생:\n{e}")

    def on_generate_veo_prompt(self):
        """FR-09: Gemini Cinematic Director Agent를 호출하여 Veo 모션 프롬프트 생성"""
        if not GEMINI_API_KEY:
            QMessageBox.warning(self, "API Key", ".env 파일에 GEMINI_API_KEY가 설정되어 있지 않습니다.")
            return
            
        if not self.cinematic_director.has_both_keyframes():
            QMessageBox.warning(self, "카메라 미등록", "시작점(A)과 종료점(B)을 모두 먼저 등록해주세요.")
            return
        
        try:
            motion_vector = self.cinematic_director.compute_motion_vector()
            scene_desc = self.input_scene_concept.text().strip()
            if not scene_desc:
                scene_desc = "Modern architectural building with clean geometric forms"
            
            # 한글 요약 먼저 표시
            summary = self.cinematic_director.get_summary_text()
            self.lbl_veo_result.setText(f"분석 중... {summary}")
            
            # Gemini 비동기 워커 호출
            self._veo_worker = VeoPromptWorker(
                start_cam=self.cinematic_director.keyframe_a,
                end_cam=self.cinematic_director.keyframe_b,
                motion_vector=motion_vector,
                scene_desc=scene_desc,
                api_key=GEMINI_API_KEY,
            )
            self._veo_worker.prompt_ready.connect(self._on_veo_prompt_ready)
            self._veo_worker.error.connect(self._on_veo_error)
            self._veo_worker.start()
            
            self.btn_generate_veo.setEnabled(False)
            log_action("VEO_PROMPT_REQUEST", f"씬: {scene_desc}")
        except Exception as e:
            QMessageBox.warning(self, "오류", f"프롬프트 생성 요청 중 오류:\n{e}")

    def _on_veo_prompt_ready(self, prompt_text):
        """Veo 프롬프트 생성 완료 시 호출"""
        self.btn_generate_veo.setEnabled(True)
        self._last_veo_prompt = prompt_text
        
        summary = self.cinematic_director.get_summary_text()
        self.lbl_veo_result.setText(f"✅ {summary}\n\n📝 Veo Prompt:\n{prompt_text}")
        log_action("VEO_PROMPT_READY", prompt_text[:100])
        
        QMessageBox.information(
            self, "Veo 프롬프트 생성 완료",
            f"Cinematic Director Agent가 모션 프롬프트를 생성했습니다.\n\n{prompt_text[:200]}..."
            if len(prompt_text) > 200 else
            f"Cinematic Director Agent가 모션 프롬프트를 생성했습니다.\n\n{prompt_text}"
        )

    def _on_veo_error(self, err_msg):
        """Veo 프롬프트 생성 실패 시 호출"""
        self.btn_generate_veo.setEnabled(True)
        self.lbl_veo_result.setText(f"❌ 오류: {err_msg}")
        QMessageBox.warning(self, "Director Agent 오류", err_msg)

    def on_export_conti_pdf(self):
        """FR-10: First/Last Frame PNG + 모션 프롬프트 + 콘티 PDF 일괄 추출"""
        if not self.cinematic_director.has_both_keyframes():
            QMessageBox.warning(self, "카메라 미등록", "시작점(A)과 종료점(B)을 모두 먼저 등록해주세요.")
            return
        
        try:
            from datetime import datetime as dt
            
            # 출력 디렉토리 설정
            today_str = dt.now().strftime("%Y-%m-%d")
            output_dir = os.path.join(BASE_DIR, "Output", today_str, "Cinematic")
            os.makedirs(output_dir, exist_ok=True)
            
            # 1. First/Last Frame 캡처
            plotter = self.viewport.plotter_single
            start_path, end_path = self.cinematic_director.capture_frames(plotter, output_dir)
            
            # 2. 카메라 데이터 수집
            camera_data = self.cinematic_director.get_camera_data_for_pdf()
            
            # 3. 씬 콘셉트 및 프롬프트
            scene_desc = self.input_scene_concept.text().strip() or "Architectural Scene"
            motion_prompt = getattr(self, '_last_veo_prompt', '(Veo 프롬프트를 먼저 생성해주세요)')
            
            # 4. 프롬프트 텍스트 파일 저장
            timestamp = dt.now().strftime("%H%M%S")
            prompt_txt_path = os.path.join(output_dir, f"motion_prompt_{timestamp}.txt")
            with open(prompt_txt_path, 'w', encoding='utf-8') as f:
                f.write(f"Scene Concept: {scene_desc}\n\n")
                f.write(f"Veo Motion Prompt:\n{motion_prompt}\n\n")
                f.write(f"Camera Data:\n{camera_data}\n")
            
            # 5. 콘티 PDF 생성
            pdf_path = os.path.join(output_dir, f"cinematic_sheet_{timestamp}.pdf")
            if ContiPDFBuilder is not None:
                ContiPDFBuilder.build(
                    start_img_path=start_path,
                    end_img_path=end_path,
                    motion_prompt=motion_prompt,
                    scene_desc=scene_desc,
                    camera_data=camera_data,
                    output_path=pdf_path,
                )
                pdf_msg = f"• 콘티 PDF: {os.path.basename(pdf_path)}"
            else:
                pdf_msg = "• 콘티 PDF: reportlab 미설치로 스킵됨"
            
            QMessageBox.information(
                self, "콘티 일괄 추출 완료",
                f"시네마틱 에셋이 추출되었습니다!\n\n"
                f"• 시작 프레임: {os.path.basename(start_path)}\n"
                f"• 종료 프레임: {os.path.basename(end_path)}\n"
                f"• 프롬프트: {os.path.basename(prompt_txt_path)}\n"
                f"{pdf_msg}\n\n"
                f"저장 경로: {output_dir}"
            )
            log_action("CONTI_EXPORT", f"경로: {output_dir}")
        except Exception as e:
            QMessageBox.warning(self, "콘티 추출 오류", f"추출 중 오류 발생:\n{e}")

    # ── GLB 게임 에셋 익스포트 핸들러 (FR-11) ──────────────────────

    def on_export_glb(self):
        """FR-11: 현재 로드된 모델을 텍스처 내장형 .glb 파일로 내보내기"""
        if not hasattr(self, 'part_meshes') or not self.part_meshes:
            QMessageBox.warning(self, "Export Error", "내보낼 모델이 없습니다. 먼저 모델을 로드하세요.")
            return
        
        try:
            import trimesh
            from datetime import datetime as dt
            
            # 저장 경로 설정
            today_str = dt.now().strftime("%Y-%m-%d")
            default_dir = os.path.join(BASE_DIR, "Output", "GameObject", today_str)
            os.makedirs(default_dir, exist_ok=True)
            default_path = os.path.join(default_dir, "game_asset.glb")
            
            file_path, _ = QFileDialog.getSaveFileName(
                self, "GLB 파일 내보내기", default_path, "GLB Files (*.glb)"
            )
            if not file_path:
                return
            
            # 각 파트를 trimesh로 변환하여 Scene에 추가
            tri_meshes = []
            for name, pv_mesh in self.part_meshes.items():
                try:
                    faces_raw = pv_mesh.faces
                    if len(faces_raw) == 0:
                        continue
                    faces = faces_raw.reshape(-1, 4)[:, 1:]
                    tri_mesh = trimesh.Trimesh(vertices=pv_mesh.points, faces=faces)
                    tri_mesh.fix_normals()
                    
                    # UV가 있으면 바인딩
                    if pv_mesh.active_texture_coordinates is not None:
                        tri_mesh.visual = trimesh.visual.TextureVisuals(
                            uv=pv_mesh.active_texture_coordinates
                        )
                    
                    tri_meshes.append(tri_mesh)
                except Exception as mesh_err:
                    print(f"[GLB] '{name}' 변환 스킵: {mesh_err}")
                    continue
            
            if not tri_meshes:
                QMessageBox.warning(self, "Export Error", "변환 가능한 메시가 없습니다.")
                return
            
            # 텍스처 바인딩 (최종 디퓨즈가 존재하면)
            final_diffuse = os.path.join(BASE_DIR, "final_clothed_diffuse.png")
            if os.path.exists(final_diffuse):
                try:
                    import cv2
                    diffuse_img = cv2.imread(final_diffuse)
                    if diffuse_img is not None:
                        diffuse_rgb = cv2.cvtColor(diffuse_img, cv2.COLOR_BGR2RGB)
                        material = trimesh.visual.texture.SimpleMaterial(image=diffuse_rgb)
                        for tm in tri_meshes:
                            if hasattr(tm.visual, 'uv') and tm.visual.uv is not None:
                                tm.visual = trimesh.visual.TextureVisuals(
                                    uv=tm.visual.uv, material=material
                                )
                except Exception as tex_err:
                    print(f"[GLB] 텍스처 바인딩 실패 (무시): {tex_err}")
            
            # 씬 조합 및 내보내기
            scene = trimesh.Scene(tri_meshes)
            scene.export(file_path)
            
            QMessageBox.information(
                self, "GLB Export 완료",
                f"게임 엔진용 GLB 파일이 저장되었습니다!\n\n"
                f"파일: {file_path}\n"
                f"부품 수: {len(tri_meshes)}개"
            )
            log_action("GLB_EXPORT", f"경로: {file_path}, 부품: {len(tri_meshes)}")
        except Exception as e:
            QMessageBox.critical(self, "GLB Export 오류", f"오류 발생:\n{e}")

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
        try:
            if hasattr(self, 'viewport'):
                for plotter in getattr(self.viewport, 'plotters', []):
                    try:
                        if hasattr(plotter, 'render_timer'):
                            plotter.render_timer.stop()
                    except Exception:
                        pass
        except Exception:
            pass
        event.accept()
        # 창 닫기(X) 시 백그라운드 VTK 스레드 잔여 및 Qt 루프 대기(Hang / 0xCFFFFFFF)를 원천 차단하고 즉시 정상 종료(0)
        os._exit(0)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    DARK_THEME_QSS = """
    QMainWindow, QDialog, QWidget#central_widget {
        background-color: #1e1e24;
        color: #e2e8f0;
        font-family: 'Segoe UI', 'Malgun Gothic', sans-serif;
    }
    QWidget {
        color: #e2e8f0;
    }
    QMenuBar, QMenu {
        background-color: #25252e;
        color: #e2e8f0;
    }
    QMenu::item:selected {
        background-color: #2563eb;
        color: #ffffff;
    }
    QTabWidget::pane {
        border: 1px solid #333842;
        background-color: #23272e;
    }
    QTabBar::tab {
        background-color: #1a1d24;
        color: #94a3b8;
        padding: 8px 16px;
        border-top-left-radius: 4px;
        border-top-right-radius: 4px;
    }
    QTabBar::tab:selected {
        background-color: #23272e;
        color: #38bdf8;
        font-weight: bold;
    }
    QPushButton {
        background-color: #2b303c;
        color: #e2e8f0;
        border: 1px solid #3e4451;
        border-radius: 4px;
        padding: 6px 12px;
    }
    QPushButton:hover {
        background-color: #3b4252;
        border-color: #60a5fa;
    }
    QPushButton:pressed {
        background-color: #1e222a;
    }
    QTreeWidget {
        background-color: #D3D3D3;
        color: #111111;
        border: 1px solid #B0B0B0;
    }
    QTreeWidget::item {
        color: #111111;
    }
    QTreeWidget::item:selected {
        background-color: #2563eb;
        color: #ffffff;
    }
    QTreeWidget::item:hover {
        background-color: #BEBEBE;
        color: #000000;
    }
    QListWidget, QTableWidget {
        background-color: #1b1d23;
        color: #e2e8f0;
        border: 1px solid #333842;
        gridline-color: #2d313b;
    }
    QHeaderView::section {
        background-color: #D3D3D3;
        color: #111111;
        font-weight: bold;
        padding: 4px;
        border: 1px solid #B0B0B0;
    }
    QLineEdit, QTextEdit, QPlainTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {
        background-color: #16181d;
        color: #f1f5f9;
        border: 1px solid #333842;
        border-radius: 4px;
        padding: 4px;
    }
    QComboBox QAbstractItemView {
        background-color: #1e222b;
        color: #e2e8f0;
        selection-background-color: #2563eb;
    }
    QGroupBox {
        border: 1px solid #333842;
        border-radius: 6px;
        margin-top: 10px;
        padding-top: 10px;
        font-weight: bold;
        color: #f1f5f9;
    }
    QGroupBox::title {
        subcontrol-origin: margin;
        subcontrol-position: top left;
        padding: 0 6px;
        color: #38bdf8;
    }
    QScrollBar:vertical {
        background: #181a1f;
        width: 10px;
    }
    QScrollBar::handle:vertical {
        background: #3b4252;
        border-radius: 4px;
    }
    QScrollBar::handle:vertical:hover {
        background: #4c566a;
    }
    """
    app.setStyleSheet(DARK_THEME_QSS)
    activity_logger = install_activity_logger(app)
    log_action("PROGRAM_START", "AI-NanoStudio 애플리케이션 시작")
    if os.path.exists(ICON_PATH):
        app.setWindowIcon(QIcon(ICON_PATH))
    window = MainWindow()
    window.show()
    exit_code = app.exec()
    log_action("PROGRAM_EXIT", f"애플리케이션 정상 종료 (종료 코드: {exit_code})")
    os._exit(exit_code)


