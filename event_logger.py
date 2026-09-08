import os
import sys
import time
import logging
from logging.handlers import RotatingFileHandler
from PySide6.QtCore import QObject, QEvent, Qt
from PySide6.QtWidgets import QApplication

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(BASE_DIR, "logs")
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

LOG_FILE = os.path.join(LOG_DIR, "activity.log")

# 전역 로거 설정
_logger = logging.getLogger("AI_NanoStudio_Activity")
_logger.setLevel(logging.DEBUG)

if not _logger.handlers:
    # 1. 파일 핸들러 (최대 10MB 회전, UTF-8 인코딩)
    file_handler = RotatingFileHandler(LOG_FILE, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8")
    file_fmt = logging.Formatter("[%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_fmt)
    _logger.addHandler(file_handler)

    # 2. 콘솔 출력 핸들러
    console_handler = logging.StreamHandler(sys.stdout)
    console_fmt = logging.Formatter("[%(asctime)s.%(msecs)03d] [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    console_handler.setFormatter(console_fmt)
    _logger.addHandler(console_handler)

def get_logger():
    """애플리케이션 전역 로거 반환"""
    return _logger

def log_action(action_name: str, details: str = ""):
    """주요 작업 진행 상황을 기록하는 헬퍼 함수"""
    msg = f"[ACTION] {action_name}"
    if details:
        msg += f" | {details}"
    _logger.info(msg)

class GlobalActivityFilter(QObject):
    """
    QApplication 전역에 설치되어 마우스 클릭, 드래그, 휠, 키보드 입력을 실시간 로깅하는 필터
    """
    def __init__(self):
        super().__init__()
        self.mouse_press_pos = None
        self.mouse_press_time = 0.0
        self.mouse_press_button = None
        self.last_drag_log_time = 0.0
        self.drag_move_count = 0

    def _get_widget_desc(self, widget):
        """이벤트가 발생한 위젯의 클래스명과 식별 정보를 직관적인 문자열로 반환"""
        if not widget:
            return "UnknownWidget"
        
        name = widget.objectName() or ""
        cls_name = widget.__class__.__name__
        
        # 뷰포트 인터랙터 감지
        if "QVTK" in cls_name or "Interactor" in cls_name:
            parent = widget.parent()
            while parent:
                p_cls = parent.__class__.__name__
                if "ViewportWidget" in p_cls:
                    return "Viewport (Cinematic/Quad View)"
                if "MaterialInspector" in p_cls:
                    return "Material Ball Preview (Inspector)"
                parent = parent.parent()
            return f"Viewport Interactor ({cls_name})"
        
        # 버튼 감지
        if hasattr(widget, "text") and callable(widget.text):
            btn_text = widget.text()
            if btn_text:
                return f"{cls_name}['{btn_text}']"
                
        # 윈도우 제목
        if hasattr(widget, "windowTitle") and callable(widget.windowTitle):
            title = widget.windowTitle()
            if title:
                return f"{cls_name}(Title: '{title}')"

        return f"{cls_name}{f'#{name}' if name else ''}"

    def eventFilter(self, obj, event):
        etype = event.type()

        # ---------------- 1. 마우스 클릭 (Press) ----------------
        if etype == QEvent.MouseButtonPress:
            btn_name = "Left" if event.button() == Qt.LeftButton else "Right" if event.button() == Qt.RightButton else "Middle" if event.button() == Qt.MiddleButton else str(event.button())
            widget_desc = self._get_widget_desc(obj)
            pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
            
            self.mouse_press_pos = pos
            self.mouse_press_time = time.perf_counter()
            self.mouse_press_button = btn_name
            self.last_drag_log_time = self.mouse_press_time
            self.drag_move_count = 0

            _logger.info(f"[MOUSE_PRESS] {btn_name} 버튼 클릭 | 위치: ({pos.x()}, {pos.y()}) | 위젯: {widget_desc}")

        # ---------------- 2. 마우스 릴리즈 (Release) ----------------
        elif etype == QEvent.MouseButtonRelease:
            btn_name = "Left" if event.button() == Qt.LeftButton else "Right" if event.button() == Qt.RightButton else "Middle" if event.button() == Qt.MiddleButton else str(event.button())
            widget_desc = self._get_widget_desc(obj)
            pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
            
            duration_ms = (time.perf_counter() - self.mouse_press_time) * 1000.0 if self.mouse_press_time else 0.0
            
            # 이동 거리가 있으면 드래그 완료 요약 로그
            if self.mouse_press_pos:
                dx = pos.x() - self.mouse_press_pos.x()
                dy = pos.y() - self.mouse_press_pos.y()
                dist = (dx**2 + dy**2)**0.5
                if dist > 5:
                    _logger.info(f"[MOUSE_DRAG_END] {btn_name} 드래그 완료 | 이동: dx={dx}, dy={dy} (거리: {dist:.1f}px) | 지속: {duration_ms:.0f}ms | 위젯: {widget_desc}")
                else:
                    _logger.info(f"[MOUSE_RELEASE] {btn_name} 버튼 뗌 (단순 클릭) | 지속: {duration_ms:.0f}ms | 위젯: {widget_desc}")
            else:
                _logger.info(f"[MOUSE_RELEASE] {btn_name} 버튼 뗌 | 위젯: {widget_desc}")

            self.mouse_press_pos = None
            self.mouse_press_button = None

        # ---------------- 3. 마우스 드래그 이동 (Move with buttons) ----------------
        elif etype == QEvent.MouseMove and self.mouse_press_button:
            self.drag_move_count += 1
            now = time.perf_counter()
            # 1초에 수백 번 쏟아지는 마우스 이동 중, 0.4초 간격으로 드래그 상태 주기적 요약 기록
            if now - self.last_drag_log_time >= 0.4:
                widget_desc = self._get_widget_desc(obj)
                pos = event.position().toPoint() if hasattr(event, "position") else event.pos()
                if self.mouse_press_pos:
                    dx = pos.x() - self.mouse_press_pos.x()
                    dy = pos.y() - self.mouse_press_pos.y()
                    _logger.debug(f"[MOUSE_DRAGGING] {self.mouse_press_button} 드래그 조작 중... | 델타: ({dx}, {dy}) | 누적 이동: {self.drag_move_count}회 | 위젯: {widget_desc}")
                self.last_drag_log_time = now

        # ---------------- 4. 마우스 휠 (Wheel) ----------------
        elif etype == QEvent.Wheel:
            widget_desc = self._get_widget_desc(obj)
            delta = event.angleDelta().y()
            direction = "Zoom In (확대)" if delta > 0 else "Zoom Out (축소)"
            _logger.info(f"[MOUSE_WHEEL] {direction} (delta={delta}) | 위젯: {widget_desc}")

        # ---------------- 5. 키보드 입력 (Key Press) ----------------
        elif etype == QEvent.KeyPress:
            key = event.key()
            modifiers = []
            if event.modifiers() & Qt.ControlModifier:
                modifiers.append("Ctrl")
            if event.modifiers() & Qt.ShiftModifier:
                modifiers.append("Shift")
            if event.modifiers() & Qt.AltModifier:
                modifiers.append("Alt")

            key_name = ""
            text = event.text()
            if text and text.isprintable():
                key_name = f"'{text}'"
            else:
                key_map = {
                    Qt.Key_Escape: "ESC", Qt.Key_Return: "Enter", Qt.Key_Enter: "Enter",
                    Qt.Key_Space: "Space", Qt.Key_Backspace: "Backspace", Qt.Key_Delete: "Delete",
                    Qt.Key_Tab: "Tab", Qt.Key_Left: "Left", Qt.Key_Right: "Right",
                    Qt.Key_Up: "Up", Qt.Key_Down: "Down",
                    Qt.Key_F1: "F1", Qt.Key_F2: "F2", Qt.Key_F3: "F3", Qt.Key_F4: "F4",
                    Qt.Key_F5: "F5", Qt.Key_F6: "F6", Qt.Key_F7: "F7", Qt.Key_F8: "F8",
                    Qt.Key_F9: "F9", Qt.Key_F10: "F10", Qt.Key_F11: "F11", Qt.Key_F12: "F12",
                }
                key_name = key_map.get(key, f"Key_{key}")

            combo = "+".join(modifiers + [key_name]) if modifiers else key_name
            widget_desc = self._get_widget_desc(obj)
            _logger.info(f"[KEY_PRESS] 키보드 입력: {combo} | 위젯: {widget_desc}")

        return super().eventFilter(obj, event)

_global_filter = None

def install_activity_logger(app: QApplication):
    """QApplication에 전역 사용자 활동 로거 설치"""
    global _global_filter
    if _global_filter is None:
        _global_filter = GlobalActivityFilter()
        app.installEventFilter(_global_filter)
        _logger.info(">>> 전역 마우스/키보드 실시간 로깅 시스템이 활성화되었습니다. (로그 파일: logs/activity.log)")
    return _global_filter
