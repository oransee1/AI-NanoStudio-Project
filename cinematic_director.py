"""
cinematic_director.py — NanoStudio-3D 시네마틱 디렉터 엔진
==========================================================
FR-08: 카메라 키프레임 A/B 레코더 + 3D 이동 벡터 산출
FR-09: Gemini 2.5 Flash 기반 Cinematic Director Agent (Veo 모션 프롬프트 생성)

필수 기술 요소 "AI Agent" 충족 핵심 모듈
"""

import os
import math
import numpy as np
from datetime import datetime

from PySide6.QtCore import QThread, Signal

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None


class CinematicDirector:
    """
    3D 뷰포트의 시작점(A)과 끝점(B) 카메라 좌표를 관리하고,
    3차원 이동 벡터(Δx, Δy, Δz, Yaw, Pitch)를 산출하는 키프레임 레코더.
    """

    def __init__(self):
        self.keyframe_a = None  # 시작점 카메라 상태
        self.keyframe_b = None  # 종료점 카메라 상태

    def set_keyframe(self, slot: str, plotter):
        """
        현재 PyVista 뷰포트의 카메라 상태를 키프레임으로 저장합니다.

        :param slot: 'A' 또는 'B' (시작점/종료점)
        :param plotter: PyVista QtInteractor 인스턴스
        :return: 저장된 카메라 상태 딕셔너리
        """
        cam = plotter.camera
        pose = {
            "position": tuple(cam.position),
            "focal_point": tuple(cam.focal_point),
            "view_up": tuple(cam.up) if hasattr(cam, 'up') else tuple(cam.viewup),
            "view_angle": cam.view_angle,
            "clipping_range": tuple(cam.clipping_range),
        }

        if slot.upper() == 'A':
            self.keyframe_a = pose
        elif slot.upper() == 'B':
            self.keyframe_b = pose
        else:
            raise ValueError(f"슬롯은 'A' 또는 'B'만 가능합니다: {slot}")

        return pose

    def has_both_keyframes(self) -> bool:
        """시작점(A)과 종료점(B) 모두 등록되었는지 확인"""
        return self.keyframe_a is not None and self.keyframe_b is not None

    def get_registration_status(self) -> str:
        """현재 키프레임 등록 상태를 문자열로 반환"""
        a_status = "✅" if self.keyframe_a else "⬜"
        b_status = "✅" if self.keyframe_b else "⬜"
        return f"시작점(A): {a_status}  |  종료점(B): {b_status}"

    def compute_motion_vector(self) -> dict:
        """
        시작점(A)과 끝점(B) 카메라 좌표로부터 3차원 이동 벡터를 계산합니다.

        Returns:
            dict: {
                "dx": float, "dy": float, "dz": float,
                "distance": float,
                "horizontal_dist": float,
                "altitude_change": float,
                "yaw_deg": float,
                "pitch_deg": float,
            }
        """
        if not self.has_both_keyframes():
            raise ValueError("시작점(A)과 종료점(B)이 모두 등록되어야 합니다.")

        pos_a = np.array(self.keyframe_a["position"])
        pos_b = np.array(self.keyframe_b["position"])
        focal_a = np.array(self.keyframe_a["focal_point"])
        focal_b = np.array(self.keyframe_b["focal_point"])

        # 위치 변화량
        delta = pos_b - pos_a
        dx, dy, dz = delta[0], delta[1], delta[2]

        # 총 이동 거리
        distance = float(np.linalg.norm(delta))

        # 수평 이동 거리 (XY 평면)
        horizontal_dist = float(np.sqrt(dx ** 2 + dy ** 2))

        # 고도 변화량
        altitude_change = float(dz)

        # 시선 방향 벡터 (카메라 → 주시점)
        gaze_a = focal_a - pos_a
        gaze_b = focal_b - pos_b

        # Yaw 변화 (수평 회전각) — XY 평면 기준
        yaw_a = math.atan2(gaze_a[0], gaze_a[1])
        yaw_b = math.atan2(gaze_b[0], gaze_b[1])
        yaw_diff = math.degrees(yaw_b - yaw_a)
        while yaw_diff > 180:
            yaw_diff -= 360
        while yaw_diff < -180:
            yaw_diff += 360

        # Pitch 변화 (수직 기울기각)
        gaze_a_len_h = np.sqrt(gaze_a[0] ** 2 + gaze_a[1] ** 2)
        gaze_b_len_h = np.sqrt(gaze_b[0] ** 2 + gaze_b[1] ** 2)
        pitch_a = math.atan2(gaze_a[2], gaze_a_len_h) if gaze_a_len_h > 1e-6 else 0
        pitch_b = math.atan2(gaze_b[2], gaze_b_len_h) if gaze_b_len_h > 1e-6 else 0
        pitch_diff = math.degrees(pitch_b - pitch_a)

        return {
            "dx": float(dx),
            "dy": float(dy),
            "dz": float(dz),
            "distance": distance,
            "horizontal_dist": horizontal_dist,
            "altitude_change": altitude_change,
            "yaw_deg": round(yaw_diff, 2),
            "pitch_deg": round(pitch_diff, 2),
        }

    def capture_frames(self, plotter, output_dir: str) -> tuple:
        """
        시작점(A)과 종료점(B) 카메라 앵글에서 고해상도 스크린샷을 캡처합니다.

        :param plotter: PyVista QtInteractor
        :param output_dir: 저장 디렉토리 경로
        :return: (start_path, end_path) 튜플
        """
        if not self.has_both_keyframes():
            raise ValueError("시작점(A)과 종료점(B)이 모두 등록되어야 합니다.")

        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%H%M%S")

        # 현재 카메라 상태 백업
        cam = plotter.camera
        backup = {
            "position": tuple(cam.position),
            "focal_point": tuple(cam.focal_point),
            "view_up": tuple(cam.up) if hasattr(cam, 'up') else tuple(cam.viewup),
        }

        start_path = os.path.join(output_dir, f"shot_start_{timestamp}.png")
        end_path = os.path.join(output_dir, f"shot_end_{timestamp}.png")

        # 시작점(A) 캡처
        self._apply_camera(plotter, self.keyframe_a)
        plotter.screenshot(start_path, transparent_background=False, window_size=[1920, 1080])

        # 종료점(B) 캡처
        self._apply_camera(plotter, self.keyframe_b)
        plotter.screenshot(end_path, transparent_background=False, window_size=[1920, 1080])

        # 원래 카메라 복원
        cam.position = backup["position"]
        cam.focal_point = backup["focal_point"]
        if hasattr(cam, 'up'):
            cam.up = backup["view_up"]
        else:
            cam.viewup = backup["view_up"]
        plotter.render()

        print(f"[Done] 시네마틱 프레임 캡처 완료: {start_path}, {end_path}")
        return start_path, end_path

    def get_summary_text(self) -> str:
        """카메라 이동 요약을 한글 문자열로 반환합니다."""
        if not self.has_both_keyframes():
            return "카메라 시작점과 종료점이 모두 등록되지 않았습니다."

        try:
            mv = self.compute_motion_vector()
        except Exception as e:
            return f"벡터 계산 오류: {e}"

        parts = []
        if mv["horizontal_dist"] > 0.5:
            if abs(mv["yaw_deg"]) > 30:
                direction = "반시계방향" if mv["yaw_deg"] > 0 else "시계방향"
                parts.append(f"{direction} {abs(mv['yaw_deg']):.0f}° 궤도 회전")
            else:
                parts.append(f"수평 직선 이동 {mv['horizontal_dist']:.1f}m")

        if abs(mv["altitude_change"]) > 0.3:
            if mv["altitude_change"] > 0:
                parts.append(f"고도 {mv['altitude_change']:.1f}m 상승 (크레인 업)")
            else:
                parts.append(f"고도 {abs(mv['altitude_change']):.1f}m 하강")

        if abs(mv["pitch_deg"]) > 5:
            if mv["pitch_deg"] > 0:
                parts.append(f"틸트 업 {mv['pitch_deg']:.0f}°")
            else:
                parts.append(f"틸트 다운 {abs(mv['pitch_deg']):.0f}°")

        if not parts:
            parts.append(f"미세 이동 (총 거리: {mv['distance']:.2f}m)")

        summary = f"카메라 동선: {', '.join(parts)} | 총 이동거리: {mv['distance']:.1f}m"
        return summary

    def get_camera_data_for_pdf(self) -> dict:
        """ContiPDFBuilder에 전달할 카메라 데이터 딕셔너리를 반환합니다."""
        mv = self.compute_motion_vector()
        return {
            "start": self.keyframe_a,
            "end": self.keyframe_b,
            "delta": mv,
        }

    @staticmethod
    def _apply_camera(plotter, pose: dict):
        """카메라 상태를 뷰포트에 적용합니다."""
        cam = plotter.camera
        cam.position = pose["position"]
        cam.focal_point = pose["focal_point"]
        if hasattr(cam, 'up'):
            cam.up = pose.get("view_up", (0, 1, 0))
        else:
            cam.viewup = pose.get("view_up", (0, 1, 0))
        if "view_angle" in pose:
            cam.view_angle = pose["view_angle"]
        if "clipping_range" in pose:
            cam.clipping_range = pose["clipping_range"]
        plotter.render()

    def reset(self):
        """키프레임 데이터를 초기화합니다."""
        self.keyframe_a = None
        self.keyframe_b = None


class VeoPromptWorker(QThread):
    """
    FR-09: Gemini 2.5 Flash 기반 Cinematic Director Agent.
    3D 카메라 궤적 벡터와 씬 콘셉트를 해석하여
    Google Flow(Veo) 전용 영문 카메라 모션 프롬프트를 자동 생성합니다.
    """
    prompt_ready = Signal(str)   # 성공 시 프롬프트 텍스트 반환
    error = Signal(str)          # 에러 메시지 반환

    DIRECTOR_SYSTEM_PROMPT = """You are a professional cinematic camera director and Google Flow/Veo video prompt specialist.

Your task is to analyze 3D camera trajectory data (start/end positions, delta vectors, yaw/pitch changes) 
and a scene description, then compose a single, precise English motion prompt optimized for Google Veo video generation.

Rules:
1. Use professional cinematography terminology: Dolly in/out, Truck left/right, Pedestal up/down, 
   Pan left/right, Tilt up/down, Orbital/Arc shot, Crane shot, Push-in, Pull-back, Steadicam, Tracking shot.
2. Include lens specifications when appropriate (e.g., "24mm wide-angle lens", "85mm telephoto").
3. Specify motion quality: "smooth", "seamless", "slow", "gradual", "cinematic".
4. Describe lighting consistency to prevent flickering between First and Last frames.
5. Keep the prompt to 1-3 sentences maximum.
6. The prompt must connect the First Frame and Last Frame naturally.
7. Return ONLY the final prompt text — no explanations, no headers, no markdown."""

    def __init__(self, start_cam: dict, end_cam: dict, motion_vector: dict,
                 scene_desc: str, api_key: str = None):
        super().__init__()
        self.start_cam = start_cam
        self.end_cam = end_cam
        self.motion_vector = motion_vector
        self.scene_desc = scene_desc
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")

    def run(self):
        if not genai:
            self.error.emit("google-genai 모듈이 설치되어 있지 않습니다.")
            return

        if not self.api_key:
            self.error.emit("GEMINI_API_KEY가 설정되어 있지 않습니다.")
            return

        try:
            client = genai.Client(api_key=self.api_key)

            mv = self.motion_vector
            pos_a = self.start_cam.get("position", (0, 0, 0))
            pos_b = self.end_cam.get("position", (0, 0, 0))
            focal_a = self.start_cam.get("focal_point", (0, 0, 0))
            focal_b = self.end_cam.get("focal_point", (0, 0, 0))

            user_query = f"""Scene/Architecture Description: {self.scene_desc}

Camera Start Position (A): {pos_a}
Camera Start Focal Point: {focal_a}
Camera End Position (B): {pos_b}
Camera End Focal Point: {focal_b}

Motion Delta Vector:
- dx: {mv.get('dx', 0):.2f}m, dy: {mv.get('dy', 0):.2f}m, dz: {mv.get('dz', 0):.2f}m
- Total Travel Distance: {mv.get('distance', 0):.2f}m
- Horizontal Distance: {mv.get('horizontal_dist', 0):.2f}m
- Altitude Change: {mv.get('altitude_change', 0):.2f}m
- Yaw Rotation: {mv.get('yaw_deg', 0):.1f} degrees
- Pitch Change: {mv.get('pitch_deg', 0):.1f} degrees

Compose a single cinematic camera motion prompt for Google Veo that will smoothly connect the First Frame (shot from position A) to the Last Frame (shot from position B)."""

            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=[user_query],
                config=types.GenerateContentConfig(
                    system_instruction=self.DIRECTOR_SYSTEM_PROMPT,
                    temperature=0.7,
                    max_output_tokens=300,
                )
            )

            prompt_text = response.text.strip()
            if not prompt_text:
                self.error.emit("Gemini가 빈 응답을 반환했습니다.")
                return

            # 마크다운 서식 제거
            if prompt_text.startswith("```"):
                lines = prompt_text.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].startswith("```"):
                    lines = lines[:-1]
                prompt_text = "\n".join(lines).strip()

            prompt_text = prompt_text.strip('"').strip("'")
            self.prompt_ready.emit(prompt_text)

        except Exception as e:
            self.error.emit(f"Cinematic Director Agent 오류: {str(e)}")


# ─────────────────────────────────────────────────────────────
#  Standalone Test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  NanoStudio-3D Cinematic Director — 단독 테스트")
    print("=" * 60)

    director = CinematicDirector()

    director.keyframe_a = {
        "position": (10.0, 5.0, 3.0),
        "focal_point": (0.0, 0.0, 1.0),
        "view_up": (0.0, 0.0, 1.0),
        "view_angle": 30.0,
        "clipping_range": (0.01, 1000.0),
    }
    director.keyframe_b = {
        "position": (5.0, 8.0, 6.0),
        "focal_point": (0.0, 0.0, 1.5),
        "view_up": (0.0, 0.0, 1.0),
        "view_angle": 30.0,
        "clipping_range": (0.01, 1000.0),
    }

    vector = director.compute_motion_vector()
    print(f"\n[이동 벡터 계산 결과]")
    for key, val in vector.items():
        print(f"  {key}: {val}")

    summary = director.get_summary_text()
    print(f"\n[한글 요약] {summary}")

    pdf_data = director.get_camera_data_for_pdf()
    print(f"\n[PDF 데이터] start/end/delta 키 존재: "
          f"{'start' in pdf_data and 'end' in pdf_data and 'delta' in pdf_data}")

    status = director.get_registration_status()
    print(f"\n[등록 상태] {status}")

    print("\n✅ 모든 단독 테스트 통과!")
