import pyvista as pv
import numpy as np
import cv2
import os

class CameraManager:
    def __init__(self, plotter: pv.Plotter):
        self.plotter = plotter
        self.current_camera_data = {}

    def capture_viewport(self, output_path="scene_capture.png"):
        """현재 뷰포트의 상태를 고해상도 이미지로 캡처하여 저장합니다."""
        self.plotter.screenshot(output_path, transparent_background=True)
        print(f"[Done] 뷰포트 캡처 완료: {output_path}")
        return output_path

    def save_camera_state(self):
        """현재 카메라의 위치, 초점, 방향 정보를 저장합니다."""
        cam = self.plotter.camera
        self.current_camera_data = {
            "position": list(cam.position),
            "focal_point": list(cam.focal_point),
            "viewup": list(cam.viewup),
            "view_angle": cam.view_angle,
            "clipping_range": list(cam.clipping_range)
        }
        return self.current_camera_data

    def load_camera_state(self, state_dict):
        """저장된 카메라 상태를 뷰포트에 복원합니다."""
        if not state_dict:
            return
        cam = self.plotter.camera
        cam.position = state_dict.get("position", cam.position)
        cam.focal_point = state_dict.get("focal_point", cam.focal_point)
        cam.viewup = state_dict.get("viewup", cam.viewup)
        cam.view_angle = state_dict.get("view_angle", cam.view_angle)
        cam.clipping_range = state_dict.get("clipping_range", cam.clipping_range)
        self.plotter.render()
