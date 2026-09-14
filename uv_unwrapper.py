import numpy as np
import trimesh
import cv2

try:
    import xatlas
except ImportError:
    xatlas = None
    print("Warning: xatlas module is not installed. UV unwrapping will not work.")

class UVUnwrapper:
    def __init__(self):
        self.atlas = None

    def unwrap_and_pack(self, parts_dict, img_size=2048, part_colors=None):
        """다수의 파트(부품)들을 언랩하고 하나의 아틀라스로 패킹합니다."""
        if xatlas is None:
            raise ImportError("xatlas module is required for UV unwrapping.")

        self.atlas = xatlas.Atlas()
        part_names = list(parts_dict.keys())
        
        for name in part_names:
            pv_mesh = parts_dict[name]
            # pv.PolyData를 trimesh.Trimesh로 변환 (PyVista 면 포맷: [3, v1,v2,v3, 3, v4,v5,v6])
            faces = pv_mesh.faces.reshape(-1, 4)[:, 1:]
            mesh = trimesh.Trimesh(vertices=pv_mesh.points, faces=faces)
            mesh.fix_normals()
            self.atlas.add_mesh(mesh.vertices, mesh.faces)
            
        print("[xatlas] 병합 패킹 계산 중...")
        self.atlas.generate()
        
        output_image = "character_atlas_uv_map.png"
        self._export_uv_wireframe(part_names, img_size, output_image, part_colors)
        
        # 반환: 패킹된 메시 데이터, 이미지 경로
        return self.atlas, output_image

    def _export_uv_wireframe(self, part_names, img_size, output_filename, part_colors=None):
        canvas = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        
        fallback_colors = [
            (0, 255, 255),  # Yellow
            (0, 200, 0),    # Green
            (255, 100, 0),  # Blue
            (255, 0, 200),  # Magenta
            (0, 0, 255)     # Red
        ]
        
        for i, (vmapping, indices, uvs) in enumerate(self.atlas):
            # part_colors가 제공되었으면 해당 부품의 색상 사용, 없으면 폴백
            if part_colors and i < len(part_names) and part_names[i] in part_colors:
                # part_color는 [R, G, B] 정규화 (0~1)
                r, g, b = part_colors[part_names[i]]
                # OpenCV는 BGR을 사용
                color = (int(b * 255), int(g * 255), int(r * 255))
            else:
                color = fallback_colors[i % len(fallback_colors)]
            
            pixel_uvs = (uvs * [img_size - 1, img_size - 1]).astype(np.int32)
            pixel_uvs[:, 1] = (img_size - 1) - pixel_uvs[:, 1]
            
            for face in indices:
                pts = pixel_uvs[face]
                cv2.polylines(canvas, [pts], isClosed=True, color=color, thickness=1)
                
        cv2.imwrite(output_filename, canvas)
        print(f"[Done] 2D 전개도 이미지 저장 완료: {output_filename}")
