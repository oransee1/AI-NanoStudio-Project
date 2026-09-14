import numpy as np
import trimesh
import cv2

class ShadowBaker:
    def __init__(self, parts_dict):
        self.parts_dict = parts_dict

    def bake_ambient_and_sunlight(self, atlas, sun_pos=np.array([10.0, 15.0, 10.0]), img_size=2048):
        print("[Baking] 인테리어 씬 라이트맵 베이킹 시작...")
        
        # 합쳐진 전체 씬 메시 생성 (광선 충돌 검사용)
        combined_meshes = []
        for m in self.parts_dict.values():
            faces = m.faces.reshape(-1, 4)[:, 1:]
            tri_mesh = trimesh.Trimesh(vertices=m.points, faces=faces)
            tri_mesh.fix_normals()
            combined_meshes.append(tri_mesh)
            
        full_scene = trimesh.util.concatenate(combined_meshes)
        
        # embree (또는 내장) 기반 레이트레이싱 인터섹터 생성
        ray_intersector = full_scene.ray

        bake_canvas = np.ones((img_size, img_size), dtype=np.float32)

        for i, (vmapping, indices, uvs) in enumerate(atlas):
            original_mesh = combined_meshes[i]
            pixel_uvs = (uvs * [img_size - 1, img_size - 1]).astype(np.int32)
            pixel_uvs[:, 1] = (img_size - 1) - pixel_uvs[:, 1]

            for face in indices:
                pts_uv = pixel_uvs[face]
                orig_vert_indices = vmapping[face]
                pts_3d = original_mesh.vertices[orig_vert_indices]
                
                center_3d = np.mean(pts_3d, axis=0)
                edge1 = pts_3d[1] - pts_3d[0]
                edge2 = pts_3d[2] - pts_3d[0]
                face_normal = np.cross(edge1, edge2)
                norm = np.linalg.norm(face_normal)
                if norm == 0: 
                    continue
                face_normal /= norm

                light_dir = sun_pos - center_3d
                dist = np.linalg.norm(light_dir)
                light_dir /= dist

                dot = np.dot(face_normal, light_dir)
                if dot <= 0:
                    intensity = 0.3 # 기본 앰비언트 음영
                else:
                    origin = center_3d + face_normal * 0.01
                    hits = ray_intersector.intersects_any(
                        ray_origins=[origin],
                        ray_directions=[light_dir]
                    )
                    intensity = 0.35 if hits[0] else min(1.0, 0.4 + 0.6 * dot)

                triangle_contour = np.array([pts_uv], dtype=np.int32)
                cv2.fillPoly(bake_canvas, pts=triangle_contour, color=float(intensity))

        baked_lightmap = (bake_canvas * 255).astype(np.uint8)
        baked_lightmap = cv2.GaussianBlur(baked_lightmap, (5, 5), 0)
        
        output_filename = "baked_shadow_lightmap.png"
        cv2.imwrite(output_filename, baked_lightmap)
        print(f"[Done] 라이트맵 베이킹 완료: {output_filename}")
        
        return output_filename
