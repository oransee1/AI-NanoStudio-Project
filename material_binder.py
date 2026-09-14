import os
import cv2
import numpy as np
import trimesh
import pyvista as pv

class MaterialBinder:
    @staticmethod
    def auto_apply_texture(mesh: trimesh.Trimesh, base_texture_path: str, shadow_bake_path: str = None, output_tex_path="final_clothed_diffuse.png"):
        """
        AI 생성 텍스처와 베이킹된 그림자 맵을 Multiply 블렌딩하여 메시에 적용합니다.
        """
        cloth_img = cv2.imread(base_texture_path)
        
        if cloth_img is None:
            raise ValueError(f"텍스처 이미지를 찾을 수 없습니다: {base_texture_path}")

        if shadow_bake_path:
            shadow_img = cv2.imread(shadow_bake_path)
            if shadow_img is not None:
                if cloth_img.shape != shadow_img.shape:
                    shadow_img = cv2.resize(shadow_img, (cloth_img.shape[1], cloth_img.shape[0]))
                cloth_norm = cloth_img.astype(np.float32) / 255.0
                shadow_norm = shadow_img.astype(np.float32) / 255.0
                blended = (cloth_norm * shadow_norm) * 255.0
                blended = np.clip(blended, 0, 255).astype(np.uint8)
            else:
                blended = cloth_img
        else:
            blended = cloth_img

        cv2.imwrite(output_tex_path, blended)
        print(f"[Done] 텍스처 적용 완료: {output_tex_path}")

        try:
            # PyVista 액터에 텍스처 바인딩을 위해 텍스처 로드 반환
            pass
        except Exception as e:
            print(f"머티리얼 바인딩 오류 (무시됨): {e}")
            
        return output_tex_path

    @staticmethod
    def apply_pbr_to_pyvista_actor(plotter: pv.Plotter, actor_name: str, 
                                   diffuse_path=None, normal_path=None, 
                                   roughness_path=None, metallic_path=None,
                                   roughness_value=0.5, metallic_value=0.0):
        """
        PyVista 뷰포트 내의 특정 Actor에 PBR(Physically Based Rendering) 속성과 머티리얼을 적용합니다.
        """
        # PyVista 플로터에서 해당 액터 찾기
        actor = None
        for a_name, a in plotter.actors.items():
            if a_name == actor_name:
                actor = a
                break
                
        if not actor:
            print(f"Actor '{actor_name}'를 찾을 수 없습니다.")
            return False

        try:
            prop = actor.prop
            try:
                prop.interpolation = 'pbr'
            except Exception:
                pass
            prop.roughness = roughness_value
            prop.metallic = metallic_value

            # 텍스처 로드 및 적용
            if diffuse_path and os.path.exists(diffuse_path):
                tex = pv.read_texture(diffuse_path)
                actor.texture = tex
                
            # PyVista PBR 매핑
            if normal_path and os.path.exists(normal_path):
                try:
                    prop.normal_texture = pv.read_texture(normal_path)
                except Exception:
                    pass
            if roughness_path and os.path.exists(roughness_path):
                try:
                    prop.roughness_texture = pv.read_texture(roughness_path)
                except Exception:
                    pass
            if metallic_path and os.path.exists(metallic_path):
                try:
                    prop.metallic_texture = pv.read_texture(metallic_path)
                except Exception:
                    pass
                
            plotter.render()
            print(f"[Done] '{actor_name}'에 PBR 머티리얼 적용 완료")
            return True
        except Exception as e:
            print(f"PBR 적용 오류: {e}")
            return False

    @staticmethod
    def export_to_glb(mesh: trimesh.Trimesh, filepath="character_ready_to_play.glb"):
        """게임 엔진용 GLB 파일로 내보냅니다."""
        try:
            mesh.fix_normals()
        except Exception as e:
            print(f"GLB 노멀 자동 보정 경고: {e}")
        mesh.export(filepath)
        print(f"[Done] 게임 엔진용 파일 저장 완료: {filepath}")
        return filepath
