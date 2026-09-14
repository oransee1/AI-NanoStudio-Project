import os
import sys
from dotenv import load_dotenv
import pyvista as pv
import trimesh
import cv2

# 로컬 모듈 임포트
from skp_loader import load_skp_to_pyvista
from uv_unwrapper import UVUnwrapper
from shadow_baker import ShadowBaker
from material_binder import MaterialBinder

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None

def main():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY가 없습니다.")
        return

    # 1. 파일 로드
    skp_file = r"C:\Users\OWNER\PycharmProjects\AI-NanoStudio-Project\3d_model\car.skp"
    if not os.path.exists(skp_file):
        print(f"파일이 없습니다: {skp_file}")
        return
        
    print("1. 스케치업 모델 로드 시작...")
    actors = load_skp_to_pyvista(skp_file)
    if not actors:
        print("스케치업 모델 로드 실패")
        return
    print(f"로드 성공: {list(actors.keys())}")
    
    # 2. 뷰포트 오프스크린 캡처 (건축 렌더링용)
    print("2. 조감도 뷰포트 캡처...")
    plotter = pv.Plotter(off_screen=True)
    for name, mesh in actors.items():
        plotter.add_mesh(mesh, color='white')
    plotter.view_isometric()
    capture_path = "arch_capture.png"
    plotter.screenshot(capture_path)
    print(f"캡처 완료: {capture_path}")
    
    # 제미나이 클라이언트
    client = genai.Client(api_key=api_key)
    
    # 3. 건축 렌더링 생성
    print("3. 건축/인테리어용 실사 렌더링 AI 생성 중...")
    try:
        # Img2Img가 아니라 Text2Image로 작동하도록 셋팅
        prompt_arch = "Photorealistic architectural rendering of a car, highly detailed, beautiful lighting, automotive photography style, 8k"
        result_arch = client.models.generate_images(
            model='imagen-3.0-generate-001',
            prompt=prompt_arch,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                output_mime_type="image/png",
                aspect_ratio="16:9",
                person_generation="DONT_ALLOW"
            )
        )
        
        if result_arch.generated_images:
            arch_output_path = "final_architectural_render.png"
            with open(arch_output_path, "wb") as f:
                f.write(result_arch.generated_images[0].image.image_bytes)
            print(f"[Done] 건축 렌더링 결과물 생성 완료: {arch_output_path}")
        else:
            print("건축 렌더링 이미지 생성 실패")
    except Exception as e:
        print(f"건축 렌더링 에러: {e}")
        
    # 4. UV 언랩 및 패킹
    print("4. 게임용 에셋 UV 언랩 및 아틀라스 패킹...")
    unwrapper = UVUnwrapper()
    atlas, uv_img_path = unwrapper.unwrap_and_pack(actors)
    print(f"[Done] 언랩 완료: {uv_img_path}")
    
    # 5. 그림자 라이트맵 베이킹
    print("5. 그림자 및 라이트맵 베이킹...")
    baker = ShadowBaker(actors)
    shadow_map_path = baker.bake_ambient_and_sunlight(atlas)
    print(f"[Done] 베이킹 완료: {shadow_map_path}")
    
    # 6. 게임 에셋 텍스처 생성
    print("6. 게임 에셋 텍스처 AI 생성 중...")
    try:
        prompt_game = "Hand-painted stylized RPG game texture for a car, vibrant colors, clean UV map, flat diffuse map"
        result_game = client.models.generate_images(
            model='imagen-3.0-generate-001',
            prompt=prompt_game,
            config=types.GenerateImagesConfig(
                number_of_images=1,
                output_mime_type="image/png",
                aspect_ratio="1:1",
                person_generation="DONT_ALLOW"
            )
        )
        
        if result_game.generated_images:
            ai_texture_path = "ai_generated_texture.png"
            with open(ai_texture_path, "wb") as f:
                f.write(result_game.generated_images[0].image.image_bytes)
            print(f"[Done] 게임 텍스처 생성 완료: {ai_texture_path}")
        else:
            print("게임 텍스처 이미지 생성 실패 (가짜 텍스처로 대체)")
            ai_texture_path = shadow_map_path
    except Exception as e:
        print(f"게임 애셋 파이프라인 에러: {e} (가짜 텍스처로 진행)")
        ai_texture_path = shadow_map_path

    # 7. 텍스처 바인딩 및 GLB 내보내기 (모든 부품 통합)
    print("7. 전체 부품 텍스처 바인딩 및 GLB 내보내기...")
    
    # 텍스처는 이미 만들어졌으므로 Material 생성
    # (원래 auto_apply_texture가 첫번째 메시에 적용하므로, 여기서는 이미지 합성 로직만 차용)
    final_diffuse = MaterialBinder.auto_apply_texture(actors[list(actors.keys())[0]], ai_texture_path, shadow_map_path)
    print(f"[Done] 최종 디퓨즈 합성 완료: {final_diffuse}")
    
    material = trimesh.visual.texture.SimpleMaterial(
        image=cv2.cvtColor(cv2.imread(final_diffuse), cv2.COLOR_BGR2RGB)
    )
    
    # 각 부품에 해당하는 UV를 맵핑하고 씬에 추가
    combined_tri_meshes = []
    atlas_list = list(atlas)
    
    for i, (name, pv_mesh) in enumerate(actors.items()):
        faces = pv_mesh.faces.reshape(-1, 4)[:, 1:]
        tri_mesh = trimesh.Trimesh(vertices=pv_mesh.points, faces=faces)
        tri_mesh.fix_normals()
        
        # xatlas에서 생성된 해당 파트의 uv 가져오기
        if i < len(atlas_list):
            _, _, uvs = atlas_list[i]
            tri_mesh.visual = trimesh.visual.TextureVisuals(uv=uvs, material=material)
            combined_tri_meshes.append(tri_mesh)
            
    # 전체 씬으로 묶기
    final_scene = trimesh.Scene(combined_tri_meshes)
    export_path = "final_game_asset_car.glb"
    final_scene.export(export_path)
    print(f"[Done] 최종 게임 애셋 내보내기 완료: {export_path}")

if __name__ == "__main__":
    main()
