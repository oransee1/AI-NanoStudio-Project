import os
from datetime import datetime
import numpy as np
import pyvista as pv
try:
    from openskp import SkpFile
except ImportError:
    pass

def load_skp_to_pyvista(skp_path):
    """
    openskp를 이용해 .skp 파일을 PyVista PolyData 액터 목록으로 변환합니다.
    """
    try:
        try:
            from openskp import legacy
            if hasattr(legacy, '_read_instance'):
                def safe_read_instance(ar, r):
                    cls = ar.current_class
                    pre = legacy._preamble(ar, r)
                    db = legacy._drawbase(ar, r)
                    ds, dn, _ = ar.read_object(r, expect='CComponentDefinition')
                    if dn != 'CComponentDefinition':
                        raise legacy.LegacyParseError(f'instance definition ref is {dn} {r.ctx()}')
                    xf = r.f64s(13)
                    name = r.utf16()
                    schema = ar.class_schema.get(cls)
                    min_schema = 1 if cls == 'CGroup' else 5
                    if ar.ver >= 14 and (schema is None or schema >= min_schema):
                        guid = r.raw(16)
                    else:
                        guid = b''
                    return {'k': 'instance', 'db': db, 'def': ds, 'xf': xf,
                            'name': name, 'guid': guid.hex().upper(), 'attrs': pre['attrs']}
                legacy._read_instance = safe_read_instance
                legacy._READERS['CComponentInstance'] = safe_read_instance
                legacy._READERS['CGroup'] = safe_read_instance
        except Exception:
            pass

        model = SkpFile.open(skp_path).parse()
        print(f"SketchUp 버전: {model.version}")
        print(f"발견된 레이어: {[l.name for l in model.layers]}")
        
        scene = SkpFile.open(skp_path).build_scene()
        
        actors = {}
        
        # mesh_index에서 의미 있는 이름 추출 및 동일 부품 병합
        grouped_actors = {}
        
        for i, prim in enumerate(scene.glb_primitives):
            vertices = np.array(prim.positions, dtype=np.float32).reshape(-1, 3)
            indices = np.array(prim.indices, dtype=np.int32)
            faces = np.column_stack([np.full(len(indices)//3, 3), indices.reshape(-1, 3)]).ravel()
            pv_mesh = pv.PolyData(vertices, faces)
            
            # openskp에서 스케치업 파일 파싱 시 이미 미터(m) 또는 적절한 비율로 반환되므로 별도의 1000배 축소/확대가 필요하지 않습니다.
            # 스케치업 원본 1 unit = 1m 스케일을 그대로 유지합니다.
            
            # 스케치업 Z-up을 PyVista Y-up으로 보정
            pv_mesh.rotate_x(90, inplace=True)
            
            # 각진 모서리가 뭉개져(smudged) 보이는 현상을 방지하기 위해, 특정 각도(예: 45도) 이상의 날카로운 엣지는 면을 분리하여 하드 엣지로 렌더링
            pv_mesh = pv_mesh.compute_normals(split_vertices=True, feature_angle=45)
            
            def_name = f"Layer_{i}"
            if hasattr(scene, 'mesh_index') and scene.mesh_index:
                # key 찾기
                key = f"mesh_{i}"
                for k in scene.mesh_index.keys():
                    if k.startswith(key + "_"):
                        meta = scene.mesh_index[k]
                        if meta.definition_name:
                            def_name = meta.definition_name
                        break
            
            # 같은 컴포넌트(definition)끼리 병합
            if def_name in grouped_actors:
                grouped_actors[def_name] = grouped_actors[def_name].merge(pv_mesh)
            else:
                grouped_actors[def_name] = pv_mesh
                
        # 이름 유일성 보장 (만약 필요하다면)
        final_actors = {}
        for name, mesh in grouped_actors.items():
            final_actors[name] = mesh
            
        return final_actors
    except Exception as e:
        print(f"SKP 파싱 오류: {e}")
        return {}

def load_obj_to_pyvista(obj_path):
    """
    .obj 파일을 PyVista 뷰포트용 메시로 로드합니다 (안정적인 Fallback).
    """
    try:
        mesh = pv.read(obj_path)
        mesh.rotate_x(90, inplace=True) # 스케치업 Z-up 뷰를 Y-up 뷰로 90도 회전
        return {"Exported_Model": mesh}
    except Exception as e:
        print(f"OBJ 로드 오류: {e}")
        return {}

def subdivide_meshes(actors, n_subdivisions=1, subfilter='linear'):
    """
    로드된 PyVista PolyData 액터(메시)들의 면을 분할(Subdivision)합니다.
    
    :param actors: 딕셔너리 형태의 메시 데이터 (이름: pv.PolyData)
    :param n_subdivisions: 분할 단계 (높을수록 폴리곤 증가)
    :param subfilter: 'linear', 'butterfly', 'loop' 중 선택
    :return: 면이 분할된 새로운 액터 딕셔너리
    """
    subdivided_actors = {}
    for name, mesh in actors.items():
        try:
            # 면 분할 실행
            sub_mesh = mesh.subdivide(nsub=n_subdivisions, subfilter=subfilter)
            subdivided_actors[name] = sub_mesh
            print(f"[{name}] 면 분할 완료: 기존 {mesh.n_faces}면 -> {sub_mesh.n_faces}면")
        except Exception as e:
            print(f"[{name}] 면 분할 실패: {e}")
            subdivided_actors[name] = mesh
            
    return subdivided_actors

def export_meshes_to_obj(actors, output_path):
    """
    PyVista 메시를 .obj 파일로 내보냅니다.
    각 메시를 개별 오브젝트(o name)로 분리하여 계층구조(오브젝트 구분)를 유지합니다.
    """
    try:
        # 상위 디렉토리가 없으면 생성
        export_dir = os.path.dirname(output_path)
        if export_dir and not os.path.exists(export_dir):
            os.makedirs(export_dir)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("# Exported by AI-NanoStudio for Blender Import\n")
            
            # mtllib 선언 (필요 시 향후 mtl 파일 연결용)
            mtl_filename = os.path.basename(output_path).rsplit('.', 1)[0] + ".mtl"
            f.write(f"mtllib {mtl_filename}\n")
            
            vertex_offset = 1
            uv_offset = 1
            normal_offset = 1
            
            for name, mesh in actors.items():
                f.write(f"\no {name}\n")
                f.write(f"g {name}\n")        # Blender의 'Split By Group' 지원
                f.write(f"usemtl {name}_mat\n") # 재질 그룹(Material) 지원
                
                # 법선(Normal) 계산이 안되어 있다면 계산
                if not "Normals" in mesh.point_data:
                    mesh = mesh.compute_normals(split_vertices=True, feature_angle=45)
                
                points = mesh.points
                normals = mesh.point_normals if "Normals" in mesh.point_data else None
                uvs = mesh.active_texture_coordinates if mesh.active_texture_coordinates is not None else None
                
                # 1. 정점(v) 기록 (PyVista의 (x, -z, y)에서 원래 스케치업의 (x, y, z)로 복구하여 내보냅니다)
                for p in points:
                    f.write(f"v {p[0]} {p[2]} {-p[1]}\n")
                    
                # 2. 텍스처 좌표(vt) 기록
                has_uv = uvs is not None
                if has_uv:
                    for uv in uvs:
                        f.write(f"vt {uv[0]} {uv[1]}\n")
                        
                # 3. 법선(vn) 기록 (정점과 동일하게 역회전하여 복구)
                has_normal = normals is not None
                if has_normal:
                    for n in normals:
                        f.write(f"vn {n[0]} {n[2]} {-n[1]}\n")
                
                f.write("s 1\n") # 스무스 쉐이딩 켜기
                
                # 4. 면(f) 데이터 기록
                faces = mesh.faces
                i = 0
                while i < len(faces):
                    n_verts = faces[i]
                    i += 1
                    face_indices = faces[i : i + n_verts]
                    
                    f_str = ""
                    for idx in face_indices:
                        v_idx = idx + vertex_offset
                        vt_idx = idx + uv_offset if has_uv else ""
                        vn_idx = idx + normal_offset if has_normal else ""
                        
                        if has_uv and has_normal:
                            f_str += f" {v_idx}/{vt_idx}/{vn_idx}"
                        elif has_uv:
                            f_str += f" {v_idx}/{vt_idx}"
                        elif has_normal:
                            f_str += f" {v_idx}//{vn_idx}"
                        else:
                            f_str += f" {v_idx}"
                            
                    f.write(f"f{f_str}\n")
                    i += n_verts
                    
                # 오프셋 업데이트
                num_points = len(points)
                vertex_offset += num_points
                if has_uv:
                    uv_offset += num_points
                if has_normal:
                    normal_offset += num_points
                
        print(f"Blender 호환성이 향상된 OBJ 파일 내보내기 완료: {output_path}")
        return output_path
    except Exception as e:
        print(f"내보내기 오류: {e}")
        return None
