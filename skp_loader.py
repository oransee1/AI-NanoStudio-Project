import os
import time
from datetime import datetime
import numpy as np
import pyvista as pv
try:
    from openskp import SkpFile
except ImportError:
    pass

def load_skp_to_pyvista(skp_path, return_info=False, progress_callback=None):
    """
    openskp를 이용해 .skp 파일을 PyVista PolyData 액터 목록으로 변환합니다.
    return_info=True 지정 시 (final_actors, parse_info) 튜플을 반환합니다.
    """
    parse_info = {
        "skp_version": "Unknown",
        "layers": [],
        "total_parts": 0,
        "sample_parts": [],
        "tag_counts": {}
    }
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

        if progress_callback:
            progress_callback(10, "SKP 파일 구조 읽기 및 객체 파싱 중...")

        t_start = time.time()
        skp_obj = SkpFile.open(skp_path)
        model = skp_obj.parse()
        version_str = str(getattr(model, 'version', 'Unknown'))
        layers_list = [l.name for l in model.layers] if hasattr(model, 'layers') else []
        print(f"[TIMER] parse() 소요시간: {time.time() - t_start:.3f}초", flush=True)
        
        parse_info["skp_version"] = version_str
        parse_info["layers"] = layers_list

        if progress_callback:
            progress_callback(20, "3D 씬 데이터 구성 및 바인딩 중...")

        t_scene = time.time()
        scene = skp_obj.build_scene()
        print(f"[TIMER] build_scene() 소요시간: {time.time() - t_scene:.3f}초", flush=True)
        
        # 스케치업 모델의 Tags(레이어) 및 고유 RGB 색상 추출
        layer_colors = {}
        valid_layers = []
        if hasattr(model, 'layers') and model.layers:
            for l in model.layers:
                l_name = getattr(l, 'name', 'Untagged')
                r = getattr(l, 'color_r', 200) / 255.0
                g = getattr(l, 'color_g', 200) / 255.0
                b = getattr(l, 'color_b', 200) / 255.0
                layer_colors[l_name] = [r, g, b]
                if l_name and str(l_name).strip() not in ['Layer0', 'Untagged']:
                    valid_layers.append(str(l_name).strip())

        # 유효 레이어가 있을 경우 건축 Z축 높이 순서(기초부 -> 축부 -> 수장부 -> 지붕가구부 -> 지붕부)로 정렬
        def get_layer_z_order(name):
            n = str(name).strip()
            if '기초' in n or '기단' in n or '초석' in n:
                return 0
            elif '축' in n or '기둥' in n or '주' in n:
                return 1
            elif '수장' in n or '수정' in n or '창방' in n or '평방' in n:
                return 2
            elif '가구' in n or '도리' in n or '대들보' in n or '서래' in n:
                return 3
            elif '지붕' in n or '기와' in n or '막새' in n:
                return 4
            return 5

        if valid_layers:
            valid_layers = sorted(valid_layers, key=get_layer_z_order)

        # mesh_index O(1) 사전 색인 매핑 구축 (파싱 속도 초고속화)
        mesh_meta_map = {}
        if hasattr(scene, 'mesh_index') and scene.mesh_index:
            for k, meta in scene.mesh_index.items():
                parts = k.split('_')
                if len(parts) >= 2 and parts[0] == 'mesh' and parts[1].isdigit():
                    m_idx = int(parts[1])
                    if m_idx not in mesh_meta_map:
                        mesh_meta_map[m_idx] = meta

        # 1차 패스: 씬 내 모든 메쉬의 위치 및 Z축 범위 사전 수집
        prim_meta_list = []
        z_centers = []
        total_glb = len(scene.glb_primitives)
        for i, prim in enumerate(scene.glb_primitives):
            if progress_callback and total_glb > 0 and i % 100 == 0:
                pct = 20 + int((i / total_glb) * 5) # 20% ~ 25%
                progress_callback(pct, f"3D 씬 위치 분석 및 바운딩 박스 사전 수집 중... ({i}/{total_glb})")

            vertices = np.array(prim.positions, dtype=np.float32).reshape(-1, 3)
            indices = np.array(prim.indices, dtype=np.int32)
            meta = mesh_meta_map.get(i, None)
            if len(vertices) > 0 and len(indices) > 0:
                z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
                z_center = (z_min + z_max) / 2.0
                z_centers.append(z_center)
            else:
                z_center = 0.0
            prim_meta_list.append((vertices, indices, z_center, meta))

        overall_z_min = min(z_centers) if z_centers else 0.0
        overall_z_max = max(z_centers) if z_centers else 1.0
        z_span = overall_z_max - overall_z_min

        grouped_actors = {}
        tag_group_counters = {}
        
        t_loop = time.time()
        total_prims = len(prim_meta_list)
        for i, (vertices, indices, z_center, meta) in enumerate(prim_meta_list):
            if progress_callback and total_prims > 0 and i % 50 == 0:
                pct = 25 + int((i / total_prims) * 45) # 25% ~ 70%
                progress_callback(pct, f"3D 메쉬 생성 및 레이어/그룹 추출 중... ({i}/{total_prims})")

            if len(vertices) == 0 or len(indices) == 0:
                continue

            # 스케치업 앞/뒷면 구분 없이 모든 방향에서 완전한 앞면(양면, Double-Sided Frontface)으로 로딩
            n_v = len(vertices)
            double_vertices = np.vstack([vertices, vertices])
            triangles = indices.reshape(-1, 3)
            rev_triangles = np.column_stack([triangles[:, 0], triangles[:, 2], triangles[:, 1]]) + n_v
            double_triangles = np.vstack([triangles, rev_triangles])
            faces = np.column_stack([np.full(len(double_triangles), 3), double_triangles]).ravel()
            pv_mesh = pv.PolyData(double_vertices, faces)
            
            # 스케치업 Z-up을 PyVista Y-up으로 보정
            pv_mesh.rotate_x(90, inplace=True)
            try:
                if "Normals" not in pv_mesh.point_data or pv_mesh.point_normals is None:
                    pv_mesh.compute_normals(inplace=True)
            except Exception:
                pass
            
            tag_name = ""
            sub_name = ""
            if meta:
                l = getattr(meta, 'layer', None)
                p = getattr(meta, 'path', None)
                d = getattr(meta, 'definition_name', None)
                n = getattr(meta, 'name', None)
                
                l_str = str(l).strip() if l else ""
                p_str = str(p).strip() if p else ""
                d_str = str(d).strip() if d else ""
                n_str = str(n).strip() if n else ""
                
                # 1. 메쉬의 layer 속성이 valid_layers 중 하나인 경우
                if l_str and l_str in valid_layers:
                    tag_name = l_str
                
                # 자식 컴포넌트/그룹 서브이름 추출
                comp_part = ""
                if p_str and '/' in p_str:
                    path_parts = [pt.strip() for pt in p_str.split('/') if pt.strip()]
                    if len(path_parts) >= 2 and path_parts[0] == 'ROOT':
                        comp_part = path_parts[1]
                    elif len(path_parts) >= 1:
                        comp_part = path_parts[0]
                elif d_str:
                    comp_part = d_str
                    
                sub_part = n_str if n_str else (d_str if d_str else "")
                if comp_part and sub_part and comp_part != sub_part:
                    sub_name = f"{comp_part}_{sub_part}"
                elif comp_part:
                    sub_name = comp_part
                elif sub_part:
                    sub_name = sub_part

            # ROOT_MODEL_ROOT 등 최상위/추가 래퍼 메쉬는 모델에 필요한 요소이므로 '추가 부품' 그룹으로 분류
            is_extra_part = ("ROOT_MODEL" in sub_name or sub_name in ["ROOT", "MODEL", "ROOT_MODEL_ROOT", "MODEL_ROOT"])
            if is_extra_part:
                tag_name = "추가 부품"
            elif not tag_name or tag_name not in valid_layers:
                if valid_layers:
                    rel_z = (z_center - overall_z_min) / z_span if z_span > 0 else 0.0
                    layer_idx = min(int(rel_z * len(valid_layers)), len(valid_layers) - 1)
                    tag_name = valid_layers[layer_idx]
                else:
                    tag_name = "Untagged"

            if not sub_name:
                sub_name = "Group"
                
            # 스케치업 개별 그룹(Group) 단위로 독립성을 보존하기 위한 유일 키 생성
            counter_key = f"{tag_name}_{sub_name}"
            tag_group_counters[counter_key] = tag_group_counters.get(counter_key, 0) + 1
            idx = tag_group_counters[counter_key]
            
            full_key = f"{tag_name}/{sub_name}_{idx}"
            
            # 스케치업 Tag 고유 색상 및 태그명을 메시에 안전하게 보관 (VTK non-ASCII 에러 방지용 user_dict 적용)
            tag_color = layer_colors.get(tag_name, [0.7, 0.7, 0.7])
            if not hasattr(pv_mesh, 'user_dict') or pv_mesh.user_dict is None:
                pv_mesh.user_dict = {}
            pv_mesh.user_dict['tag_name'] = tag_name
            pv_mesh.user_dict['tag_color'] = tag_color
            
            grouped_actors[full_key] = pv_mesh
        print(f"[TIMER] glb_primitives 루프 소요시간: {time.time() - t_loop:.3f}초", flush=True)
                
        final_actors = {}
        for name, mesh in grouped_actors.items():
            final_actors[name] = mesh
            
        parse_info["total_parts"] = len(final_actors)
        parse_info["sample_parts"] = list(final_actors.keys())[:10]
        
        tag_counts = {}
        for k in final_actors.keys():
            t = k.split('/')[0] if '/' in k else 'Untagged'
            tag_counts[t] = tag_counts.get(t, 0) + 1
        parse_info["tag_counts"] = tag_counts

        if return_info:
            return final_actors, parse_info
        return final_actors
    except Exception as e:
        print(f"SKP 파싱 오류: {e}")
        if return_info:
            return {}, parse_info
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
                
                # 법선(Normal) 계산이 안되어 있다면 계산 (노멀 방향 자동 정렬 포함)
                if not "Normals" in mesh.point_data:
                    mesh = mesh.compute_normals(auto_orient_normals=True, split_vertices=True, feature_angle=45)
                
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
