import sys
import skp_loader

from openskp import SkpFile, legacy

try:
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

def print_skp_tree():
    model = SkpFile.open('3d_model/car.skp').build_scene()
    if hasattr(model, 'nodes'):
        for i, n in enumerate(model.nodes):
            name = n.name if hasattr(n, 'name') else f"Node_{i}"
            mesh_idx = getattr(n, 'mesh', None)
            print(f"Node {i}: {name}, mesh_idx={mesh_idx}")
    else:
        print("No nodes in scene.")

print_skp_tree()
