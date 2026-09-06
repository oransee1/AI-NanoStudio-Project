import pyvista as pv
mesh = pv.read('final_game_asset_car.glb')

def print_tree(m, name="Root", ind=""):
    pts = m.n_points if hasattr(m, 'n_points') else 0
    print(f"{ind}{name} ({type(m).__name__}) - {pts} pts")
    if hasattr(m, 'n_blocks'):
        for i in range(m.n_blocks):
            bname = m.keys()[i] if m.keys() else f"Block_{i}"
            print_tree(m[i], bname, ind + "  ")

print_tree(mesh)
