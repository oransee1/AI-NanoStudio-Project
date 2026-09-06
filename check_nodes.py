import pyvista as pv

mesh = pv.read('final_game_asset_car.glb')

def get_pts(m):
    if hasattr(m, 'n_points'):
        return m.n_points
    if hasattr(m, 'n_blocks'):
        total = 0
        for i in range(m.n_blocks):
            if m[i] is not None:
                total += get_pts(m[i])
        return total
    return 0

for i, m in enumerate(mesh):
    print(f"Node_{i}: {get_pts(m)} points")
