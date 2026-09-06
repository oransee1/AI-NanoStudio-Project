import pyvista as pv
cubemap = pv.examples.download_sky_box_cube_map()
skybox = cubemap.to_skybox()
print(hasattr(skybox, 'SetFloorPlane'))
