import pyvista as pv
plotter = pv.Plotter(off_screen=True)
cubemap = pv.examples.download_sky_box_cube_map()
skybox = cubemap.to_skybox()
skybox.RotateX(90)
plotter.add_actor(skybox)
plotter.camera.up = (0, 0, 1)
plotter.screenshot('skybox_test3.png')
plotter.close()
