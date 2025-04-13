import open3d as o3d

path = "/home/fabio/dev/Arch/Training/10_SStefano_portico_1.txt"
pcd = o3d.io.read_point_cloud("/")
o3d.visualization.draw_geometries([pcd])