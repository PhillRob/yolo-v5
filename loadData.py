import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import open3d as o3d
import pyvista as pv
data_folder = '/Users/philipp/Projects/PycharmProjects/las-exploration/'
dataset = "points.xyz"
# polygon vector
annotations = gpd.read_file('/Users/philipp/Projects/PycharmProjects/las-exploration/annotations.json')
annotations = gpd.read_file('/Users/philipp/Projects/PycharmProjects/mesh/output-merged-v1.geojson')
#print(annotations.head())
annotations = annotations.to_crs("EPSG:32638")
# point cloud import
npcloud = np.loadtxt(data_folder + dataset, delimiter=',')


#Transform to geopandas GeoDataFrame
crs = None
geometry = [Point(xyz) for xyz in npcloud]
geodf = gpd.GeoDataFrame(npcloud, crs=crs, geometry=geometry)
geodf.crs = {'init' :'epsg:32638'} # set correct spatial reference'init' :
pointInPolys = gpd.tools.sjoin(geodf, annotations, predicate="within", how='left')
pointInPolysNA = pointInPolys.dropna(subset=['id'])
pointInPolysNP=pointInPolysNA.to_numpy()
# print(pointInPolys1.id)

my_dict = {"id": [], "volume": []}
for i in pointInPolysNA.id.unique():
	print(i)
	subset = pointInPolysNP[pointInPolysNA.id == i, ]
	subsetb = subset
	subsetb[:,2] = min(subset[:, 2])
	base = np.append(subset, subsetb, axis=0)
	#base = np.delete(base, 7, 1)
	geodf1 = gpd.GeoDataFrame(base, crs=crs, geometry=[Point(xyz) for xyz in base])
	base = np.unique(geodf1, axis=0)
	base[0,0]
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(base[:, :3])
	#pcd.estimate_normals()
	#pcd.compute_convex_hull()
	try:
		ch = pcd.compute_convex_hull()
		chv = ch[0].get_volume()
	except RuntimeError:
		#ch = 0
		chv =0

	my_dict["id"].append(i)
	my_dict["volume"].append(chv)


merged = annotations.merge(pd.DataFrame.from_dict(my_dict), on='id')
merged.to_file('dataframe1.shp')

