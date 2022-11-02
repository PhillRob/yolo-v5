import geopandas as gpd
import pandas as pd
import string as str
from shapely.geometry import Polygon
data_folder = '/Users/philipp/Projects/PycharmProjects/mesh/'
dataset = pd.read_csv(data_folder+'detected_Dhibiyah_Drone_GeoTiff_005.txt', sep='\s+',header=None)
# data set cleaning script

# function to return polygon
def bbox(x1, y1, x2, y2):
	# world file content
	# Line 1: A: x-component of the pixel width (x-scale)
	xscale = 0.1617903116883119
	# Line 2: D: y-component of the pixel width (y-skew)
	yskew = 0
	# Line 3: B: x-component of the pixel height (x-skew)
	xskew = 0
	# Line 4: E: y-component of the pixel height (y-scale), typically negative
	yscale = -0.1617903116883119
	# Line 5: C: x-coordinate of the center of the original image's upper left pixel transformed to the map
	xpos = 655854.20159515587147325
	# Line 6: F: y-coordinate of the center of the original image's upper left pixel transformed to the map
	ypos = 2716038.70000312989577651

	##print(long0, lat0, lat1, long1)

	X_proj = xpos + (xscale * x1) + (xskew * y1)
	Y_proj = ypos + (yscale * y1) + (yskew * x1)

	X1_proj = xpos + (xscale * x2) + (xskew * y2)
	Y1_proj = ypos + (yscale * y2) + (yskew * x2)

	return Polygon([[X_proj, Y_proj],
	                [X1_proj, Y_proj],
	                [X1_proj, Y1_proj],
	                [X_proj, Y1_proj]])


outGDF=gpd.GeoDataFrame(geometry = dataset.apply(lambda g: bbox(int(g[0]),int(g[1]),int(g[2]),int(g[3])),axis=1),crs = {'init':'epsg:32638'})


# test
# x1, y1, x2, y2 =int(dataset.loc[:0, 0]), int(dataset.loc[:0, 1]), int(dataset.loc[:0, 2]), int(dataset.loc[:0, 3])


outGDF.to_file('detected_Dhibiyah_Drone_GeoTiff_005.shp')


