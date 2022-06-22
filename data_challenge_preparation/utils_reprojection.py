import pyproj


def ecef_to_wgs84(x, y, z):
 """
 Reproject ECDF coordinate into lat/lng WGS84 coordinates
 """
 lat, lon, alt = pyproj.Transformer.from_crs("epsg:4978", "epsg:4979").transform(x, y, z)
 return [lon, lat, alt]


def wgs84_to_ecef(lng, lat, alt):
 """
 Reproject  lat/lng WGS84 coordinates into ECDF coordinate
 """
 x,y,z = pyproj.Transformer.from_crs("epsg:4979", "epsg:4978").transform(lng, lat, alt)
 return [x,y,z]

