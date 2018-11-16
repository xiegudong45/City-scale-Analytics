import json
from shapely.geometry import shape, GeometryCollection

with open("C:\\Users\\xiegudong45\\Desktop\\CSE495\\City-scale-Analytics\\Shenghao\\data\\park_only_boundary1.geojson") as f:
  features = json.load(f)["features"]

# NOTE: buffer(0) is a trick for fixing scenarios where polygons have overlapping coordinates
GeometryCollection([shape(feature["geometry"]).buffer(0) for feature in features])