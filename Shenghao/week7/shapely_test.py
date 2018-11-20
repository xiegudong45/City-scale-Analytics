import json
from shapely.geometry import shape, GeometryCollection

with open("../data/park_only_boundary1.geojson") as f:
    features = json.load(f)["features"]
    print(features)
    # print(type(feature['geometry']))
# NOTE: buffer(0) is a trick for fixing scenarios where polygons have overlapping coordinates
GeometryCollection([shape(features["geometry"]).buffer(0) for feature in features])