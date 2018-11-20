import csv
import json

dir1 = "../data/park_only_boundary1.geojson"
f = open(dir1)
data = json.load(f)
print(data.keys())
print(data['features'][0].keys())
print(len(data['features']))
print(data['features'][0]['properties'].keys())
f.close()


out_file = open('../data/park_only_boundary1.csv', 'w', encoding='utf-8')
csv_file = csv.writer(out_file)
csv_file.writerow(["name", "shape_area", "geometry"])


out_file.close()