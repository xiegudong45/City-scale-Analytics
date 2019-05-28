import json
import pandas as pd

with open('../raw_data/transportation.geojson') as f:
    data = json.load(f)

df_crossing = pd.DataFrame(columns=['ID',
                                    'curbramps',
                                    'length',
                                    'description',
                                    'incline',
                                    'surface',
                                    'width',
                                    'layer',
                                    'elevator',
                                    'indoor',
                                    'opening_hours',
                                    'v_coordinates',
                                    'u_coordinates'])

df_sidewalk = pd.DataFrame(columns=[
                                    'ID',
                                    'curbramps',
                                    'length',
                                    'description',
                                    'incline',
                                    'surface',
                                    'width',
                                    'layer',
                                    'elevator',
                                    'indoor',
                                    'opening_hours',
                                    'v_coordinates',
                                    'u_coordinates'])

side_id = 0
cros_id = 0
count = 0

for feature in data['features']:

    count += 1
    print("current count: ", count)
    properties = feature["properties"]

    if 'footway' in properties.keys() :
        if properties['footway'] == 'crossing':
            coor_lst = feature['geometry']['coordinates']
            cros_id += 1
            df_crossing.loc[df_crossing.shape[0]] = [
                            cros_id,
                            properties['curbramps'] if 'curbramps' in properties else "N/A",
                            properties['length'] if 'length' in properties else "N/A",
                            properties['description'] if 'description' in properties else "N/A",
                            properties['incline'] if 'incline' in properties else "N/A",
                            properties['surface'] if 'surface' in properties else "N/A",
                            properties['width'] if 'width' in properties else "N/A",
                            properties['layer'] if 'layer' in properties else "N/A",
                            properties['elevator'] if 'elevator' in properties else "N/A",
                            properties['indoor'] if 'indoor' in properties else "N/A",
                            properties['opening_hours'] if 'opening_hours' in properties else "N/A",
                            "(" + str(coor_lst[0][0]) + "," + str(coor_lst[0][1]) + ")",
                            "(" + str(coor_lst[1][0]) + "," + str(coor_lst[1][1]) + ")"]


        elif properties['footway'] == 'sidewalk':
            coor_lst = feature['geometry']['coordinates']
            side_id += 1
            df_sidewalk.loc[df_sidewalk.shape[0]] = [
                side_id,
                properties['curbramps'] if 'curbramps' in properties else "N/A",
                properties['length'] if 'length' in properties else "N/A",
                properties['description'] if 'description' in properties else "N/A",
                properties['incline'] if 'incline' in properties else "N/A",
                properties['surface'] if 'surface' in properties else "N/A",
                properties['width'] if 'width' in properties else "N/A",
                properties['layer'] if 'layer' in properties else "N/A",
                properties['elevator'] if 'elevator' in properties else "N/A",
                properties['indoor'] if 'indoor' in properties else "N/A",
                properties['opening_hours'] if 'opening_hours' in properties else "N/A",
                "(" + str(round(coor_lst[0][0], 7)) + "," + str(round(coor_lst[0][1], 7)) + ")",
                "(" + str(round(coor_lst[-1][0], 7)) + "," + str(round(coor_lst[-1][1], 7)) + ")"]


print("============")
print("count: ", count)
# print("sidewalk: ", side_id)
print("crossing: ", cros_id)

filename_sidewalk = "../raw_data/sidewalks.csv"
df_sidewalk.to_csv(filename_sidewalk,  encoding='utf-8')

filename_crossing = "../raw_data/crossings.csv"
df_crossing.to_csv(filename_crossing, encoding='utf-8')
