import csv
import json

csv_file = '../output/final_sidewalk.csv'

external_lst = dict()
with open(csv_file, 'r') as f:
    read_csv = csv.reader(f, delimiter=',')
    i = 0
    for row in read_csv:
        if i > 0:
            entry_dict = dict()
            u = row[-2][1:-1]
            v = row[-1][1:-1]
            entry_dict['df_num'] = row[-6]
            entry_dict['pb_num'] = row[-5]
            entry_dict['h_num'] = row[-4]
            entry_dict['dof_num'] = row[-3]
            key = "(" + u + " " + v + ")"

            external_lst[key] = entry_dict
            # print(key)
            # break
        else:
            i = 1
    print('ex_lst', len(external_lst))
f.close()

final_dict = dict()
with open('../raw_data/transportation.geojson') as f:
    data = json.load(f)
    # print(data.keys())
    final_dict['type'] = 'FeatureCollection'
    final_dict['features'] = []

    valid_count = 0
    for item in data['features']:
        # print('key' ,item['properties'].keys())
        entry_dic = dict()

        # create key
        ux = str(round(float(item['geometry']['coordinates'][0][0]), 7))
        uy = str(round(float(item['geometry']['coordinates'][0][1]), 7))
        vx = str(round(float(item['geometry']['coordinates'][-1][0]), 7))
        vy = str(round(float(item['geometry']['coordinates'][-1][1]), 7))
        target_u = ux + ", " + uy
        target_v = vx + ", " + vy
        target_key = "(" + target_u + " " + target_v + ")"

        entry_dic['type'] = 'Feature'

        if 'footway' in item['properties'].keys():
            if item['properties']['footway'] == 'sidewalk' and target_key in external_lst.keys():
                valid_count += 1
                entry_dic['properties'] = {
                                            **item['properties'],
                                            'df_num': external_lst[target_key]['df_num'],
                                            'pb_num': external_lst[target_key]['pb_num'],
                                            'h_num': external_lst[target_key]['h_num'],
                                            'dof_num': external_lst[target_key]['dof_num']
                                            }
            else:
                entry_dic['properties'] = item['properties']
        else:
            entry_dic['properties'] = item['properties']

        entry_dic['geometry'] = item['geometry']
        final_dict['features'].append(entry_dic)
    print('count', valid_count)
f.close()

with open('../output/tranportation_amend.geojson', 'w') as outfile:
    json.dump(final_dict, outfile)
outfile.close()
