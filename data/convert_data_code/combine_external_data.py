import json
import pandas as pd

def combine_data(file_name1, file_name2, file_name3, file_name4):
    output = dict()
    output["type"] = "FeatureCollection"
    output['features'] = []

    count = 1
    # hospitals
    with open(file_name1) as f1:
        data = json.load(f1)

    for feature in data['features']:
        entry = dict()
        entry['type'] = 'Feature'
        entry['properties'] = dict()
        entry['properties']['id'] = count
        entry['properties']['name'] = feature['properties']['FACILITY']
        entry['properties']['address'] = feature['properties']['ADDRESS']
        entry['properties']['type'] = 'hospital'

        entry['geometry'] = feature['geometry']

        output['features'].append(entry)
        count += 1

    f1.close()

    # drinking fountains
    with open(file_name2) as f2:
        data = json.load(f2)

    for feature in data['features']:
        entry = dict()
        entry['type'] = 'Feature'
        entry['properties'] = dict()
        entry['properties']['id'] = count
        entry['properties']['type'] = 'drinking_fountain'

        entry['geometry'] = feature['geometry']

        output['features'].append(entry)
        count += 1

    f2.close()

    # Dog off
    with open(file_name3) as f3:
        data = json.load(f3)

    for feature in data['features']:
        entry = dict()
        entry['type'] = 'Feature'
        entry['properties'] = dict()
        entry['properties']['id'] = count
        entry['properties']['name'] = feature['properties']['name']
        entry['properties']['type'] = 'dog_off_leash_area'

        entry['geometry'] = feature['geometry']

        output['features'].append(entry)
        count += 1

    f3.close()

    # public restroom
    with open(file_name4) as f4:
        data = json.load(f4)

    for feature in data['features']:
        entry = dict()
        entry['type'] = 'Feature'
        entry['properties'] = dict()
        entry['properties']['id'] = count
        entry['properties']['park'] = feature['properties']['park']
        entry['properties']['type'] = 'public_restroom'

        entry['geometry'] = feature['geometry']

        output['features'].append(entry)
        count += 1

    f4.close()

    with open('../all_external_data.geojson', 'w') as outfile:
        json.dump(output, outfile)
    outfile.close()

def main():
    file_name1 = '../external_data/Hospitals.geojson'
    file_name2 = '../external_data/Drinking Fountain.geojson'
    file_name3 = '../external_data/Dog Off Leash Areas.geojson'
    file_name4 = '../external_data/Public Restroom.geojson'
    combine_data(file_name1, file_name2, file_name3, file_name4)

if __name__ == '__main__':
    main()
