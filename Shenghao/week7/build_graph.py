import matplotlib.pyplot as plt
import csv
import pandas as pd
import json
from shapely.geometry.polygon import LinearRing, Polygon
from descartes.patch import PolygonPatch
from shapely.geometry import mapping
import os



#prepare: the following two variables need to change.
default_dir = '/home/xiegudong45/Desktop/City-scale-Analytics/' # dis may alter with different local machine.
buffer_ratio = 0.0003    # buffering ratio we'd like to alter. (0.0001 - 0.0005) 0.0003 is ok right now.

upper = 47.6868735
lower = 47.6649775
left = -122.354941
right = -122.317496



sidewalk_dir = default_dir + 'data_table/sidewalks.csv'    # original sidewalks
small_sidewalk_dir = default_dir + 'data_table/small_sidewalks.csv'

# park_boundary = '/home/xiegudong45/Desktop/City-scale-Analytics/Shenghao/data/park_only_boundary1.csv'

origin_pb_dir = default_dir + 'Shenghao/data/park_only_boundary1.geojson'

small_boundary_dir = default_dir + 'Shenghao/data/small_boundary.csv'


def make_small_sw_csv():
    """
    extract all the streets around green lake area, stores the information in a csv file.
    :return: a csv file contains the street name and its linestring.
    """
    if not os.path.isfile(origin_pb_dir):

        df = pd.DataFrame(columns=['id', 'forward', 'incline', 'layer', 'side', 'street_name',
                                   'surface', 'width', 'length', 'v_coordinates', 'u_coordinates'])

        file = open(sidewalk_dir)
        for row in csv.DictReader(file):
            v = row['v_coordinates']
            u = row['u_coordinates']
            v = v.replace('(', '').replace(')', '')
            u = u.replace('(', '').replace(')', '')
            vx = float(v.split(',')[0])
            vy = float(v.split(',')[1])
            ux = float(u.split(',')[0])
            uy = float(u.split(',')[1])
            ok = left <= vx <= right and left <= ux <= right and lower <= vy <= upper and lower <= uy <= upper
            if ok:
                df.loc[df.shape[0]] = [row['ID'],
                                       row['forward'],
                                       row['incline'],
                                       row['layer'],
                                       row['side'],
                                       row['street_name'],
                                       row['surface'],
                                       row['width'],
                                       row['length'],
                                       v,
                                       u]
        df.to_csv(small_sidewalk_dir,  encoding='utf-8')
        file.close()


def plot_map(geo_lst):
    """
    plot the small area, return the polygon for selected park boundaries.
    :param geo_lst: dict that contains all the park boundary around greenlake with its name
    :return: enlarged park boundaries.
    """
    plt.figure(dpi=700)
    ax = plt.axes()
    enlarged_pb_dict = dict()
    with open(small_sidewalk_dir, 'r') as f:
        read_csv = csv.reader(f, delimiter=',')
        count = 0
        for row in read_csv:
            if count != 0:
                ux = float(row[-1].split(',')[0])
                uy = float(row[-1].split(',')[1])
                vx = float(row[-2].split(',')[0])
                vy = float(row[-2].split(',')[1])

                plot_coords(ux, uy, vx, vy)
            count += 1
    for pb_name in geo_lst.keys():
        if pb_name == 'GREEN LAKE PARK':
            outside_poly = geo_lst[pb_name][1][0]
        else:
            outside_poly = geo_lst[pb_name][0][0]
        x_lst = []
        y_lst = []
        for point in outside_poly:
            x_lst.append(point[0])
            y_lst.append(point[1])
        plt.plot(x_lst, y_lst)
        polygon = Polygon(outside_poly)
        patch = PolygonPatch(polygon, facecolor=[1, 1, 1], edgecolor=[0, 0.8, 0], alpha=0.5, zorder=2)
        t = Polygon(polygon.buffer(buffer_ratio))
        t_patch = PolygonPatch(t, facecolor=[1, 1, 1], edgecolor=[1, 0.5, 1], alpha=0.3, zorder=2)
        enlarged_pb_dict[pb_name] = mapping(t)
        ax.add_patch(patch)
        ax.add_patch(t_patch)
    return enlarged_pb_dict     # return enlarged park boundary with polygon


def plot_coords(ux, uy, vx, vy):
    """
    plot streets
    :param ux: x coordinates for u
    :param uy: y coordinates for u
    :param vx: x coordinates for v
    :param vy: y coordinates for v
    :return: none
    """
    x = [vx, ux]
    y = [vy, uy]
    plt.plot(x, y, 'k', linewidth=1.0)


# def make_small_pb_csv():
#     df = pd.DataFrame(columns=['id', 'name',
#                           'shape_area',
#                           'geometry'])
#
#     file = open(park_boundary)
#     count = 0
#     for row in csv.DictReader(file):
#
#         #
#         # print(row)
#         mp = row['geometry']
#         mp = mp.replace('[', '')
#         mp = mp.replace(']', '')
#         mp = mp.replace(' ', '')
#         split_lst = mp.split(',')
#         # print(split_lst)
#         ok = True
#         # if count == 2:
#         for i in range(0, len(split_lst), 2):
#             px = float(split_lst[i])
#             py = float(split_lst[i + 1])
#             if px < left or px > right or py < lower or py > upper:
#                 ok = False
#         if ok:
#             df.loc[df.shape[0]] = [row['id'], row['name'], row['shape_area'], row['geometry']]
#         # elif count == 3:
#         #     break
#         count += 1
#
#     df.to_csv(small_boundary_dir, encoding='utf-8')
#     file.close()


def extract_gl_wpz():
    """
    extract the geo info from park_only_boundary1.geojson
    :return:
    """
    with open(origin_pb_dir) as f:
        data = json.load(f)
    f.close()
    id = 0
    count = 0
    geo_lst = dict()
    for feature in data['features']:
        count += 1
        id += 1
        properties = feature["properties"]
        name = properties['name']
        if name == 'WOODLAND PARK ZOO' or name == 'LINDEN ORCHARD PARK' or \
                name == 'GREEN LAKE PARK' or name == 'WOODLAND PARK':

            geo_lst[name] = feature['geometry']['coordinates']
    return geo_lst





def main():
    make_small_sw_csv()
    geo_dict = extract_gl_wpz()
    enlarged_pb_dict = plot_map(geo_dict)
    print(enlarged_pb_dict)
    # make_small_pb_csv()   #not work for some reason.

    plt.show()


if __name__ == '__main__':
    main()