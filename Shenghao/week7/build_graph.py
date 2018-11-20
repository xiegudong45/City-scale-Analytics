import shapely
import matplotlib.pyplot as plt
import csv
import pandas as pd
from shapely.geometry.polygon import LinearRing, Polygon
from shapely.geometry import Point, LineString


upper = 47.6868735
lower = 47.6649775
left = -122.354941
right = -122.317496


sidewalk_dir = '/Users/studentuser/Desktop/City-scale-Analytics/data_table/sidewalks.csv'
small_sidewalk_dir = '/Users/studentuser/Desktop/City-scale-Analytics/data_table/small_sidewalks.csv'
park_boundary = '/Users/studentuser/Desktop/City-scale-Analytics/Shenghao/data/park_only_boundary1.csv'
small_boundary_dir = '/Users/studentuser/Desktop/City-scale-Analytics/Shenghao/data/small_boundary.csv'


def make_small_sw_csv():
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


def plot_map():
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


def plot_coords(ux, uy, vx, vy):
    x = [vx, ux]
    y = [vy, uy]
    plt.plot(x, y, 'k')


def make_small_pb_csv():
    df = pd.DataFrame(columns=['id', 'name',
                          'shape_area',
                          'geometry'])

    file = open(park_boundary)
    count = 0
    for row in csv.DictReader(file):

        #
        # print(row)
        mp = row['geometry']
        mp = mp.replace('[', '')
        mp = mp.replace(']', '')
        mp = mp.replace(' ', '')
        split_lst = mp.split(',')
        # print(split_lst)
        ok = True
        # if count == 2:
        for i in range(0, len(split_lst), 2):
            px = float(split_lst[i])
            py = float(split_lst[i + 1])
            if px < left or px > right or py < lower or py > upper:
                ok = False
        if ok:
            df.loc[df.shape[0]] = [row['id'], row['name'], row['shape_area'], row['geometry']]
        # elif count == 3:
        #     break
        count += 1

    df.to_csv(small_boundary_dir, encoding='utf-8')
    file.close()






def main():
    # make_small_sw_csv()()
    # plot_map()
    make_small_pb_csv()
    # plt.show()


if __name__ == '__main__':
    main()