import pandas as pd
import csv

origin_file = '../output/new_sw_wth_fountain_restroom.csv'

df_sidewalk = pd.DataFrame(columns=[
                                    'subclass',
                                    'footway',
                                    'crossing',
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
                                    'drinking_fountain',
                                    'public_restroom',
                                    'hospital',
                                    'dog_off_leash_areas',
                                    'u_coordinates',
                                    'v_coordinates'])

with open(origin_file, 'r') as file:
    read_csv = csv.reader(file, delimiter=',')
    i = 0
    for row in read_csv:
        if i > 0:
            coor_lst = row[-5][12:-1].split(",")
            u_coor = "(" + str(coor_lst[0].split(" ")[0]) + "," + str(coor_lst[0].split(" ")[1]) + ")"
            v_coor = "(" + str(coor_lst[1][1:].split(" ")[0]) + "," + str(coor_lst[1][1:].split(" ")[1]) + ")"

            df_sidewalk.loc[df_sidewalk.shape[0]] = [

                row[1],
                row[2],
                row[3],
                row[4],
                row[5],
                row[6],
                row[7],
                row[8],
                row[9],
                row[10],
                row[11],
                row[12],
                row[13],
                row[-4],
                row[-3],
                row[-2],
                row[-1],
                u_coor,
                v_coor
            ]
            # if i == 3:
            #     break
        print(i)
        i += 1

file.close()

df_sidewalk.to_csv("../output/final_sidewalk.csv")
