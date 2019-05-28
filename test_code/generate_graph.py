import networkx as nx
import math
import pandas as pd
import csv
import json
from unweaver.algorithms.reachable import reachable

sidewalk_csv = "../data/output/final_sidewalk.csv"
crossing_csv = "../data/raw_data/crossings.csv"
sw = pd.read_csv(sidewalk_csv)
G = nx.Graph()

# parameters
WALK_BASE = 1.3
WHEELCHAIR_BASE = 0.6
POWERED_BASE = 2

DIVISOR = 5
INCLINE_IDEAL = -0.0087


def generate_sdwk_network(csv_file):
    """
    Given sidewalk csv files, add them into networkx graph.
    based on coordinates.
    :param csv_file: a csv file which stores all the sidewalks
    in Seattle.
    :return: None
    """
    file = open(csv_file)

    for row in csv.DictReader(file):
        # attr_lst = ['drinking_fountain']
        # print(row['drinking_fountain'])
        # break
        # start node
        coordinates = row["v_coordinates"][1: -1].split(',')
        xu = "%.7f" % float(coordinates[0])
        yu = "%.7f" % float(coordinates[1])
        u = '(' + str(xu) + ', ' + str(yu) + ')'

        # end node
        coordinates = row["u_coordinates"][1: -1].split(',')
        xv = "%.7f" % float(coordinates[0])
        yv = "%.7f" % float(coordinates[1])
        v = '(' + str(xv) + ', ' + str(yv) + ')'

        if not G.has_node(v):
            G.add_node(v, pos_x=xv, pos_y=yv)

        if not G.has_node(u):
            G.add_node(u, pos_x=xu, pos_y=yu)

        incline = float(row["incline"])

        # surface = str(row["surface"])

        length = float(row["length"])

        # edge
        G.add_edge(u, v)
        G[u][v]['subclass'] = 'footway'
        G[u][v]['footway'] = 'sidewalk'

        G[u][v]['incline'] = incline
        G[u][v]['length'] = length
        # print('uv incline', G[u][v]['incline'])
        # G[u][v]['surface'] = surface

        G.add_edge(v, u)

        G[v][u]['subclass'] = 'footway'
        G[v][u]['footway'] = 'sidewalk'

        G[v][u]['incline'] = -1 * incline
        G[v][u]['length'] = length
        # print('vu incline', G[v][u]['incline'])
        # G[v][u]['surface'] = surface

        G[u][v]['time'] = cal_time(G[u][v])
        G[v][u]['time'] = cal_time(G[v][u])

        # print('G[u][v][\'time\']', G[u][v]['time'])
        # print('G[v][u][\'time\']', G[v][u]['time'])

    file.close()
    return G


def generate_crossing_network(csv_file):
    """
    Given crossing csv files, add them into networkx graph.
    based on coordinates
    :param csv_file: a csv file which stores all the crossings
    in Seattle
    :return: None
    """
    file = open(csv_file)

    for row in csv.DictReader(file):

        # start node
        coordinates = row["v_coordinates"][1: -1].split(',')
        xu = "%.7f" % float(coordinates[0])
        yu = "%.7f" % float(coordinates[1])
        u = '(' + str(xu) + ', ' + str(yu) + ')'

        # end node
        coordinates = row["u_coordinates"][1: -1].split(',')
        xv = "%.7f" % float(coordinates[0])
        yv = "%.7f" % float(coordinates[1])
        v = '(' + str(xv) + ', ' + str(yv) + ')'

        if not G.has_node(v):
            G.add_node(v, pos_x=xv, pos_y=yv)

        if not G.has_node(u):
            G.add_node(u, pos_x=xu, pos_y=yu)

        # incline
        incline = 0
        length = float(row["length"])
        # marked
        # marked = int(row["marked"])
        # curbramps
        curbramps = int(row["curbramps"])

        # edge
        G.add_edge(u, v)
        G[u][v]['subclass'] = 'footway'
        G[u][v]['footway'] = 'crossing'
        G[u][v]['length'] = length
        G[u][v]['curbramps'] = curbramps

        G.add_edge(v, u)
        G[v][u]['subclass'] = 'footway'
        G[v][u]['footway'] = 'crossing'
        G[v][u]['length'] = length
        G[v][u]['curbramps'] = curbramps

        if row['incline'] == 'N/A':
            G[u][v]['incline'] = 0
            G[v][u]['incline'] = 0
        else:
            G[u][v]['incline'] = float(row['incline'])
            G[v][u]['incline'] = -1 * float(row['incline'])

        G[u][v]['time'] = cal_time(G[u][v])
        G[v][u]['time'] = cal_time(G[v][u])

        # print('G[u][v][\'time\']', G[u][v]['time'])
        # print('G[v][u][\'time\']', G[v][u]['time'])

    file.close()
    return G


def join_attributes_to_node(G):
    """
    From the new_sw_collections.csv, extract the attribute of
    each sidewalk and add them onto the relative edges.
    :param G: networkx Graph
    :return: None
    """
    for idx, row in sw.iterrows():
        coordinates = row["v_coordinates"][1: -1].split(',')
        xv = "%.7f" % float(coordinates[0])
        yv = "%.7f" % float(coordinates[1])
        v = '(' + str(xv) + ', ' + str(yv) + ')'

        # end node
        coordinates = row["u_coordinates"][1: -1].split(',')
        xu = "%.7f" % float(coordinates[0])
        yu = "%.7f" % float(coordinates[1])
        u = '(' + str(xu) + ', ' + str(yu) + ')'

        # fountain number
        if pd.notna(row['drinking_fountain']):
            fountain = row['drinking_fountain'].strip('[]').split(',')
            fountain_num = len(fountain)

        else:
            fountain_num = 0
            # print('fountain_num', fountain_num)

        # restroom number
        if pd.notna(row['public_restroom']):
            restroom = row['public_restroom'].strip('[]').split(',')
            restroom_num = len(restroom)

        else:
            restroom_num = 0

        # hospital number
        if pd.notna(row['hospital']):
            hospital = row['hospital'].strip('[]').split(',')
            hospital_num = len(hospital)
        else:
            hospital_num = 0

        # dog off leash area number
        if pd.notna(row['dog_off_leash_areas']):
            dog = row['dog_off_leash_areas'].strip('[]').split(',')
            dog_num = len(dog)
        else:
            dog_num = 0
        #

        G[v][u]['fountain_num'] = fountain_num
        G[u][v]['fountain_num'] = fountain_num

        G[v][u]['restroom_num'] = restroom_num
        G[u][v]['restroom_num'] = restroom_num

        G[v][u]['hospital_num'] = hospital_num
        G[u][v]['hospital_num'] = hospital_num

        G[v][u]['dog_num'] = dog_num
        G[u][v]['dog_num'] = dog_num

        # print(G[v][u].keys())


def find_k(g, m, n):
    return math.log(n) / abs(g - m)


def tobler_function(grade, k=3.5, m=INCLINE_IDEAL, base=WALK_BASE):
    # Modified to be in meters / second rather than km / h
    return base * math.exp(-k * abs(grade - m))


def get_speed(edge, base_speed=WALK_BASE, downhill=0.1, uphill=0.085):
    k_down = find_k(-downhill, INCLINE_IDEAL, DIVISOR)
    k_up = find_k(uphill, INCLINE_IDEAL, DIVISOR)

    time = 0

    speed = base_speed

    length = edge["length"]
    subclass = edge["subclass"]

    if subclass == "footway":
        if "footway" in edge:
            if edge["footway"] == "sidewalk":
                incline = float(edge["incline"])
                # Decrease speed based on incline
                if length > 3:
                    if incline > uphill:
                        speed = tobler_function(incline, k=uphill, m=INCLINE_IDEAL, base=base_speed)
                    if incline < -downhill:
                        speed = tobler_function(incline, k=-downhill, m=INCLINE_IDEAL, base=base_speed)

                if incline > INCLINE_IDEAL:
                    speed = tobler_function(incline, k=k_up, m=INCLINE_IDEAL, base=base_speed)
                else:
                    speed = tobler_function(incline, k=k_down, m=INCLINE_IDEAL, base=base_speed)

            elif edge["footway"] == "crossing":
                incline = float(edge["incline"])
                # if avoidCurbs:
                #     if "curbramps" in edge:
                #         if not edge["curbramps"]:
                #             return None
                #     else:
                #         # TODO: Make this user-configurable - we assume no
                #         # curb ramps by default now
                #         return None
                # Add delay for crossing street
                # TODO: tune this based on street type crossed and/or markings.
                if incline != 0:
                    speed = tobler_function(incline, k=0, m=INCLINE_IDEAL, base=base_speed)
                time += 30

    return time, speed, length


def cal_time(edge, base_speed=WALK_BASE, downhill=0.1, uphill=0.085, avoidCurbs=True, timestamp=None):
    """Calculates a cost-to-travel that balances distance vs. steepness vs.
    needing to cross the street.
    :param downhill: Maximum downhill incline indicated by the user, e.g.
                     0.1 for 10% downhill.
    :type downhill: float
    :param uphill: Positive incline (uphill) maximum, as grade.
    :type uphill: float
    :param avoidCurbs: Whether curb ramps should be avoided.
    :type avoidCurbs: bool
    """

    time, speed, length = get_speed(edge, base_speed=WALK_BASE, downhill=0.1, uphill=0.085)

    time += length / speed
    return time


def shortest_paths(G, source_node, cost_fun, max_cost=15):
    costs, paths = nx.algorithms.shortest_paths.single_source_dijkstra(
        G,
        source_node,
        weight=cost_fun,
        cutoff=max_cost
    )

    edge_ids = list(set([(u, v) for path in paths.values() for u, v in zip(path, path[1:])]))

    def edge_generator(G, edge_ids):
        for u, v in edge_ids:
            edge = dict(G[u][v])
            edge['_u'] = u
            edge['_v'] = v
            yield edge

    edges = edge_generator(G, edge_ids)
    print(edges)

    nodes = {}
    # costs: {node: cost}
    for node_id, cost in costs.items():
        nodes[node_id] = {**G.node[node_id], "cost": cost}  # **: unpack the key-value pairs in the dictionary
    return nodes, paths, edges

def main():
    print("preparing graph...")
    generate_sdwk_network(sidewalk_csv)

    generate_crossing_network(crossing_csv)
    join_attributes_to_node(G)

    print("finished preparing graph!")
    print()

    # edge_gen(G)
    start_node = "(-122.3323077, 47.6105820)"
    print("computing walkshed starting from ", start_node)

    nodes, paths, edges = shortest_paths(G, start_node, cal_time)




if __name__ == '__main__':
    main()
