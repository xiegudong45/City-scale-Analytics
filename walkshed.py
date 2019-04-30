import networkx as nx
# from datetime import datetime
import math
import pandas as pd
import csv
import json
# import pytz

# art, drinking_fountain, public_restroom, hospital, dog_off_leash_areas
# dataset
sidewalk_csv = "./data/output/final_sidewalk.csv"
crossing_csv = "./data/raw_data/crossings.csv"
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

        print('G[u][v][\'time\']', G[u][v]['time'])
        print('G[v][u][\'time\']', G[v][u]['time'])

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

        print('G[u][v][\'time\']', G[u][v]['time'])
        print('G[v][u][\'time\']', G[v][u]['time'])


    file.close()
    return G


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


# Walksheds:
def walkshed(G, node, max_cost=15):
    """
    Use single_source_dijkstra to calculate all the
    paths that a people can reach within max_cost time
    starting from a node.
    :param G: networkx graph
    :param node: starting point
    :param max_cost: time threshold (unit: minutes)
    :return: paths: all the edges that a people can reach within
    max_cost(min) time.
    sums: sum of the attributes along the path.
    """
    sum_columns = ["time", "art_num", 'fountain_num', 'restroom_num', 'dog_num', 'hospital_num']
    # Use Dijkstra's method to get the below-400 walkshed paths
    distances, paths = nx.algorithms.shortest_paths.single_source_dijkstra(
        G,
        node,
        weight="cost",
        cutoff=max_cost
    )

    # We need to do two things:
    # 1) Grab any additional reachable fringe edges. The user cannot traverse them in their
    #    entirety, but if they are traversible at all we still want to sum up their attributes
    #    in the walkshed. In particular, we're using "length" in this example. It is not obvious
    #    how we should assign "partial" data to these fringes, we will will just add the fraction
    #    to get to max_cost.
    #
    # 2) Sum up the attributes of the walkshed, assigning partial data on the fringes proportional
    #    to the fraction of marginal cost / edge cost.

    # Enumerate the fringes along with their fractional costs.
    fringe_edges = {}
    for n, distance in distances.items():
        for successor in G.successors(n):
            if successor not in distances:
                cost = G[n][successor]["time"]
                if cost is not None:
                    marginal_cost = max_cost - distance
                    fraction = marginal_cost / cost
                    fringe_edges[(n, successor)] = fraction

    # Create the sum total of every attribute in sum_columns
    edges = []
    for destination, path in paths.items():
        for node1, node2 in zip(path, path[1:]):
            edges.append((node1, node2))

    # All unique edges for which to sum attributes
    edges = set(edges)

    sums = {k: 0 for k in sum_columns}

    for n1, n2 in edges:
        d = G[n1][n2]
        for column in sum_columns:
            # print(d.get(column, 0))
            sums[column] += d.get(column, 0)

    # TODO: add in fringes!
    # for (n1, n2), fraction in fringe_edges.items():
    #     d = G[n1][n2]
    #     for column in sum_columns:
    #         sums[column] += d.get(column, 0) * fraction

    # return sums, list(zip(*paths))[1]
    return sums, paths, edges


def extract_node_from_string(node_str):
    """
    Convert node from string type to a list.
    :param node_str: nodes in the path
    :return: a list which represents the
    coordinates of a node.
    """
    coords = node_str.strip("()").split(",")
    return coords


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
        # # art number
        # if pd.notna(row['art']):
        #     art = row['art'].strip('[]').split(',')
        #     art_num = len(art)
        # else:
        #     art_num = 0

        # add the attributes to each edge
        # G[v][u]['art_num'] = art_num
        # G[u][v]['art_num'] = art_num

        G[v][u]['fountain_num'] = fountain_num
        G[u][v]['fountain_num'] = fountain_num

        G[v][u]['restroom_num'] = restroom_num
        G[u][v]['restroom_num'] = restroom_num

        G[v][u]['hospital_num'] = hospital_num
        G[u][v]['hospital_num'] = hospital_num

        G[v][u]['dog_num'] = dog_num
        G[u][v]['dog_num'] = dog_num


def paths_to_geojson(paths, filename, edges_set):
    """
    Store the path into a geojson file.
    :param paths: the path which calculate by
    single_source_dijkstra algorithm.
    :param filename: directory of the output file
    :return: None
    """
    output = dict()
    output["type"] = "FeatureCollection"
    output['features'] = []
    node_set = set()

    for it in paths.items():
        path = it[1]
        one_path = dict()
        one_path['type'] = 'Feature'
        one_path['geometry'] = {}
        one_path['geometry']['type'] = 'LineString'
        line_lst = []
        for i in range(0, len(path)):
            node_set.add(path[i])
            coords = extract_node_from_string(str(path[i]))
            line = [float(coords[0]), float(coords[1])]
            line_lst.append(line)
        one_path['geometry']['coordinates'] = line_lst
        output['features'].append(one_path)

    # line_lst = []
    # for i in range(1, len(path)-1):
    #     coords1 = extract_node_from_string(str(path[i]))
    #     coords2 = extract_node_from_string(str(path[i+1]))
    #     line = LineString((float(coords1[0]), float(coords1[1])), (float(coords2[0]), float(coords2[1])))
    #     line_lst.append(line)
    # gc = GeometryCollection(line_lst)
    # return gc.geojson

    for node1 in node_set:
        node1_neighbors = [n for n in G.neighbors(node1)]
        for node2 in node1_neighbors:
            curr_edge = (node1, node2)
            if node2 in node_set and not (curr_edge in edges_set):
                line = []
                node1_coords = extract_node_from_string(node1)
                node2_coords = extract_node_from_string(node2)
                point1 = [float(node1_coords[0]), float(node1_coords[1])]
                point2 = [float(node2_coords[0]), float(node2_coords[1])]
                line.append(point1)
                line.append(point2)

                one_path = {}
                one_path['type'] = 'Feature'
                one_path['geometry'] = {}
                one_path['geometry']['type'] = 'LineString'
                one_path['geometry']['coordinates'] = line
                output['features'].append(one_path)
                edges_set.add(curr_edge)

    with open(filename, 'w') as outfile:
        json.dump(output, outfile)
    outfile.close()


def start_pt_to_geojson(start_node, filename):
    """
    Store the start point to a geojson file.
    :param start_node: start point node (coordinate)
    :param filename: directory of the file
    :return: None
    """
    node = start_node.strip('()').split(', ')
    for i in range(len(node)):
        node[i] = float(node[i])
    vis_point_d = dict()
    vis_point_d['type'] = "FeatureCollection"
    vis_point_d['features'] = []
    geometry = dict()
    geometry['geometry'] = dict()
    geometry['geometry']['type'] = 'Point'
    geometry['geometry']['coordinates'] = node
    vis_point_d['features'].append(geometry)

    with open(filename, 'w') as outfile:
        json.dump(vis_point_d, outfile)
    outfile.close()


def main():
    print("preparing graph...")
    generate_sdwk_network(sidewalk_csv)
    
    generate_crossing_network(crossing_csv)
    #G = ent.graphs.digraphdb.DiGraphDB('18 AU/data_db/sidewalks.db')
    join_attributes_to_node(G)
    # nx.write_edgelist(G, 'edgelist.txt')

    print("finished preparing graph!")
    print()

    # edge_gen(G)
    start_node = "(-122.3323077, 47.6105820)"
    print("computing walkshed starting from ", start_node)
    sums, paths, edges_set = walkshed(G, start_node)

    print("Number of arts: ", sums["art_num"])
    print('Number of public restrooms: ', sums['restroom_num'])
    print('Number of drinking fountains: ', sums['fountain_num'])
    print('Number of public hospitals: ', sums['hospital_num'])
    print('Number of dog off-leash areas: ', sums['dog_num'])
    # print("Total length: ", sums["time"])

    print("output paths in walkshed...")

    path_filename = './walkshed test/test_walkshed.geojson'
    paths_to_geojson(paths, path_filename, edges_set)
    start_pt_filename = './walkshed test/start_node.geojson'
    start_pt_to_geojson(start_node, start_pt_filename)

    print('Done!')


if __name__ == "__main__":
    main()
