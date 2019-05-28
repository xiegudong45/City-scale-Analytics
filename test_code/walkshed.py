import networkx as nx
import math
import pandas as pd
import csv
import json

# art, drinking_fountain, public_restroom, hospital, dog_off_leash_areas
# dataset













# Walksheds:
# def walkshed(G, node, cost_function, max_cost=15):
#     """
#     Use single_source_dijkstra to calculate all the
#     paths that a people can reach within max_cost time
#     starting from a node.
#     :param G: networkx graph
#     :param node: starting point
#     :param max_cost: time threshold (unit: minutes)
#     :return: paths: all the edges that a people can reach within
#     max_cost(min) time.
#     sums: sum of the attributes along the path.
#     """
#     sum_columns = ["time", 'fountain_num', 'restroom_num', 'dog_num', 'hospital_num']
#     # Use Dijkstra's method to get the below-400 walkshed paths
#     nodes, paths, edges = shortest_paths(G, node, cost_function, max_cost)
#
#     traveled_edges = set((e['_u'], e['_v']) for e in edges)
#     traveled_nodes = set([n for path in paths.values() for n in path])
#
#     fringe_candidates = {}
#
#     for u in traveled_nodes:
#         for v in G.neighbors(u):
#             if (u, v) in traveled_edges:
#                 continue
#             traveled_edges.add((u, v))
#
#             edge = dict(G[u][v])
#             cost = cost_function(u, v, edge)
#
#             if cost is None:
#                 continue
#
#             if v in nodes and nodes[v]['time'] + cost < max_cost:
#                 interpolate_proportion = 1
#             else:
#                 remaining = max_cost - nodes[u]
#                 interpolate_proportion = remaining / cost
#
#             edge["_u"] = u
#             edge["_v"] = v
#
#             fringe_candidates[(u, v)] = {
#                 "cost": cost,
#                 "edge": edge,
#                 "proportion": interpolate_proportion,
#             }
#
#     def make_partial_edge(edge, proportion):
#         # Create edge and pseudonode
#         # TODO: use real length
#         data = {'type': 'LineString', 'coordinates': [(edge['_u'][0], edge['_u'][1]), (edge['_v'][0], edge['_v'][1])]}
#         geom = shape(data)
#         geom_length = geom.length
#         interpolate_distance = proportion * geom_length
#
#         # Create a new edge with pseudo-node
#         fringe_edge = copy.deepcopy(edge) ## ???
#         fringe_edge["_geometry"] = mapping(cut_edge(geom, interpolate_distance)[0])
#         fringe_point = geom.interpolate(interpolate_distance)
#         fringe_node_id = "({}, {})".format(*list(fringe_point.coords)[0])
#         fringe_node = {"_key": fringe_node_id, "_geometry": mapping(fringe_point)}
#         fringe_edge["_v"] = fringe_node_id
#
#         return fringe_edge, fringe_node
#
#     fringe_edges = []
#     seen = set()
#
#     # Don't treat origin point edge as fringe-y: each start point in the shortest-path
#     # tree was reachable from the initial half-edge.
#
#     # started = list(set([path[0] for target, path in paths.items()]))
#
#     for edge_id, candidate in fringe_candidates.items():
#         # Skip already-seen edges (e.g. reverse edges we looked ahead for).
#         if edge_id in seen:
#             continue
#
#         edge = candidate["edge"]
#         proportion = candidate["proportion"]
#         cost = candidate["cost"]
#
#         # Can traverse whole edge - keep it
#         if proportion == 1:
#             fringe_edges.append(edge)
#             continue
#
#         rev_edge_id = (edge_id[1], edge_id[0])
#         reverse_intersected = False
#         has_reverse = rev_edge_id in fringe_candidates
#         if has_reverse:
#             # This edge is "internal": it's being traversed from both sides
#             rev_proportion = fringe_candidates[rev_edge_id]["proportion"]
#             if proportion + rev_proportion > 1:
#                 # They intersect - the entire edge can be traversed.
#                 fringe_edges.append(edge)
#                 continue
#             else:
#                 # They do not intersect. Keep the original proportions
#                 pass
#
#         # If this point has been reached, this is:
#         # (1) A partial extension down an edge
#         # (2) It doesn't overlap with any other partial edges
#
#         # Create primary partial edge and node and append to the saved data
#         fringe_edge, fringe_node = make_partial_edge(edge, proportion)
#
#         fringe_edges.append(fringe_edge)
#         fringe_node_id = fringe_node.pop("_key")
#
#         nodes[fringe_node_id] = {**fringe_node, "cost": max_cost}
#
#         seen.add(edge_id)
#
#     edges = edges + fringe_edges
#
#     return nodes, edges


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

    nodes, paths, edges = shortest_paths(G, start_node, cal_time)
    print(type(edges))
    sums, paths, edges_se = walkshed(G, start_node)

    # print("Number of arts: ", sums["art_num"])
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
