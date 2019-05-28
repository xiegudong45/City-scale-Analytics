import unweaver
import entwiner
import networkx as nx
import simplejson as json
from shapely.geometry import mapping, shape
import copy
from unweaver.geo import cut
import math
import csv



# parameters
WALK_BASE = 1.3
WHEELCHAIR_BASE = 0.6
POWERED_BASE = 2

DIVISOR = 5
INCLINE_IDEAL = -0.0087

# dict_keys(['_layer', '_geometry', 'subclass', 'footway',
# 'length', 'description', 'incline', 'surface', 'width', 'layer'])

# _layer
# _geometry: {'type': 'LineString',
#               'coordinates': [[-122.23708053, 47.50964081],
#                               [-122.2382473299999, 47.51031620999999],
#                               [-122.23825555, 47.51032063]
#                             ]
#             }


def add_attributes(G, csv_file):
    file = open(csv_file)

    for row in csv.DictReader(file):
        # start node
        coordinates = row["v_coordinates"][11: -1].split(',')
        xu = "%.7f" % float(coordinates[0].split(' ')[0])
        yu = "%.7f" % float(coordinates[0].split(' ')[1])
        u = str(xu) + ', ' + str(yu)

        # end node
        coordinates = row["u_coordinates"][1: -1].split(',')
        xv = "%.7f" % float(coordinates[1].split(' ')[0])
        yv = "%.7f" % float(coordinates[1].split(' ')[1])
        v = str(xv) + ', ' + str(yv)

    file.close()


def cost_function_generator(base_speed=WALK_BASE, avoidCurbs=True, downhill=0.1, uphill=0.085):
    def find_k(g, m, n):
        return math.log(n) / abs(g - m)

    def tobler_function(grade, k=3.5, m=INCLINE_IDEAL, base=WALK_BASE):
        # Modified to be in meters / second rather than km / h
        return base * math.exp(-k * abs(grade - m))

    def cost_function(u, v, d):
        k_down = find_k(-downhill, INCLINE_IDEAL, DIVISOR)
        k_up = find_k(uphill, INCLINE_IDEAL, DIVISOR)

        time = 0
        speed = base_speed

        length = d["length"]
        subclass = d["subclass"]

        if subclass == "footway":
            if "footway" in d:
                if d["footway"] == "sidewalk":
                    incline = float(d["incline"])
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

                elif d["footway"] == "crossing":
                    time += 30

        time += length / speed
        return time


    return cost_function


def precalculate_weight(G, weight_column, cost_fun_generator):
    cost_function = cost_fun_generator()

    batch = []
    for i, (u, v, edge) in enumerate(G.iter_edges()):
        weight = cost_function(u, v, edge)
        if len(batch) == 1000:
            G.update_edges(batch)
            batch = []
        batch.append((u, v, {weight_column: weight}))
    if batch:
        G.update_edges(batch)


def shortest_path(G, start_node, cost_fun, max_cost=15):

    costs, paths = nx.algorithms.shortest_paths.single_source_dijkstra(
        G,
        start_node,
        weight=cost_fun,
        cutoff=max_cost * 60
    )
    # for reached_node, cost in costs.items():
    #     if

    edge_ids = list(set([(u, v) for path in paths.values() for u, v in zip(path, path[1:])]))

    def edge_generator(G, edge_ids):
        for u, v in edge_ids:
            edge = dict(G[u][v])
            edge['_u'] = u
            edge['_v'] = v
            yield edge

    edges = list(edge_generator(G, edge_ids))

    nodes = {}
    # costs: {node: cost}
    for node_id, cost in costs.items():
        # print(G.node[node_id]['_geometry'])

        nodes[node_id] = {**G.node[node_id], "cost": cost}  # unpack the key-value pairs in the dictionary
        # print(nodes[node_id])

    def change_coordinates(nodes):
        for node_id in nodes.keys():
            px = nodes[node_id]['_geometry']['coordinates'][0]
            py = nodes[node_id]['_geometry']['coordinates'][1]
            px = round(px, 7)
            py = round(py, 7)
            nodes[node_id]['_geometry']['coordinates'] = [px, py]
        return nodes

    nodes = change_coordinates(nodes)
    # for node_id in nodes.keys():
    #     print('new nodes', nodes[node_id])

    traveled_edges = set((e['_u'], e['_v']) for e in edges)
    traveled_nodes = set([n for path in paths.values() for n in path])

    def get_all_edges(edges):
        edge_set = set()
        for item in edges:
            u = item['_u']
            v = item['_v']
            edge_set.add((u, v))
        return edge_set

    edge_set = get_all_edges(edges)

    fringe_candidates = {}
    for u in traveled_nodes:
        for v in G.neighbors(u):
            print('u', u)
            if (u, v) in traveled_edges:
                continue

            traveled_edges.add((u, v))

            edge = dict(G[u][v])
            cost = cost_fun(u, v, edge)

            if v in traveled_nodes:
                new_edge = dict(G[u][v])
                ux = new_edge['_geometry']['coordinates'][0][0]
                uy = new_edge['_geometry']['coordinates'][0][1]
                vx = new_edge['_geometry']['coordinates'][1][0]
                vy = new_edge['_geometry']['coordinates'][1][1]

                new_u = str(round(ux, 7)) + ', ' + str(round(uy, 7))
                new_v = str(round(vx, 7)) + ', ' + str(round(vy, 7))

                new_edge['_u'] = new_u
                new_edge['_v'] = new_v

                edges.append(new_edge)

            if cost is None:
                continue

            if v in nodes and nodes[v]["cost"] + cost < max_cost:
                interpolate_proportion = 1
            else:
                remaining = max_cost - nodes[u]["cost"]
                interpolate_proportion = remaining / cost
            edge['_u'] = u
            edge['_v'] = v

            fringe_candidates[(u, v)] = {
                'cost': cost,
                'edge': edge,
                'proportion': interpolate_proportion
            }

    def make_partial_edge(edge, proportion):
        geom = shape(edge["_geometry"])
        geom_length = geom.length
        interpolate_distance = proportion * geom_length

        fringe_edge = copy.deepcopy(edge)
        fringe_edge['_geometry'] = mapping(cut(geom, interpolate_distance)[0])
        fringe_point = geom.interpolate(interpolate_distance)
        fringe_node_id = "{}, {}".format(*list(fringe_point.coords)[0])
        fringe_node = {"_key": fringe_node_id, "_geometry": mapping(fringe_point)}
        fringe_edge['_v'] = fringe_node_id

        return fringe_edge, fringe_node

    fringe_edges = []
    # print('1 ', type(fringe_edges))
    seen = set()

    for edge_id, candidate in fringe_candidates.items():
        if edge_id in seen:
            continue

        edge = candidate['edge']
        proportion = candidate['proportion']
        cost = candidate['cost']

        if proportion == 1:
            fringe_edges.append(edge)
            continue

        rev_edge_id = (edge_id[1], edge_id[0])
        reverse_intersected = False
        has_reverse = rev_edge_id in fringe_candidates
        if has_reverse:
            # This edge is "internal": it's being traversed from both sides
            rev_proportion = fringe_candidates[rev_edge_id]["proportion"]
            if proportion + rev_proportion > 1:
                # They intersect - the entire edge can be traversed.
                fringe_edges.append(edge)
                continue
            else:
                # They do not intersect. Keep the original proportions
                pass

        # If this point has been reached, this is:
        # (1) A partial extension down an edge
        # (2) It doesn't overlap with any other partial edges

        # Create primary partial edge and node and append to the saved data
        fringe_edge, fringe_node = make_partial_edge(edge, proportion)

        fringe_edges.append(fringe_edge)
        fringe_node_id = fringe_node.pop("_key")

        nodes[fringe_node_id] = {**fringe_node, "cost": max_cost}

        seen.add(edge_id)
    edges = edges + fringe_edges

    return nodes, paths, edges


def extract_node_from_string(node_str):
    """
    Convert node from string type to a list.
    :param node_str: nodes in the path
    :return: a list which represents the
    coordinates of a node.
    """
    coords = node_str.strip("()").split(",")
    return coords


def paths_to_geojson(filename, edges, start_node):
    """
    Store the path into a geojson file.
    :param filename: directory of the output file
    :type: filename: string

    :param edges: list of edges which stores all the edges in the walkshed
    :type: list

    :param start_node: the start point's coordinate
    :type: string
    :return: None
    """
    output = dict()
    output["type"] = "FeatureCollection"
    output['features'] = []
    node = start_node.strip('()').split(', ')
    for i in range(len(node)):
        node[i] = float(node[i])

    geometry = dict()
    geometry['geometry'] = dict()
    geometry['geometry']['type'] = 'LineString'
    geometry['geometry']['coordinates'] = node
    output['features'].append(geometry)

    for item in edges:
        one_edge = dict()
        one_edge['type'] = 'Feature'
        one_edge['geometry'] = {}
        one_edge['geometry']['type'] = 'LineString'

        u_lst = item['_u'].split(', ')
        ux = float(u_lst[0])
        uy = float(u_lst[1])

        v_lst = item['_v'].split(', ')
        vx = float(v_lst[0])
        vy = float(v_lst[1])

        u = [ux, uy]
        v = [vx, vy]
        edge_pt_lst = (u, v)
        one_edge['geometry']['coordinates'] = edge_pt_lst
        output['features'].append(one_edge)


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
    layers_files = ["../data/raw_data/transportation.geojson"]
    db_path = "../data/unweaver/graph.db"
    # G = entwiner.build.create_graph(layers_files, db_path, batch_size=10000, changes_sign=['incline'])
    G = entwiner.DiGraphDB(path=db_path)
    precalculate_weight(G, 'time', cost_function_generator)

    start_node = "-122.3323077, 47.610582"

    lat = 47.610582
    lon = -122.3323077
    candidates = unweaver.network_queries.dwithin.candidates_dwithin(G, lon, lat, 1)
    #
    candidates_dict = dict(candidates)

    cost_function = cost_function_generator()
    max_cost = 15 * 60
    # nodes, edges = unweaver.algorithms.reachable.reachable(G, candidates_dict, cost_function, max_cost)
    # print(nodes)
    # print(edges)

    nodes, paths, edges = shortest_path(G, start_node, cost_function)
    # print(edges)

    path_filename = './test_walkshed1.geojson'
    paths_to_geojson(path_filename, edges, start_node)
    # start_pt_filename = './start_node.geojson'
    # start_pt_to_geojson(start_node, start_pt_filename)


if __name__ == '__main__':
    main()
