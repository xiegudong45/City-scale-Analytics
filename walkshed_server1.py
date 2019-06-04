#!flask/bin/python

import sys

from flask import Flask, render_template, request, redirect, Response
import random, json
import networkx as nx
import os

import math
import pandas as pd
import csv
import json
import unweaver
import entwiner
import networkx as nx
import simplejson as json
from shapely.geometry import mapping, shape
import copy
from unweaver.geo import cut
import math


# parameters
WALK_BASE = 1.3
WHEELCHAIR_BASE = 0.6
POWERED_BASE = 2

DIVISOR = 5
INCLINE_IDEAL = -0.0087

app = Flask(__name__)

# dataset
db_path = "./data/unweaver/graph.db"
G = entwiner.DiGraphDB(path=db_path)


def save_graph(G, filename):
    json.dump(dict(nodes=[[n, G.node[n]] for n in G.nodes()],
                   edges=[[u, v, G.edge[u][v]] for u, v in G.edges()]),
              open(filename, 'w'), indent=4)


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

    print('11', edges[0]['_u'])
    print('12', edges[0]['_v'])
    fringe_candidates = {}
    for u in traveled_nodes:
        for v in G.neighbors(u):

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
    # print('2 ', type(fringe_edges))
    edges = edges + fringe_edges

    return nodes, paths, edges


def paths_to_geojson(paths,  edges_set):
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
            coords = extract_node_from_string(str(path[i]))
            line = [float(coords[0]), float(coords[1])]
            line_lst.append(line)

        one_path['geometry']['coordinates'] = line_lst

        output['features'].append(one_path)

    return json.dumps(output)


def extract_node_from_string(node_str):
    coords = node_str.strip("()").split(",")
    return coords


def compute_distance(x_lon, x_lat, y_lon, y_lat):
    return math.pow(x_lon - y_lon, 2) + math.pow(x_lat - y_lat, 2)


@app.route("/")
def output():
    return render_template('index.html', name='Joe')


@app.route('/receiver', methods=['GET', 'POST'])
def worker():
    # read json + reply
    data = request.get_json()
    max_time = data['max_time']
    feature = data['feature']

    start_lat = round(float(data['start_lat']), 7)
    lon = round(float(data['start_lon']), 7)
    start_lon = lon - (360 if lon > 0 else 0)
    start_node = str(start_lon) + ", " + str(start_lat)
    print('start_node1w222: ', start_node)
    node_lst = list(G.nodes)
    print('node_lst', len(node_lst))
    # find closest node in G for start node
    if start_node not in node_lst:
        print("true")
        min_dist = float('inf')
        print('min_dist', min_dist)
        print('a', len(list(G.nodes)))
        for node in list(G.nodes):
            print('node', node)
            coords = node.split(', ')
            print("coords", coords)
            dist = compute_distance(start_lon, start_lat, float(coords[0]), float(coords[1]))
            print('dist', dist)
            if dist < min_dist:
                min_dist = dist
                start_node = node
                print('start_node: ', start_node)

    # if feature == "Drinking Fountains":
    #     col = "drinking_fountain_num"
    # elif feature == "Public Restrooms":
    #     col = "public_restroom_num"
    # elif feature == "Hospitals":
    #     col = "hospital_num"
    # elif feature == "Dog Off Leash Areas":
    #     col = "dola_num"
    # else:
    #     raise ValueError("Invalid feature requested!")
    cost_function = cost_function_generator()
    nodes, paths, edges = shortest_path(G, start_node, cost_function, max_cost=int(max_time))
    # print("sum of utilities: ", sums[col])
    result = paths_to_geojson(paths, edges,)
    return result


def main():
    #preprocess
    print("loading data...")
    db_path = "./data/unweaver/graph.db"
    layers_files = ["./data/raw_data/transportation.geojson"]
    # if (os.path.isfile(db_path)):
    G = entwiner.DiGraphDB(path=db_path)
    print('main: ', len(list(G.nodes)))
    # else:
    # G = entwiner.build.create_graph(layers_files, db_path, batch_size=10000, changes_sign=['incline'])

    precalculate_weight(G, 'time', cost_function_generator)

    # join features to network




if __name__ == '__main__':
    main()
    # run!
    app.run(host='localhost', port=5001)
