from shapely.geometry import LineString

def cut_edge(line, distance):
    """
    Cuts a Shapely LineString at the stated distance. Returns a list of two new
    LineStrings for valid inputs. If the distance is 0, negative, or longer than the
    LineString, a list with the original LineString is produced.

    :param line (shapely.geometry.LineString): LineString to cut
    :param distance (float): Distance along the line where it will be cut.
    :return: list of LineString
    """

    def point_distance(p1, p2):
        """
        Calculate l2 norm of two points
        :param p1 (list): point1
        :param p2 (list): point2
        :return: distance of p1p2.
        """

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        return (dx ** 2 + dy ** 2) ** 0.5

    if distance <= 0.0 or distance >= line.length:
        return [LineString(line)]

    coords = list(line.coords)

    overall_point_distance = 0
    last_point = coords[0]
    for idx, point in enumerate(coords):
        if idx == 0:
            continue
        overall_point_distance += point_distance(last_point, point)

        if overall_point_distance == distance:
            return [LineString(coords[: idx + 1]), LineString(coords[idx:])]

        if overall_point_distance > distance:
            cut_point = line.interpolate(distance)
            return [LineString(coords[:idx] + [(cut_point.x, cut_point.y)]),
                    LineString([(cut_point.x, cut_point.y)] + coords[idx:])
                   ]
        last = point

    cut_point = line.interpolate(distance)
    return [LineString(coords[:idx] + [(cut_point.x, cut_point.y)]),
            LineString([(cut_point.x, cut_point.y)] + coords[idx:])
            ]










