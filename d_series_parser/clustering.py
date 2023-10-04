import itertools
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Union, Any, Tuple
import pandas

import numpy as np
import shapely.geometry as geom
import topojson as tp
from scipy.spatial import Delaunay, Voronoi
from sklearn.cluster import AgglomerativeClustering
from shapely.ops import unary_union
import shapely

from core.base_class import BaseClass

import matplotlib.path as pth


def alpha_shape(points, alpha, only_outer=True) -> set:
    """
    Compute the alpha shape (concave hull) of a set of points.

    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border or also inner edges.

    :returns: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are the indices in the points array.

    """
    assert points.shape[0] > 3, "Need at least four points"

    def add_edge(edges, i, j):
        """Add an edge between the i-th and j-th points, if not in the list already."""
        if (i, j) in edges or (j, i) in edges:
            # already added
            assert (j, i) in edges, "Can't go twice over same directed edge right?"
            if only_outer:
                # if both neighboring triangles are in shape, it's not a boundary edge
                edges.remove((j, i))
            return
        edges.add((i, j))

    tri = Delaunay(points)
    edges = set()
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.simplices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            add_edge(edges, ia, ib)
            add_edge(edges, ib, ic)
            add_edge(edges, ic, ia)
    return edges


def find_edges_with(i: int, edge_set: Dict) -> Tuple[List, List]:
    """Function that finds edges where point i is included.

    :param i: Lookup point index
    :param edge_set: List of all edges

    :returns: First and second edges were the point is included
    """
    i_first = [j for (x, j) in edge_set if x == i]
    i_second = [j for (j, x) in edge_set if x == i]
    return i_first, i_second


def stitch_boundaries(edges: Dict) -> List:
    """Function that stitches edges (list of lines) into consecutive edges.

    :param edges: List
    :param edges: List

    :returns: List of sorted boundaries
    """
    edge_set = edges.copy()
    boundary_lst = []
    while len(edge_set) > 0:
        boundary = []
        edge0 = edge_set.pop()
        boundary.append(edge0)
        last_edge = edge0
        while len(edge_set) > 0:
            i, j = last_edge
            j_first, j_second = find_edges_with(j, edge_set)
            if j_first:
                edge_set.remove((j, j_first[0]))
                edge_with_j = (j, j_first[0])
                boundary.append(edge_with_j)
                last_edge = edge_with_j
            elif j_second:
                edge_set.remove((j_second[0], j))
                edge_with_j = (j, j_second[0])  # flip edge rep
                boundary.append(edge_with_j)
                last_edge = edge_with_j

            if edge0[0] == last_edge[1]:
                break

        boundary_lst.append(boundary)
    return boundary_lst


@dataclass
class ClusteringLayers(BaseClass):
    """
    Class that clusters points with a certain value to a polygon list

    :param points: List of initial points
    :param normalized_points: List of normalized points
    :param clusters: Clusters of points after the analysis
    :param global_polygon_list: Created polygon list with a dense points in edges
    :param extracted_value_per_polygon: Value extracted per polygon
    :param extracted_std_per_polygon: Std extracted per polygon

    """

    points: Union[List, None, np.array] = None
    normalized_points: Union[List, None, np.array] = None
    clusters: Union[List, None, np.array] = None
    global_polygon_list: Union[List, None, np.array] = None
    simplified_polygons: Union[List, None, np.array] = None
    extracted_value_per_polygon: Union[List, None, np.array] = None
    extracted_std_per_polygon: Union[List, None, np.array] = None

    def normalise_data(self):
        """
        Normalize point data with the min max technique.
        """
        x = (self.points[:, 0] - min(self.points[:, 0])) / (
                max(self.points[:, 0]) - min(self.points[:, 0])
        )
        y = (self.points[:, 1] - min(self.points[:, 1])) / (
                max(self.points[:, 1]) - min(self.points[:, 1])
        )
        value = (self.points[:, -1] - min(self.points[:, -1])) / (
                max(self.points[:, -1]) - min(self.points[:, -1])
        )
        self.normalized_points = np.array([x, y, value]).T

    def denormalise_coordinates(self, x: List, y: List):
        """
        Denormalize point data with the min max technique.

        :param x: List of x coordinates
        :param y: List of y coordinates

        :returns: De-normalized tuple of (x, y) coordinates
        """
        x = np.array(x) * (max(self.points[:, 0]) - min(self.points[:, 0])) + min(
            self.points[:, 0]
        )
        y = np.array(y) * (max(self.points[:, 1]) - min(self.points[:, 1])) + min(
            self.points[:, 1]
        )
        return x, y

    def get_encompassing_shape(self) -> shapely.geometry.Polygon:
        """
        Function that gets the encompassing shape of a geometry.
        """
        encopassing_shape_edges = alpha_shape(
            np.array([self.normalized_points[:, 0], self.normalized_points[:, 1]]).T,
            0.01,
        )
        encopassing_shape_boundaries = stitch_boundaries(encopassing_shape_edges)
        points_boundary = np.array(
            [
                (self.normalized_points[i, 0], self.normalized_points[i, 1])
                for i, j in encopassing_shape_boundaries[0]
            ]
        )
        encopassing_shape = geom.Polygon(points_boundary)
        return encopassing_shape

    def evaluation_plots(self, polygons_dataframe):
        """
        Function that creates evaluation plots as a Dash app for the 2D clustering method.

        :param polygons_dataframe: Dataframe with the clustered polygons
        """

        import plotly.express as px
        from dash import Dash, html, dcc

        fig_scatter = px.scatter_matrix(
            polygons_dataframe,
            dimensions=[
                "x",
                "y",
                "value",
            ],
            color="cluster_labels",
            height=1000,
        )
        # distribution plots
        fig_x = px.histogram(
            polygons_dataframe, x="x", color="cluster_labels", marginal="rug"
        )
        fig_y = px.histogram(
            polygons_dataframe, x="y", color="cluster_labels", marginal="rug"
        )
        fig_value = px.histogram(
            polygons_dataframe, x="value", color="cluster_labels", marginal="rug"
        )

        import plotly.graph_objects as go

        x = [
            list(polygon.exterior.xy[0]) + [None]
            for polygon in self.simplified_polygons
        ]
        x = [item for sublist in x for item in sublist]
        y = [
            list(polygon.exterior.xy[1]) + [None]
            for polygon in self.simplified_polygons
        ]
        y = [item for sublist in y for item in sublist]
        fig_final_cluster = go.Figure(go.Scatter(x=x, y=y, fill="toself"))
        fig_final_cluster.add_trace(
            go.Scatter(
                x=self.points.T[0],
                y=self.points.T[1],
                mode="markers",
                marker_color=self.points.T[2],
            ),
        )

        # create dashboard
        app = Dash()
        app.css.config.serve_locally = True
        server = app.server
        app.layout = html.Div(
            children=[
                html.H1(children="2d Slice clustering overview"),
                html.Div(
                    children="""
            Dash: The clustering is based on Spatially Constrained Hierarchical Clustering 
        """
                ),
                dcc.Graph(id="clustered geometry", figure=fig_final_cluster),
                dcc.Graph(id="scatter plot", figure=fig_scatter),
                dcc.Graph(id="dist x", figure=fig_x),
                dcc.Graph(id="dist y", figure=fig_y),
                dcc.Graph(id="dist value", figure=fig_value),
            ]
        )
        app.run_server(debug=True, use_reloader=False, port=8051)

    def get_index_of_polygon_list(
            self, poly: geom.polygon.Polygon, polygon_list: List[geom.polygon.Polygon]
    ):
        """
        Function that returns the index of a polygon if it part of a list

        :param poly: Polygon that the function will search for
        :param polygon_list: List of polygons that for the index to be found

        :returns: the index that the polygon corresponds to
        """
        for counter, polygon in enumerate(polygon_list):
            if polygon == poly:
                return counter

    def intersect_all_polygon_combinations(
            self, polygon_list: List[geom.polygon.Polygon]
    ) -> List[geom.polygon.Polygon]:
        """
        Function that intersects all every polygon in a list with all other polygons.

        :param polygon_list: List of shapely polygons.
        """
        new_polygon_list = polygon_list.copy()
        combination_index_list = list(
            itertools.combinations(range(len(new_polygon_list)), 2)
        )
        for index_a, index_b in combination_index_list:
            a = new_polygon_list[index_a]
            b = new_polygon_list[index_b]
            if not (math.isclose(a.intersection(b).area, 0)):
                new_poly = a.difference(b)
                index = self.get_index_of_polygon_list(a, new_polygon_list)
                new_polygon_list[index] = new_poly
        return new_polygon_list

    def cluster_with_schc_method(
            self,
            n_of_clusters: int,
            encopassing_shape: geom.Polygon,
            cluster_variables: List[str],
            spatial_connectivity_methods: Any,
    ) -> pandas.DataFrame:
        """
        Function that clusters points based on the Spatially Constrained Hierarchical Clustering method.
        More on that method can be read on https://geographicdata.science/book/notebooks/10_clustering_and_regionalization.html

        :param n_of_clusters: Number of clusters that are required
        :param encopassing_shape: 2D slice encopassing shape
        :param cluster_variables: List of variables to cluster with these can be ["x", "y", "value"]
        :param spatial_connectivity_methods: Spatial weights as defined in the libpysal library (for more information https://pysal.org/libpysal/api.html)
        """
        # create polygons to facilitate the polygon creation
        vor = Voronoi(self.normalized_points[:, 0:2])
        # create polygons from delaunay triangles assign value to them
        lines = [
            shapely.geometry.LineString(vor.vertices[line])
            for line in vor.ridge_vertices
            if -1 not in line
        ]
        polygon_mesh = [poly for poly in shapely.ops.polygonize(lines)]
        # filter mesh elements that fall out side the geometry
        filtered_polygon_mesh = []
        for polygon in polygon_mesh:
            poly_path = pth.Path(np.array(polygon.exterior.xy).T)
            points_logic_array = poly_path.contains_points(self.normalized_points[:, 0:2])
            value = np.average(self.normalized_points[:, -1][points_logic_array])
            if not (math.isclose(encopassing_shape.intersection(polygon).area, 0)):
                new_poly = encopassing_shape.intersection(polygon)
                filtered_polygon_mesh.append([new_poly, value, polygon.centroid.xy[0][0], polygon.centroid.xy[1][0]])
            else:
                filtered_polygon_mesh.append([polygon, value, polygon.centroid.xy[0][0], polygon.centroid.xy[1][0]])

        filtered_polygon_mesh = np.array(filtered_polygon_mesh).T

        # create dataframe
        polygons_dataframe = pandas.DataFrame(
            {
                "geometry": filtered_polygon_mesh[0],
                "value": filtered_polygon_mesh[1],
                "x": filtered_polygon_mesh[2],
                "y": filtered_polygon_mesh[3],
                "centroid": [poly.centroid for poly in filtered_polygon_mesh[0]]
            }
        )
        weight_clusters = ["geometry"] + cluster_variables
        w = spatial_connectivity_methods.from_dataframe(
            polygons_dataframe[weight_clusters]
        )
        # Set seed for reproducibility
        np.random.seed(123456)
        # Specify cluster model with spatial constraint
        self.clusters = AgglomerativeClustering(
            linkage="ward", connectivity=w.sparse, n_clusters=n_of_clusters
        )
        # Fit algorithm to the data
        self.clusters.fit(polygons_dataframe[cluster_variables])

        cluster_s = pandas.Series(self.clusters.labels_)
        polygons_dataframe = polygons_dataframe.assign(cluster_labels=cluster_s)
        return polygons_dataframe

    def simplify_polygons(self):
        """
        Function that simplifies exterior of polygons using the douglas-peucker algorithm
        """
        topo = tp.Topology(
            self.global_polygon_list, prequantize=False, prevent_oversimplify=True
        )
        geojson_objects_collection = json.loads(
            topo.toposimplify(epsilon=0.01).to_geojson()
        )
        self.simplified_polygons = [
            geom.shape(geo_object["geometry"])
            for geo_object in geojson_objects_collection["features"]
        ]

    def get_value_per_polygon(self):
        """
        Function that calculates values and standard deviation per polygon.
        """
        self.extracted_value_per_polygon = np.zeros_like(self.simplified_polygons)
        self.extracted_std_per_polygon = np.zeros_like(self.simplified_polygons)
        for counter, polygon in enumerate(self.simplified_polygons):
            collected_values = [
                point[-1]
                for point in self.points
                if polygon.contains(geom.Point(point[0], point[1]))
            ]
            self.extracted_value_per_polygon[counter] = np.average(collected_values)
            self.extracted_std_per_polygon[counter] = np.std(collected_values)

    def cluster_2d_surface_agglomerative_clusterin(
            self,
            points: np.ndarray,
            cluster_variables: List[str],
            spatial_connectivity_methods: Any,
            k_candidates: int = 10,
            run_dash_app: bool = False,
    ):
        """
        Function that clusters a surface depending on the x, y coordinates and the values given by the user.

        :param points: list of x,y locations along with the value that will facilitate the clustering
        :param k_candidates: number of k means candidates
        :param run_dash_app: value that determines if the user want to run a dash application with an evaluation of the
        clustering
        :param cluster_variables: List of variables to cluster with these can be ["x", "y", "value"]
        :param spatial_connectivity_methods: Spatial weights as defined in the libpysal library (for more information https://pysal.org/libpysal/api.html)

        :returns list of clustered polygons
        """
        # check shape of point
        if not (points.shape[1] == 3):
            raise ValueError(
                f"Array inputted should have shape of (N, 3) but was {points.shape}"
            )
        self.points = points
        # normalize data
        self.normalise_data()
        # find encompassing shape
        encopassing_shape = self.get_encompassing_shape()
        # Cluster with the Spatially Constrained Hierarchical Clustering method
        polygons_dataframe = self.cluster_with_schc_method(
            n_of_clusters=k_candidates,
            encopassing_shape=encopassing_shape,
            cluster_variables=cluster_variables,
            spatial_connectivity_methods=spatial_connectivity_methods,
        )
        self.global_polygon_list = []
        grouped = polygons_dataframe.groupby("cluster_labels")

        for name, cluster_group in grouped:
            polygons_to_merge = cluster_group['geometry'].tolist()
            self.global_polygon_list.append(unary_union(polygons_to_merge))

        self.global_polygon_list = self.intersect_all_polygon_combinations(
            self.global_polygon_list
        )
        self.global_polygon_list = [
            polygon.intersection(encopassing_shape) for polygon in self.global_polygon_list
        ]

        self.simplify_polygons()

        # denormalize
        denormalized_polygon_coord = [
            self.denormalise_coordinates(
                list(polygon.exterior.coords.xy[0]), list(polygon.exterior.coords.xy[1])
            )
            for polygon in self.global_polygon_list
        ]
        self.global_polygon_list = [
            geom.Polygon(np.array([coords[0], coords[1]]).T)
            for coords in denormalized_polygon_coord
        ]
        denormalized_polygon_coord = [
            self.denormalise_coordinates(
                list(polygon.exterior.coords.xy[0]), list(polygon.exterior.coords.xy[1])
            )
            for polygon in self.simplified_polygons
        ]
        self.simplified_polygons = [
            geom.Polygon(np.array([coords[0], coords[1]]).T)
            for coords in denormalized_polygon_coord
        ]
        self.get_value_per_polygon()
        if run_dash_app:
            self.evaluation_plots(polygons_dataframe)
