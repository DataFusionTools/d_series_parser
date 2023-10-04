import pytest
import numpy as np
import pickle
import libpysal


from core.data_input import Geometry, Data, Variable
from d_series_parser.clustering import ClusteringLayers
from interpolation.interpolation_inv_distance_per_depth import InverseDistancePerDepth
from interpolation.interpolation_2d_slice import Interpolate2DSlice
from spatial_utils.ahn_utils import SpatialUtils
from utils import TestUtils


class TestClustering:

    @staticmethod
    def get_clustering_points(depth_method=True,
                              location_1=None,
                              location_2=None,
                              cpts_list=None,
                              spacial_utils=None):
        """
        Get the points for the clustering test case using the depth method or not.
        :param depth_method: bool
        :param location_1: Geometry
        :param location_2: Geometry
        :param cpts_list: List[Data]
        :param spacial_utils: SpatialUtils
        :return: points_2d_slice, results_2d_slice, variance
        """
        if depth_method:
            interpolation_method = InverseDistancePerDepth(nb_near_points=4, power=1)
            interpolator = Interpolate2DSlice()
            points_2d_slice, results_2d_slice, variance = interpolator.get_2d_slice_per_depth_inverse_distance(
                location_1=location_1,
                location_2=location_2,
                data=cpts_list,
                interpolate_variable="IC",
                number_of_points=100,
                number_of_independent_variable_points=120,
                interpolation_method=interpolation_method,
                top_surface=spacial_utils.AHN_data,
                bottom_surface=np.array(
                    [[location_1.x, location_1.y, -10], [location_2.x, location_2.y, -10]]
                ),
            )
            return points_2d_slice, results_2d_slice, variance
        else:
            interpolator = Interpolate2DSlice()
            points_2d_slice, results_2d_slice = interpolator.get_2d_slice_extra(
                location_1=location_1,
                location_2=location_2,
                data=cpts_list,
                interpolate_variable="IC",
                number_of_points=100,
                number_of_independent_variable_points=120,
                top_surface=spacial_utils.AHN_data,
                bottom_surface=np.array(
                    [[location_1.x, location_1.y, -10], [location_2.x, location_2.y, -10]]
                ),
            )
            return points_2d_slice, results_2d_slice, None


    @pytest.mark.parametrize("depth_method", [True, False])
    def test_cluster(self, depth_method):
        input_files = str(
            TestUtils.get_test_files_from_local_test_dir("", "test_case_DF.pickle")[0]
        )
        with open(input_files, "rb") as f:
            (cpts, resistivity, insar) = pickle.load(f)
        # create List[Data]
        cpts_list = []
        for name, item in cpts.items():
            location = Geometry(x=item["coordinates"][0], y=item["coordinates"][1], z=0)
            cpts_list.append(
                Data(
                    location=location,
                    independent_variable=Variable(value=item["NAP"], label="NAP"),
                    variables=[
                        Variable(value=item["water"], label="water"),
                        Variable(value=item["tip"], label="tip"),
                        Variable(value=item["IC"], label="IC"),
                        Variable(value=item["friction"], label="friction"),
                    ],
                )
            )
        assert len(cpts_list) == 10
        # get ahn line
        spacial_utils = SpatialUtils()
        surface_line = []
        for i in np.arange(63063, 63222, 4):
            surface_line.append([i, 387725])
        spacial_utils.get_ahn_surface_line(np.array(surface_line))
        assert len(spacial_utils.surface_line) == len(surface_line)

        # get a 2d point cloud slice via interpolation
        location_1 = Geometry(x=63222, y=387725, z=0)
        location_2 = Geometry(x=63063, y=387725, z=0)

        points_2d_slice, results_2d_slice, variance = self.get_clustering_points(depth_method=depth_method,
                                                                                 location_1=location_1,
                                                                                 location_2=location_2,
                                                                                 cpts_list=cpts_list,
                                                                                 spacial_utils=spacial_utils)
        # re-arange results
        for counter, points in enumerate(points_2d_slice):
            for double_count, row in enumerate(points_2d_slice[counter]):
                row.append(results_2d_slice[counter][double_count])
                points_2d_slice[counter][double_count] = row
        points_2d_slice = np.array(points_2d_slice)
        points_2d_slice = np.reshape(
            points_2d_slice, (points_2d_slice.shape[0] * points_2d_slice.shape[1], 4)
        )
        points_2d_slice = np.array(
            [points_2d_slice.T[0, :], points_2d_slice.T[2, :], points_2d_slice.T[3, :]]
        ).T
        # define dataset
        cluster_model = ClusteringLayers()
        cluster_model.cluster_2d_surface_agglomerative_clusterin(points_2d_slice,
                                                                 k_candidates=4,
                                                                 cluster_variables=["value"],
                                                                 spatial_connectivity_methods=libpysal.weights.Queen,
                                                                 run_dash_app=False)

        assert len(cluster_model.simplified_polygons) == 4
        assert len(cluster_model.extracted_value_per_polygon) == 4

    @pytest.mark.intergrationtest
    def test_cluster_bug(self):
        input_files = str(
            TestUtils.get_test_files_from_local_test_dir("", "multipolygon.pickle")[0]
        )
        with open(input_files, "rb") as f:
            points_2d_slice = pickle.load(f)
        cluster_model = ClusteringLayers()
        cluster_model.cluster_2d_surface_agglomerative_clusterin(points_2d_slice,
                                                                 k_candidates=4,
                                                                 cluster_variables=["value"],
                                                                 spatial_connectivity_methods=libpysal.weights.Queen,
                                                                 run_dash_app=False)
        assert len(cluster_model.simplified_polygons) == len(cluster_model.extracted_value_per_polygon)
        assert len(cluster_model.extracted_std_per_polygon) == len(cluster_model.extracted_value_per_polygon)


    @pytest.mark.intergrationtest
    def test_cluster_bug_new(self):
        input_files = str(
            TestUtils.get_test_files_from_local_test_dir("", "test.pickle")[0]
        )
        with open(input_files, "rb") as f:
            points = pickle.load(f)
        cluster_model = ClusteringLayers()
        cluster_model.cluster_2d_surface_agglomerative_clusterin(points,
                                                                 cluster_variables=["value"],
                                                                 spatial_connectivity_methods=libpysal.weights.Queen,
                                                                 k_candidates=20)
        assert len(cluster_model.simplified_polygons) == len(cluster_model.extracted_value_per_polygon)
        assert len(cluster_model.extracted_std_per_polygon) == len(cluster_model.extracted_value_per_polygon)
