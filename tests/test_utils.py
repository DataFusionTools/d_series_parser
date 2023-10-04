import pytest
import numpy as np
import pickle
import math

from core.data_input import Data, Variable, Geometry
from utils import TestUtils
from core.utils import CreateInputsML, AggregateMethod


class TestUtilsTools:
    def define_input_data_for_timeseries(self) -> Data:
        input_1 = {
            "displacement": np.array(range(0, 3, 1)),
            "time": np.array(range(40, 43, 1)),
        }
        data = Data(
            location=Geometry(x=5, y=10, z=0),
            variables=[Variable(label="displacement", value=input_1["displacement"])],
            independent_variable=Variable(label="time", value=input_1["time"]),
        )
        return data

    def define_input_data(self) -> Data:
        # initialize inputs
        input = {
            "variable_1": np.array(range(0, 10, 1)),
            "variable_2": np.array(range(20, 30, 1)),
            "variable_3": np.array(range(30, 40, 1)),
            "time": np.array(range(40, 50, 1)),
        }
        variable_1 = Variable(label="variable_1", value=input["variable_1"])
        variable_2 = Variable(label="variable_2", value=input["variable_2"])
        variable_3 = Variable(label="variable_3", value=input["variable_3"])
        location = Geometry(x=1, y=2, z=0)
        data = Data(
            location=location,
            variables=[variable_1, variable_2, variable_3],
            independent_variable=Variable(label="time", value=input["time"]),
        )
        # check initial expectations
        assert len(data.variables) == 3
        return data

    @pytest.mark.unittest
    def test_append_features_use_independent_variable(self):
        data_input = self.define_input_data()
        # initialize ML helper class
        inputs_ml_with_time = CreateInputsML()
        # run test
        inputs_ml_with_time.append_features(
            data_input, ["variable_1", "variable_3"], use_independent_variable=True
        )
        # check expectations
        assert (
            inputs_ml_with_time._input_dump[0]["input"].variables[0].label
            == "variable_1"
        )
        assert (
            inputs_ml_with_time._input_dump[0]["input"].variables[1].label
            == "variable_3"
        )
        assert inputs_ml_with_time._input_dump[0]["use_independent_variable"]

    @pytest.mark.unittest
    def test_append_features_no_use_independent_variable(self):
        data_input = self.define_input_data()
        # initialize ML helper class
        inputs_ml_with_time = CreateInputsML()
        # run test
        inputs_ml_with_time.append_features(
            data_input, ["variable_1", "variable_2"], use_independent_variable=False
        )
        # check expectations
        assert (
            inputs_ml_with_time._input_dump[0]["input"].variables[0].label
            == "variable_1"
        )
        assert (
            inputs_ml_with_time._input_dump[0]["input"].variables[-1].label
            == "variable_2"
        )
        assert not (inputs_ml_with_time._input_dump[0]["use_independent_variable"])

    @pytest.mark.unittest
    def test_add_features(self):
        data_input = self.define_input_data()
        # initialize ML helper class
        inputs_ml_without_time = CreateInputsML()
        # run test
        inputs_ml_without_time.add_features(
            data_input, ["variable_1", "variable_2"], use_independent_variable=False
        )
        # check expectations
        assert list(inputs_ml_without_time._features.keys()) == [
            "variable_1",
            "variable_2",
        ]

        # initialize ML helper class
        inputs_ml_with_time = CreateInputsML()
        # run test
        inputs_ml_with_time.add_features(
            data_input, ["variable_1", "variable_2"], use_independent_variable=True
        )
        # check expectations
        assert list(inputs_ml_with_time._features.keys()) == [
            "variable_1",
            "variable_2",
            "time",
        ]

    @pytest.mark.unittest
    def test_add_targets(self):
        data_input = self.define_input_data()
        # initialize ML helper class
        inputs_ml = CreateInputsML()
        # run test
        inputs_ml.add_targets(data_input, ["variable_2"])
        # check expectations
        assert list(inputs_ml._targets.keys()) == [
            "variable_2",
        ]

    @pytest.mark.unittest
    def test_find_closer_points_extrapolation(self):
        data_input = self.define_input_data()
        column_input = self.define_input_data_for_timeseries()
        # initialize ML helper class
        inputs_ml = CreateInputsML()
        aggregated_features = inputs_ml.find_closer_points(
            input_data=[data_input],
            combined_data=[column_input],
            aggregate_method=AggregateMethod.SUM,
            aggregate_variable="displacement",
            number_of_points=1,
            interpolate_on_independent_variable=True, 
            fill_value = 1000
        )
        assert len(aggregated_features[0].get_variable('displacement').value) == len(aggregated_features[0].get_variable('variable_1').value)
        extrapolated_part = list(aggregated_features[0].get_variable('displacement').value[3:])
        assert np.all([math.isclose(value, 1000.) for value in extrapolated_part])


    @pytest.mark.unittest
    def test_get_all_features(self):
        data_input = self.define_input_data()
        column_input = self.define_input_data_for_timeseries()
        # initialize ML helper class
        inputs_ml = CreateInputsML()
        aggregated_features = inputs_ml.find_closer_points(
            input_data=[data_input],
            combined_data=[column_input],
            aggregate_method=AggregateMethod.SUM,
            aggregate_variable="displacement",
            number_of_points=1,
        )
        inputs_ml.add_features(
            aggregated_features[0],
            ["variable_1", "variable_2", "displacement"],
            use_independent_variable=True,
            use_location_as_input=(False, False, False),
        )

        # run test
        features = inputs_ml.get_all_features(flatten=False)
        assert list(features[-1]) == [9, 29, 3, 49]
        # run test
        features = inputs_ml.get_all_features(flatten=True)
        assert features[0].shape == (40,)

    @pytest.mark.unittest
    def test_split_train_test_data(self):
        data_input = self.define_input_data()
        # initialize ML helper class
        inputs_ml = CreateInputsML()
        for i in range(0, 11):
            inputs_ml.add_features(
                data_input,
                ["variable_1", "variable_2"],
                use_independent_variable=True,
                use_location_as_input=(False, False, False),
            )
            inputs_ml.add_targets(
                data_input,
                ["variable_1"],
            )
        # run test
        inputs_ml.split_train_test_data(
            train_percentage=0.5, validation_percentage_on_test=0.1
        )
        assert len(inputs_ml._features_test["variable_1"]) == len(
            inputs_ml._features_train["variable_1"]
        )
        assert len(inputs_ml._targets_test["variable_1"]) == len(
            inputs_ml._targets_train["variable_1"]
        )
        inputs_ml.split_train_test_data(
            train_percentage=0.8, validation_percentage_on_test=0.1
        )
        assert len(inputs_ml._features_test["variable_1"]) == 2
        assert len(inputs_ml._features_train["variable_1"]) == 8
        assert len(inputs_ml._targets_test["variable_1"]) == 2
        assert len(inputs_ml._targets_train["variable_1"]) == 8

    @pytest.mark.unittest
    def test_get_test_train_data(self):
        data_input = self.define_input_data()
        # initialize ML helper class
        inputs_ml = CreateInputsML()
        for i in range(0, 100):
            inputs_ml.add_features(
                data_input,
                ["variable_1", "variable_2"],
                use_independent_variable=True,
                use_location_as_input=(False, False, False),
            )
            inputs_ml.add_targets(
                data_input,
                ["variable_1"],
            )
        # run test
        inputs_ml.split_train_test_data(
            train_percentage=0.5, validation_percentage_on_test=0.1
        )
        train_features = inputs_ml.get_features_train(flatten=False)
        train_target = inputs_ml.get_targets_train(flatten=False)
        test_features = inputs_ml.get_features_test(flatten=False)
        test_target = inputs_ml.get_targets_test(flatten=False)
        validation_features = inputs_ml.get_features_validation(flatten=False)
        validation_target = inputs_ml.get_targets_validation(flatten=False)
        assert train_features.shape == (500, 3)
        assert train_target.shape == (500, 1)
        assert test_features.shape == (450, 3)
        assert test_target.shape == (450, 1)
        assert validation_features.shape == (40, 3)
        assert validation_target.shape == (40, 1)

    @pytest.mark.unittest
    def test_find_closer_points(self):
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

        insar_list = []
        for counter, coordinates in enumerate(insar["coordinates"]):
            location = Geometry(x=coordinates[0], y=coordinates[1], z=0)
            insar_list.append(
                Data(
                    location=location,
                    independent_variable=Variable(value=insar["time"], label="time"),
                    variables=[
                        Variable(
                            value=insar["displacement"][counter], label="displacement"
                        )
                    ],
                )
            )

        inputs_ml = CreateInputsML()
        aggregated_features = inputs_ml.find_closer_points(
            input_data=cpts_list,
            combined_data=insar_list,
            aggregate_method=AggregateMethod.SUM,
            aggregate_variable="displacement",
            number_of_points=2,
        )
        assert len(aggregated_features) != 0

    def test_interpolate_on_independent_variable_outside_bounds(self):
        variable_1 = Data(
            location=Geometry(1, 1, 0),
            variables=[Variable(label="test", value=np.array([1, 2, 3, 4, 5]))],
            independent_variable=Variable(label="ind", value=np.array([1, 2, 3, 4, 5])),
        )
        variable_2 = Data(
            location=Geometry(2, 2, 2),
            variables=[Variable(label="test_new", value=[2, 2])],
            independent_variable=Variable(label="ind", value=np.array([1, 2])),
        )
        variable_3 = Data(
            location=Geometry(0, 0, 1),
            variables=[Variable(label="test_new", value=[3, 3])],
            independent_variable=Variable(label="ind", value=np.array([1, 2])),
        )
        test_list = [variable_2, variable_3]
        inputs_ml = CreateInputsML()

        with pytest.raises(ValueError) as exc_info:
            results = inputs_ml.interpolate_on_independent_variable(
            test_list, variable_1, AggregateMethod.MIN, "test_new",bounds_error=True
        )
        assert (
            "Cannot extrapolate" in str(exc_info.value)
        )


    def test_interpolate_on_independent_variable(self):
        variable_1 = Data(
            location=Geometry(1, 1, 0),
            variables=[Variable(label="test", value=np.array([1, 2, 3, 4, 5]))],
            independent_variable=Variable(label="ind", value=np.array([1, 2, 3, 4, 5])),
        )
        variable_2 = Data(
            location=Geometry(2, 2, 2),
            variables=[Variable(label="test_new", value=[2, 2])],
            independent_variable=Variable(label="ind", value=np.array([1, 2])),
        )
        variable_3 = Data(
            location=Geometry(0, 0, 1),
            variables=[Variable(label="test_new", value=[3, 3])],
            independent_variable=Variable(label="ind", value=np.array([1, 2])),
        )
        test_list = [variable_2, variable_3]
        inputs_ml = CreateInputsML()
        results = inputs_ml.interpolate_on_independent_variable(
            test_list, variable_1, AggregateMethod.MIN, "test_new"
        )
        assert np.all(results.get_variable("test_new").value == [2, 2, 2, 2, 2])
        variable_1 = Data(
            location=Geometry(1, 1, 0),
            variables=[Variable(label="test", value=np.array([1, 2, 3, 4, 5]))],
            independent_variable=Variable(label="ind", value=np.array([1, 2, 3, 4, 5])),
        )
        results = inputs_ml.interpolate_on_independent_variable(
            test_list, variable_1, AggregateMethod.MAX, "test_new"
        )
        assert np.all(results.get_variable("test_new").value == [3, 3, 3, 3, 3])
        variable_1 = Data(
            location=Geometry(1, 1, 0),
            variables=[Variable(label="test", value=np.array([1, 2, 3, 4, 5]))],
            independent_variable=Variable(label="ind", value=np.array([1, 2, 3, 4, 5])),
        )
        results = inputs_ml.interpolate_on_independent_variable(
            test_list, variable_1, AggregateMethod.SUM, "test_new"
        )
        assert np.all(results.get_variable("test_new").value == [5, 5, 5, 5, 5])
        variable_1 = Data(
            location=Geometry(1, 1, 0),
            variables=[Variable(label="test", value=np.array([1, 2, 3, 4, 5]))],
            independent_variable=Variable(label="ind", value=np.array([1, 2, 3, 4, 5])),
        )
        results = inputs_ml.interpolate_on_independent_variable(
            test_list, variable_1, AggregateMethod.MEAN, "test_new"
        )
        assert np.all(
            results.get_variable("test_new").value == [2.5, 2.5, 2.5, 2.5, 2.5]
        )

    def test_aggregate_extracted_features(self):
        variable_1 = Data(
            location=Geometry(1, 1, 0), variables=[Variable(label="test", value=[1, 1])]
        )
        variable_2 = Data(
            location=Geometry(2, 2, 2), variables=[Variable(label="test", value=[2, 2])]
        )
        variable_3 = Data(
            location=Geometry(0, 0, 1), variables=[Variable(label="test", value=[3, 3])]
        )
        variable_4 = Data(
            location=Geometry(2, 0, 0), variables=[Variable(label="test", value=[4, 4])]
        )
        variable_5 = Data(
            location=Geometry(5, 5, 5), variables=[Variable(label="test", value=[5, 5])]
        )
        test_list = [variable_1, variable_2, variable_3, variable_4, variable_5]
        inputs_ml = CreateInputsML()
        assert (
            inputs_ml.aggregate_extracted_features(
                AggregateMethod.SUM, "test", test_list
            )
            == 30
        )
        assert (
            inputs_ml.aggregate_extracted_features(
                AggregateMethod.MIN, "test", test_list
            )
            == 1
        )
        assert (
            inputs_ml.aggregate_extracted_features(
                AggregateMethod.MAX, "test", test_list
            )
            == 5
        )
        assert (
            inputs_ml.aggregate_extracted_features(
                AggregateMethod.MEAN, "test", test_list
            )
            == 6
        )

    def test_get_k_closest_features(self):
        point_compare = Geometry(x=0, y=0, z=0)
        variable_1 = Data(
            location=Geometry(1, 1, 0), variables=[Variable(label="point_1", value=[1])]
        )
        variable_2 = Data(
            location=Geometry(2, 2, 2), variables=[Variable(label="point_2", value=[1])]
        )
        variable_3 = Data(
            location=Geometry(0, 0, 1), variables=[Variable(label="point_3", value=[1])]
        )
        variable_4 = Data(
            location=Geometry(2, 0, 0), variables=[Variable(label="point_4", value=[1])]
        )
        variable_5 = Data(
            location=Geometry(5, 5, 5), variables=[Variable(label="point_5", value=[1])]
        )
        inputs_ml = CreateInputsML()
        test_result = inputs_ml.get_k_closest_features(
            point_compare,
            [variable_1, variable_2, variable_3, variable_4, variable_5],
            3,
        )
        assert len(test_result) == 3
        assert test_result[0].variables[0].label == "point_3"
        assert test_result[1].variables[0].label == "point_1"
        assert test_result[2].variables[0].label == "point_4"

    def test_get_k_closest_features_error(self):
        point_compare = Geometry(x=0, y=0, z=0)
        variable_1 = Data(
            location=Geometry(1, 1, 0), variables=[Variable(label="point_1", value=[1])]
        )
        inputs_ml = CreateInputsML()
        with pytest.raises(ValueError) as exc_info:
            inputs_ml.get_k_closest_features(
                point_compare,
                [variable_1],
                3,
            )
        assert (
            str(exc_info.value)
            == "The number of points requested (3) is smaller than the number of points provided (1)."
        )
