import pytest
import pickle
from pathlib import Path

from d_series_parser.d_stability_parser import DStabilityModel
from utils import TestUtils
from test_models.soil_type_from_IC import soil_type_from_IC


class TestDStabilityModel:
    @pytest.mark.intergrationtest
    def test_create_model_d_stability(self):
        # import pikle file with all polygons
        input_files = str(
            TestUtils.get_test_files_from_local_test_dir(
                "", "polygons_d_stability.pkl"
            )[0]
        )
        with open(input_files, "rb") as f:
            pickle_dict = pickle.load(f)
            polygons = pickle_dict["polygons"]
            values = pickle_dict["values"]
        soils_dictionary = [soil_type_from_IC(ic_value * 3.6) for ic_value in values]

        # set a stix file name
        filename = "test_model.stix"
        # create a default model
        model = DStabilityModel.create_model(polygons, filename, soils_dictionary)
        # check expectations
        assert "geolib.models.dstability.dstability_model.DStabilityModel" in str(
            type(model)
        )
        assert Path(filename).exists()
