from pathlib import Path
from typing import List

import geolib
import numpy as np


class DStabilityModel:
    """Class that creates a D-Stability model."""

    @staticmethod
    def define_default_soil():
        """Function that adds a default soil layer to the geolib model."""
        soil = geolib.soils.Soil()
        soil.name = "Soil test"
        soil.code = "default_soil"
        soil.soil_weight_parameters.saturated_weight.mean = 10.2
        soil.soil_weight_parameters.unsaturated_weight.mean = 10.2
        return soil

    @staticmethod
    def create_model(polygon_list: List, filename: str, soil_list: List = None):
        """Function that creates and saves a D-Stability model based on a polygon list.

        :param polygon_list: List of shapely polygons
        :param filename: Name of D-Stability file that is serialized. 
        :param polygon_list: List of shapely polygons
        :param filename: Name of D-Stability file that is serialized

        :returns: A geolib D-Stability model

        """
        # check types
        if not (
            all(
                [
                    "shapely.geometry.polygon.Polygon" in str(type(polygon))
                    for polygon in polygon_list
                ]
            )
        ):
            raise ValueError(
                "One or more elements of the list are not of type shapely.geometry.polygon.Polygon."
            )
        # create model
        model = geolib.models.DStabilityModel()
        # define default calculation method
        bishop_analysis_method = (
            geolib.models.dstability.analysis.DStabilityBishopAnalysisMethod(
                circle=geolib.models.dstability.analysis.DStabilityCircle(
                    center=geolib.geometry.Point(x=20, z=3), radius=15
                )
            )
        )
        model.set_model(bishop_analysis_method)
        if soil_list == None:
            soil = DStabilityModel.define_default_soil()
            soil_id = model.add_soil(soil)
        # add layers
        layers_and_soils = []
        for polygon_counter, polygon in enumerate(polygon_list):
            # create points
            layer_points = []
            # extract exterior of polygon
            exterior = np.array(polygon.exterior.xy).T
            # points are rounded of to the third decimal
            exterior = [(round(coord[0], 3), round(coord[1], 3)) for coord in exterior]
            # remove duplicates
            exterior_no_duplicates = []
            for coord in exterior:
                if coord not in exterior_no_duplicates:
                    exterior_no_duplicates.append(coord)
            # create d-stability layer
            for counter_point, coords in enumerate(exterior_no_duplicates):
                layer_points.append(
                    geolib.geometry.Point(
                        x=round(coords[0], 3),
                        z=round(coords[1], 3),
                        label=f"layer_{polygon_counter}_{counter_point}",
                    )
                )
            if soil_list == None:
                layers_and_soils.append(
                    (layer_points, "default_soil", f"layer_{polygon_counter}")
                )
            else:
                soil = geolib.soils.Soil(**soil_list[polygon_counter])
                if not (
                    soil.code
                    in [soil_applied.Code for soil_applied in model.soils.Soils]
                ):
                    soil_id = model.add_soil(soil)
                layers_and_soils.append(
                    (layer_points, soil.code, f"layer_{polygon_counter}")
                )
        # add layers to model
        layer_ids = []
        for layer, soil, layer_label in layers_and_soils:
            layer_id = model.add_layer(layer, soil, label=layer_label)
            layer_ids.append(layer_id)
        model.serialize(Path(filename))
        return model
