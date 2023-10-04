def soil_type_from_IC(ic: float):
    """
    Classifies IC based on figure 22 of Robertson, 2010, page 27
    """
    if ic < 1.31:
        return {
            "name": "dense sand",
            "code": 7,
            "soil_weight_parameters": {
                "saturated_weight": {"mean": 22},
                "unsaturated_weight": {"mean": 17},
            },
            "mohr_coulomb_parameters": {
                "cohesion": {"mean": 1},
                "friction_angle": {"mean": 32.5},
            },
        }
    elif ic >= 1.31 and ic < 2.05:
        return {
            "name": "silty sand",
            "code": 6,
            "soil_weight_parameters": {
                "saturated_weight": {"mean": 22},
                "unsaturated_weight": {"mean": 17},
            },
            "mohr_coulomb_parameters": {
                "cohesion": {"mean": 1},
                "friction_angle": {"mean": 30},
            },
        }
    elif ic >= 2.05 and ic < 2.6:
        return {
            "name": "sandy silt",
            "code": 5,
            "soil_weight_parameters": {
                "saturated_weight": {"mean": 21},
                "unsaturated_weight": {"mean": 19},
            },
            "mohr_coulomb_parameters": {
                "cohesion": {"mean": 1},
                "friction_angle": {"mean": 29},
            },
        }
    elif ic >= 2.6 and ic < 2.95:
        return {
            "name": "silty clay",
            "code": 4,
            "soil_weight_parameters": {
                "saturated_weight": {"mean": 20},
                "unsaturated_weight": {"mean": 19},
            },
            "mohr_coulomb_parameters": {
                "cohesion": {"mean": 1},
                "friction_angle": {"mean": 27},
            },
        }
    elif ic >= 2.95 and ic < 3.6:
        return {
            "name": "silty clay",
            "code": 3,
            "soil_weight_parameters": {
                "saturated_weight": {"mean": 18},
                "unsaturated_weight": {"mean": 18},
            },
            "mohr_coulomb_parameters": {
                "cohesion": {"mean": 1},
                "friction_angle": {"mean": 27},
            },
        }
    else:
        return {
            "name": "Organic soil",
            "code": 2,
            "soil_weight_parameters": {
                "saturated_weight": {"mean": 13},
                "unsaturated_weight": {"mean": 13},
            },
            "mohr_coulomb_parameters": {
                "cohesion": {"mean": 1},
                "friction_angle": {"mean": 15},
            },
        }
