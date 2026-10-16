"""Earlier file name for the DIII-D calibration of osf_0.0.12.

The calibration is d3d_0.0.1_osf_0.0.12. This module forwards to it.
"""
from osf.d3d_0_0_1_osf_0_0_12 import CAL, MODEL_ID, SOLVER_ID, solve_one_frame, solve_shot, write_shot

__all__ = ["CAL", "MODEL_ID", "SOLVER_ID", "solve_one_frame", "solve_shot", "write_shot"]
