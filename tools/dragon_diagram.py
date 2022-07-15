import imp
import inspect
import logging
import os
import sys
from functools import reduce
from itertools import starmap
from more_itertools import windowed
from math import acos, asin, ceil, copysign, floor, inf, isinf, nan, pi, sin, sqrt, cos, atan2, degrees
from pprint import pformat
import traceback
import shutil
import tempfile

import bpy
import numpy as np
from mathutils import Matrix, Vector

THIS_FILE = inspect.stack()[-2].filename
THIS_DIR = os.path.dirname(THIS_FILE)
try:
    PATH = sys.path[:]
    sys.path.insert(0, THIS_DIR)
    import common
    imp.reload(common)  # dumb hacks because of blender's pycache settings
    from common import (
        X_AXIS_3D, Y_AXIS_3D, Z_AXIS_3D, ENDLTAB, format_matrix,
        format_vector, TRI_VERTS, ATOL, ORIGIN_3D, X_AXIS_2D,
        setup_logger, mode_set, serialise_matrix, export_json,
        get_selected_polygons_suffix, sanitise_names, matrix_isclose,
        format_matrix_components, serialise_vector, format_quaternion,
        format_euler, format_vecs, format_angle, get_out_path
    )
    # from trig import gradient_cos, gradient_sin
finally:
    sys.path = PATH

# Defaults

LOG_FILE = os.path.splitext(os.path.basename(THIS_FILE))[0] + '.log'
DIAG_COLLECTION_NAME = 'DIAGS'
DEBUG_COLLECTION_NAME = 'DEBUG'


def main():
    setup_logger(LOG_FILE)

    logging.info(f"*** 🐉 Starting Dragon Diagram 🐉 ***")

    obj = bpy.context.object
    logging.info(f"Selected object: {obj.name}")
    logging.debug(f"Object World Matrix:" + ENDLTAB + format_matrix(obj.matrix_world))

    with mode_set('OBJECT'):
        if DEBUG_COLLECTION_NAME in bpy.data.collections:
            bpy.ops.object.delete({
                "selected_objects": bpy.data.collections[DEBUG_COLLECTION_NAME].all_objects})
            # debug_coll = bpy.data.collections[DEBUG_COLLECTION_NAME]
        # led_coll = bpy.data.collections[DIAG_COLLECTION_NAME]
        # debug_coll = None

    logging.info(f"*** 🐉 Completed Dragon Diagram 🐉 ***")


if __name__ == '__main__':
    main()
