import csv
import imp
import inspect
import logging
import os
import sys
from functools import reduce
from itertools import starmap
from math import acos, asin, ceil, copysign, floor, inf, isinf, nan, pi, sin, sqrt, cos, atan2, degrees
from pprint import pformat
import numpy
from scipy.constants import speed_of_light
import traceback
import shutil
import tempfile

import bpy
import numpy as np
from mathutils import Matrix, Vector

try:
    THIS_FILE = inspect.stack()[-2].filename
except:
    THIS_FILE = __file__
THIS_NAME = os.path.splitext(os.path.basename(THIS_FILE))[0]
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
LOG_FILE = THIS_NAME + '.log'
DIAG_COLLECTION_NAME = 'DIAGS'
DEBUG_COLLECTION_NAME = 'DEBUG'

REPO_PATH = os.path.dirname(THIS_DIR)
DATA_PATH = os.path.join(REPO_PATH, 'data')
# DATA_PATH = 'S:\\hyperdrive\\results'


def main():
    setup_logger(LOG_FILE)

    logging.info(f"*** 🐉 Starting Dragon Diagram 🐉 ***")

    if DIAG_COLLECTION_NAME in bpy.data.collections:
        diags_collection = bpy.data.collections[DIAG_COLLECTION_NAME]
        bpy.ops.object.delete({
            "selected_objects": diags_collection.all_objects})
    else:
        diags_collection = bpy.data.collections.new(
            name=DIAG_COLLECTION_NAME)
    diags_collection = bpy.data.collections[DIAG_COLLECTION_NAME]
    if DIAG_COLLECTION_NAME not in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.link(diags_collection)

    offset = Vector((0, 0, 0))

    for name in [
        'MODEL-o_u',
        'MODEL-o_v',
        # 'MODEL-o_U',
    ]:
        path = os.path.join(DATA_PATH, f"{name}.csv")
        lambda_extrema, data = extract_data(path)

        for pol in [
            'xx',
            # 'xy',
            # 'yx',
            'yy'
        ]:
            make_diag(diags_collection, lambda_extrema,
                      data, name, pol, offset)
        offset.x += 1

    # bpy.ops.export_mesh.threemf(filepath=f"3d/{THIS_NAME}.3mf")
    bpy.ops.export_mesh.stl(filepath=f"3d/{THIS_NAME}.stl")

    logging.info(f"*** 🐉 Completed Dragon Diagram 🐉 ***")


def extract_data(path):
    lambda_extrema = [inf, -inf]
    data = []
    with open(path) as csvfile:
        dialect = csv.Sniffer().sniff(csvfile.read(1024))
        csvfile.seek(0)
        reader = csv.reader(csvfile, dialect)
        header = [key.strip()
                  for key in reader.__next__()]
        logging.debug(header)
        for row in reader:
            datum = dict(zip(header, [float(cell) for cell in row]))
            datum['lambda'] = speed_of_light / datum['freq_hz']
            if datum['lambda'] < lambda_extrema[0]:
                lambda_extrema[0] = datum['lambda']
            if datum['lambda'] > lambda_extrema[1]:
                lambda_extrema[1] = datum['lambda']
            data.append(datum)
    return lambda_extrema, data


def make_diag(diags_collection, lambda_extrema, data, name, pol: str, offset):
    assert len(data) > 0

    bpy.ops.curve.primitive_bezier_curve_add(enter_editmode=False)
    # bpy.ops.collection.objects_add_active(DIAG_COLLECTION_NAME)
    obj = bpy.context.object
    obj.name = f"{name}_{pol}"
    old_collection = obj.users_collection[0]
    diags_collection.objects.link(obj)
    old_collection.objects.unlink(obj)
    logging.info(f"Selected object: {obj.name} <{obj.type}>")
    logging.debug(f"Object World Matrix:" + ENDLTAB +
                  format_matrix(obj.matrix_world))

    with mode_set('EDIT'):
        bpy.ops.curve.select_all(action='SELECT')

        # bpy.ops.curve.subdivide(number_cuts=len(data))
        bpy.ops.curve.delete(type='VERT')
        curve = bpy.context.active_object
        curve.data.bevel_depth = 0.05

        for i, datum in enumerate(data):
            lambda_rel = numpy.interp(
                datum['lambda'], [0, lambda_extrema[1]], [0, 1])

            norm, arg = datum[f'{pol}_norm'], datum[f'{pol}_arg']
            logging.debug(f'lambda_rel {lambda_rel} norm {norm}, arg {arg}')

            location = Vector((
                offset.x + norm * cos(arg),
                offset.y + norm * sin(arg),
                offset.z + lambda_rel
            ))

            if pol.startswith('y'):
                location.z *= -1

            logging.debug(f'location {location}')

            bpy.ops.curve.vertex_add(location=(*location,))


if __name__ == '__main__':
    main()
