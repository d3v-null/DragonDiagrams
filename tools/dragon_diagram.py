import cmath
from collections import OrderedDict
from copy import copy
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
from astropy.io import fits
from tabulate import tabulate

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
DATA_PATH = 'S:\\hyperdrive\\dragon'

UVFITS_POL_NAMES = ["xx", "yy", "xy", "yx"]


def decode_uvfits_baseline(bl: int):
    if bl < 65_535:
        ant2 = bl % 256
        ant1 = (bl - ant2) // 256
    else:
        ant2 = (bl - 65_536) % 2048
        ant1 = (bl - ant2 - 65_536) // 2048
    return (ant1, ant2)


def get_freqs_hz(hdus):
    row_0_shape = hdus[0].data[0].data.shape
    num_chans = row_0_shape[2]
    freq_delta = int(hdus[0].header['CDELT4'])
    freq_centre = int(hdus[0].header['CRVAL4'])
    freq_start = int(freq_centre -
                     (freq_delta * (hdus[0].header['CRPIX4'] - 1)))
    freq_end = num_chans * freq_delta

    print(
        f"num_chans: {num_chans} from {freq_start} to {freq_end} step {freq_delta}")

    return range(freq_start, freq_start +
                 num_chans * freq_delta, freq_delta)


def main():
    setup_logger(LOG_FILE)

    logging.info(f"*** 🐉 Starting Dragon Diagram 🐉 ***")

    if DIAG_COLLECTION_NAME in bpy.data.collections:
        diags_coll = bpy.data.collections[DIAG_COLLECTION_NAME]
        bpy.ops.object.delete({
            "selected_objects": diags_coll.all_objects})
    else:
        diags_coll = bpy.data.collections.new(
            name=DIAG_COLLECTION_NAME)
    if DIAG_COLLECTION_NAME not in bpy.context.scene.collection.children:
        bpy.context.scene.collection.children.link(diags_coll)

    offset = Vector((0, 0, 0))
    sel_bls = []
    # sel_bls = ["o-v", "o-w", "o-d"]
    # sel_pols = []
    sel_pols = [
        "xx",
        # "yy"
    ]

    path = os.path.join(DATA_PATH, "vis_synth.uvfits")
    # path = os.path.join(DATA_PATH, "spike_vis_synth_model.uvfits")
    with fits.open(path) as hdus:
        hdus.info()
        print(repr(hdus[0].header))
        print(tabulate(hdus[1].data, headers=hdus[1].data.names))

        print(f"hdus[0].data[0].data.shape: {hdus[0].data[0].data.shape}")
        freqs_hz = get_freqs_hz(hdus)
        num_chans = len(freqs_hz)

        # lambda extrema: [min, max]
        l_ext = [None, None]
        lambdas = []
        for freq in freqs_hz:
            lambda_ = speed_of_light / freq
            if l_ext[0] is None or lambda_ < l_ext[0]:
                l_ext[0] = lambda_
            if l_ext[1] is None or lambda_ > l_ext[1]:
                l_ext[1] = lambda_
            lambdas.append(lambda_)

        # which times have been seen
        ts_seen = []
        # which baselines have been seen
        bls_seen = []

        ds = []
        for row in hdus[0].data:
            r = {}
            t_mjd = float(row['DATE'])
            if t_mjd not in ts_seen:
                ts_seen.append(t_mjd)
            r['ts_idx'] = ts_seen.index(t_mjd)

            # TODO: multiple timesteps
            if r['ts_idx'] > 0:
                break

            uv_bl = int(row['BASELINE'])
            ant1_idx, ant2_idx = decode_uvfits_baseline(uv_bl)
            annames = hdus[1].data['ANNAME'][[ant1_idx - 1, ant2_idx - 1]]
            r['ant1'], r['ant2'] = annames
            bl_name = "-".join(annames)

            # filter selected baselines
            if sel_bls and bl_name not in sel_bls:
                continue

            if bl_name not in bls_seen:
                bls_seen.append(bl_name)
            bl_idx = bls_seen.index(bl_name)
            # TODO: moar bls
            if bl_idx > 31:
                break

            uvw = Vector((
                row['UU'],
                row['VV'],
                row['WW'],
            )) * speed_of_light
            u, v, w = uvw
            r['u'] = f"{u:+5.2f}"
            r['v'] = f"{v:+5.2f}"
            r['w'] = f"{w:+5.2f}"
            for pol_idx, pol_name in enumerate(UVFITS_POL_NAMES):
                if sel_pols and pol_name not in sel_pols:
                    continue
                pol_real = row.data[:, :, :,
                                    pol_idx, 0].reshape((num_chans, ))
                pol_imag = row.data[:, :, :,
                                    pol_idx, 1].reshape((num_chans, ))
                p = copy(r)
                pol_vis = pol_real - 1j * pol_imag

                mag_args = []
                for vis, l in zip(pol_vis, lambdas):
                    f = copy(p)
                    l = speed_of_light / freq
                    f['lambda'] = f"{l:+5.3f}"
                    mag, arg = abs(vis), cmath.phase(vis)
                    f['mag'] = f"{mag:+5.2f}"
                    f['phase'] = f"{arg:+5.2f}"
                    ds.append(f)
                    mag_args.append((mag, arg))

                make_diag(
                    mag_args,
                    lambdas,
                    "-".join(annames),
                    pol_name,
                    uvw,
                    l_ext,
                )

        print(tabulate([d.values() for d in ds[:100]], headers=f.keys()))
        offset.x += 1

    logging.info(f"*** 🐉 Completed Dragon Diagram 🐉 ***")


# make a dragon diagram in the active collection and export
def make_diag(mag_args, lambdas, name, pol: str, uvw, l_ext):
    print(f"name={name}, pol={pol}, uvw="+format_vector(uvw))

    # height of axes
    lambda_ax = 3
    mag_ax = 1
    phi_ax = 1
    # tick length
    tick_ax = 0.05

    # # magnitude extrema
    # m_ext = [None, None]
    # mag_args = []
    # for vis in data:
    #     mag = abs(vis)
    #     arg = cmath.phase(vis)
    #     if m_ext[0] is None or mag < m_ext[0]:
    #         m_ext[0] = mag
    #     if m_ext[1] is None or mag > m_ext[1]:
    #         m_ext[1] = mag
    #     mag_args.append((mag, arg))

    u, v, w = uvw

    points = []
    for (m, a), l in zip(mag_args, lambdas):
        # lambda normalized
        l_norm = numpy.interp(
            l, l_ext, [0, lambda_ax])

        # magnitude normalized
        m_norm = m
        # m_norm = numpy.interp(
        #     m, m_ext, [0, mag_ax]
        # )

        location = Vector((
            u + m_norm * cos(a),
            v + m_norm * sin(a),
            w + l_norm
        ))
        if pol.startswith('y'):
            location.z *= -1
        # logging.debug(
        #     f'(x{x:+5.3}, y{y:+5.3}, z{z:+5.3}) <- l{l_norm:+5.3}, m{m_norm:+8.3}, a{a:+5.3}, (u{u:+5.3}, v{v:+5.3}, w{w:+5.3})')
        points.append(location)

    # ######### #
    # DRAW AXES #
    # ######### #

    # -> axis curve data block
    ax_curve = bpy.data.curves.new(f"axis_curve_{name}_{pol}", type='CURVE')
    ax_curve.dimensions = '3D'
    ax_curve.resolution_u = 2

    # -> axis splines
    # --> lambda axis
    ax_spline_l = ax_curve.splines.new('POLY')
    ax_spline_l.points.add(2 - len(ax_spline_l.points))
    ax_spline_l.points[0].co = (u, v, w-tick_ax, 1)
    ax_spline_l.points[1].co = (u, v, w+lambda_ax, 1)

    # --> magnitude axis
    ax_spline_m = ax_curve.splines.new('POLY')
    ax_spline_m.points.add(2 - len(ax_spline_m.points))
    ax_spline_m.points[0].co = (u-tick_ax, v, w, 1)
    ax_spline_m.points[1].co = (u+mag_ax, v, w, 1)

    # --> phi axis
    circle_subdivs = 16
    ax_spline_p = ax_curve.splines.new('BEZIER')
    ax_spline_p.bezier_points.add(circle_subdivs - len(ax_spline_p.points) - 1)
    for i, phi in enumerate(numpy.linspace(0., pi, circle_subdivs)):
        ax_spline_p.bezier_points[i].handle_left_type = 'AUTO'
        ax_spline_p.bezier_points[i].handle_right_type = 'AUTO'
        ax_spline_p.bezier_points[i].co = (
            u + phi_ax * cos(phi),
            v + phi_ax * sin(phi),
            w,
        )

    # --> strands
    # gap between strands (z)
    strand_gap = 0.05
    last_z = None
    for (x, y, z) in points:
        if last_z is None or abs(z - last_z) > strand_gap:
            last_z = z
            strand_spline = ax_curve.splines.new('POLY')
            strand_spline.points.add(2 - len(strand_spline.points))
            strand_spline.points[0].co = (u, v, z, 1)
            strand_spline.points[1].co = (x, y, z, 1)

    # -> axis object
    ax_name = f"axis_{name}_{pol}"
    ax_obj = bpy.data.objects.new(ax_name, ax_curve)
    ax_obj.data.bevel_depth = 0.01

    # attach to scene and validate contex
    # scn = bpy.context.scene
    # scn.view_layers[0].objects.selected.link(ax_obj)
    view_layer = bpy.context.view_layer
    view_layer.active_layer_collection.collection.objects.link(ax_obj)
    ax_obj.select_set(False)

    # ########### #
    # DRAW DRAGON #
    # ########### #

    # -> curve data block
    dragon_curve = bpy.data.curves.new(
        f"dragon_curve_{name}_{pol}", type='CURVE')
    dragon_curve.dimensions = '3D'
    # dragon_curve.resolution_u = 2

    # -> spline
    dragon_spline = dragon_curve.splines.new('BEZIER')
    dragon_spline.bezier_points.add(
        len(points) - len(dragon_spline.points) - 1)
    for i, (x, y, z) in enumerate(points):
        dragon_spline.bezier_points[i].handle_left_type = 'AUTO'
        dragon_spline.bezier_points[i].handle_right_type = 'AUTO'
        dragon_spline.bezier_points[i].co = (x, y, z)

    dragon_name = f"dragon_{name}_{pol}"
    dragon_obj = bpy.data.objects.new(dragon_name, dragon_curve)
    dragon_obj.data.bevel_depth = 0.05
    # scn.view_layers[0].objects.selected.link(dragon_obj)
    view_layer.active_layer_collection.collection.objects.link(dragon_obj)
    dragon_obj.select_set(False)

    # ###### #
    # EXPORT #
    # ###### #

    # bpy.data.objects[ax_name].select_set(True)
    override = bpy.context.copy()
    override["selected_objects"] = [ax_obj, dragon_obj]
    with bpy.context.temp_override(**override):
        bpy.ops.export_mesh.stl(
            filepath=f"3d/{name}_{pol}.stl",
            use_selection=True
        )


if __name__ == '__main__':
    main()
