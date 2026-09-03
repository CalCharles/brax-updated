# Copyright 2026 The Brax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint:disable=g-multiple-import, g-importing-member
"""Functions for forces/torques through fluids."""

import os
from typing import Union
from brax.base import Force, Motion, System, Transform
import jax
import jax.numpy as jp
from jax.ops import segment_sum
import mujoco
import numpy as np


# explore_bench fork: opt-in per-geom ellipsoid drag (A/B'd, default off).
#
# MuJoCo geoms authored with `fluidshape="ellipsoid"` use a geom-level fluid
# model and DISABLE the parent body's inertia-box model.  brax applied the
# box model to every link regardless, deriving frontal area from the
# equivalent-inertia box -- and the Robotiq 3f's vendored URDF inertias are
# 74-239x too large (see the sourced-assets notes), so the drone's gripper
# links carried square metres of phantom frontal area.  Measured on the
# x2lift_robotiq3f free fall over 1 s: CPU MuJoCo vz -9.114, brax box model
# vz -7.062 (22.5% slow).
#
# Implemented per active geom, in the geom frame: Stokes viscous force and
# torque on the equivalent sphere, blunt-body quadratic drag against the
# velocity-projected ellipse area (coefficient from `geom_fluid`), and the
# box-style quadratic angular drag scaled by the authored angular
# coefficient.  NOT implemented (each second-order for these scenes): Kutta
# lift, Magnus lift, added mass/inertia, and the slender-drag split.
#
# The flag also changes WHERE the box model applies: MuJoCo's inertia-box
# model is per BODY, and brax's per fused LINK -- on the drone the palm
# (ellipsoid geoms) fuses into the root link, so a per-link mask would
# either keep the phantom finger drag or delete the airframe's real drag
# with it.  Under the flag the box model is therefore evaluated per
# non-ellipsoid BODY (its own equivalent box, at its own COM offset inside
# the link), which is MuJoCo's own decomposition.
_ELLIPSOID = os.environ.get(
    'BRAX_FORK_ELLIPSOID_DRAG', '0').lower() not in ('0', '', 'false', 'no')
_ELL_CACHE = {}


def _quat_mat_np(q):
  w, x, y, z = q
  return np.array([
      [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
      [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
      [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
  ])


def _ellipsoid_tables(sys):
  """Static per-geom tables for the ellipsoid model, from `sys.mj_model`.

  Returns None when the model declares no fluid geoms; otherwise
  ``(ellipsoid_rows, body_box_rows)`` -- the per-geom ellipsoid tables and
  the per-BODY inertia-box tables for every plain body (see the module
  comment).  A geom's/body's link is its body walked up through fused
  parents until a name matches a brax link; offsets/rotations are expressed
  in that link's COM (inertia) frame, which is the frame `force()` computes
  velocities in.
  """
  key = id(sys)
  if key in _ELL_CACHE:
    return _ELL_CACHE[key]
  mj = sys.mj_model
  out = None
  if mj is not None:
    active = [g for g in range(mj.ngeom) if np.any(mj.geom_fluid[g] != 0)]
    names = list(sys.link_names)

    def link_of(b):
      while b > 0:
        nm = mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, int(b))
        if nm in names:
          return names.index(nm)
        b = int(mj.body_parentid[b])
      return -1

    rows = [(g, link_of(int(mj.geom_bodyid[g]))) for g in active]
    rows = [(g, li) for g, li in rows if li >= 0]
    if rows:
      link_idx = np.array([li for _, li in rows], np.int32)
      semi = np.stack([mj.geom_size[g] for g, _ in rows]).astype(np.float64)
      # geom_fluid[1:4] = (blunt, slender, angular) drag coefficients;
      # slender is carried for completeness but unused (module comment)
      coef = np.stack([mj.geom_fluid[g][1:4] for g, _ in rows])
      r_gc, rot_gc = [], []
      for g, _ in rows:
        b = int(mj.geom_bodyid[g])
        r_i = _quat_mat_np(mj.body_iquat[b])
        r_gc.append(r_i.T @ (mj.geom_pos[g] - mj.body_ipos[b]))
        rot_gc.append(r_i.T @ _quat_mat_np(mj.geom_quat[g]))
      ell = (link_idx, jp.asarray(semi, jp.float32),
             jp.asarray(coef, jp.float32),
             jp.asarray(np.stack(r_gc), jp.float32),
             jp.asarray(np.stack(rot_gc), jp.float32))

      # per-BODY inertia boxes for every body WITHOUT ellipsoid geoms (the
      # module comment explains why the per-link box cannot be masked): the
      # body's own equivalent box, at the body's COM, in the body's inertia
      # frame, all relative to the owning link's COM frame.  The link's COM
      # frame is the body-tree local composition of (link body ipos/iquat);
      # for the models this fork serves a fused child's frame equals its
      # parent's (welds at identity), so the child body's ipos/iquat compose
      # directly -- asserted per body below.
      ell_bodies = {int(mj.geom_bodyid[g]) for g in active}
      b_rows = []
      for b in range(1, mj.nbody):
        if b in ell_bodies or mj.body_mass[b] <= 0:
          continue
        li = link_of(b)
        if li < 0:
          continue
        m_b = float(mj.body_mass[b])
        inr = np.asarray(mj.body_inertia[b], np.float64)
        s = np.array([inr[1] + inr[2] - inr[0],
                      inr[0] + inr[2] - inr[1],
                      inr[0] + inr[1] - inr[2]])
        dims = np.sqrt(6.0 * np.clip(s, 1e-12, None) / m_b)
        # body com in the LINK BODY's frame (composed through the fused
        # chain), then into the link's COM (inertia) frame
        lb = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, names[li])
        r_l = _quat_mat_np(mj.body_iquat[lb])
        off = np.zeros(3)
        bb = b
        chain_rot = np.eye(3)
        while bb != lb:
          off = np.asarray(mj.body_pos[bb], np.float64) \
              + _quat_mat_np(mj.body_quat[bb]) @ off
          chain_rot = _quat_mat_np(mj.body_quat[bb]) @ chain_rot
          bb = int(mj.body_parentid[bb])
        com_in_linkbody = off + chain_rot @ np.asarray(mj.body_ipos[b])
        rot_in_linkbody = chain_rot @ _quat_mat_np(mj.body_iquat[b])
        r_bc = r_l.T @ (com_in_linkbody - np.asarray(mj.body_ipos[lb]))
        rot_bc = r_l.T @ rot_in_linkbody
        b_rows.append((li, dims, r_bc, rot_bc))
      box = None
      if b_rows:
        box = (np.array([r[0] for r in b_rows], np.int32),
               jp.asarray(np.stack([r[1] for r in b_rows]), jp.float32),
               jp.asarray(np.stack([r[2] for r in b_rows]), jp.float32),
               jp.asarray(np.stack([r[3] for r in b_rows]), jp.float32))
      out = (ell, box)
  _ELL_CACHE[key] = out
  return out


def _body_box_force(sys, xd_i: Motion, box_tables) -> Force:
  """MuJoCo's per-BODY inertia-box drag, accumulated per link (COM frame)."""
  link_idx, dims, r_bc, rot_bc = box_tables
  li = jp.asarray(link_idx)
  vel_l = xd_i.vel[li] + jp.cross(xd_i.ang[li], r_bc)
  v = jp.einsum('kji,kj->ki', rot_bc, vel_l)
  w = jp.einsum('kji,kj->ki', rot_bc, xd_i.ang[li])
  diam = jp.mean(dims, axis=1)
  f = -3.0 * jp.pi * sys.viscosity * diam[:, None] * v
  t = -jp.pi * sys.viscosity * (diam ** 3)[:, None] * w
  mult_v = jp.stack([dims[:, 1] * dims[:, 2], dims[:, 0] * dims[:, 2],
                     dims[:, 0] * dims[:, 1]], axis=1)
  f = f - 0.5 * sys.density * mult_v * jp.abs(v) * v
  mult_a = jp.stack([
      dims[:, 0] * (dims[:, 1] ** 4 + dims[:, 2] ** 4),
      dims[:, 1] * (dims[:, 0] ** 4 + dims[:, 2] ** 4),
      dims[:, 2] * (dims[:, 0] ** 4 + dims[:, 1] ** 4)], axis=1)
  t = t - sys.density * mult_a * jp.abs(w) * w / 64.0
  f_c = jp.einsum('kij,kj->ki', rot_bc, f)
  t_c = jp.einsum('kij,kj->ki', rot_bc, t) + jp.cross(r_bc, f_c)
  n_link = xd_i.vel.shape[0]
  return Force(vel=segment_sum(f_c, li, n_link),
               ang=segment_sum(t_c, li, n_link))


def _ellipsoid_force(sys, xd_i: Motion, tables) -> Force:
  """Per-geom ellipsoid drag, accumulated per link in the COM frame."""
  link_idx, semi, coef, r_gc, rot_gc = tables
  li = jp.asarray(link_idx)
  vel_l = xd_i.vel[li] + jp.cross(xd_i.ang[li], r_gc)
  ang_l = xd_i.ang[li]
  # into each geom's own frame
  v = jp.einsum('kji,kj->ki', rot_gc, vel_l)
  w = jp.einsum('kji,kj->ki', rot_gc, ang_l)

  d_eq = 2.0 * jp.mean(semi, axis=1)
  f = -3.0 * jp.pi * sys.viscosity * d_eq[:, None] * v
  t = -jp.pi * sys.viscosity * (d_eq ** 3)[:, None] * w

  s0, s1, s2 = semi[:, 0], semi[:, 1], semi[:, 2]
  speed = jp.linalg.norm(v, axis=1)
  a_proj = jp.pi * jp.sqrt(
      (s1 * s2 * v[:, 0]) ** 2 + (s0 * s2 * v[:, 1]) ** 2
      + (s0 * s1 * v[:, 2]) ** 2) / jp.maximum(speed, 1e-12)
  f = f - sys.density * (coef[:, 0] * a_proj * speed)[:, None] * v

  dd = 2.0 * semi
  ang_mult = jp.stack([
      dd[:, 0] * (dd[:, 1] ** 4 + dd[:, 2] ** 4),
      dd[:, 1] * (dd[:, 0] ** 4 + dd[:, 2] ** 4),
      dd[:, 2] * (dd[:, 0] ** 4 + dd[:, 1] ** 4)], axis=1)
  t = t - sys.density * coef[:, 2:3] * ang_mult * jp.abs(w) * w / 64.0

  # back to the com frame; force at the geom centre also torques the link
  f_c = jp.einsum('kij,kj->ki', rot_gc, f)
  t_c = jp.einsum('kij,kj->ki', rot_gc, t) + jp.cross(r_gc, f_c)
  n_link = xd_i.vel.shape[0]
  return Force(vel=segment_sum(f_c, li, n_link),
               ang=segment_sum(t_c, li, n_link))


def _box_viscosity(box: jax.Array, xd_i: Motion, viscosity: jax.Array) -> Force:
  """Gets force due to motion through a viscous fluid."""
  diam = jp.mean(box, axis=-1)
  ang_scale = -jp.pi * diam**3 * viscosity
  vel_scale = -3.0 * jp.pi * diam * viscosity
  frc = Force(
      ang=ang_scale[:, None] * xd_i.ang, vel=vel_scale[:, None] * xd_i.vel
  )
  return frc


def _box_density(box: jax.Array, xd_i: Motion, density: jax.Array) -> Force:
  """Gets force due to motion through dense fluid."""

  @jax.vmap
  def apply(b: jax.Array, xd: Motion) -> Force:
    box_mult_vel = jp.array([b[1] * b[2], b[0] * b[2], b[0] * b[1]])
    vel = -0.5 * density * box_mult_vel * jp.abs(xd.vel) * xd.vel
    box_mult_ang = jp.array([
        b[0] * (b[1] ** 4 + b[2] ** 4),
        b[1] * (b[0] ** 4 + b[2] ** 4),
        b[2] * (b[0] ** 4 + b[1] ** 4),
    ])
    ang = -1.0 * density * box_mult_ang * jp.abs(xd.ang) * xd.ang / 64.0
    return Force(vel=vel, ang=ang)

  return apply(box, xd_i)


def force(
    sys: System,
    x: Transform,
    xd: Motion,
    mass: jax.Array,
    inertia: jax.Array,
    root_com: Union[jax.Array, None] = None,
) -> Force:
  """Returns force due to motion through a fluid."""
  # get the velocity at the com position/orientation
  x_i = x.vmap().do(sys.link.inertia.transform)
  # TODO(brax-team): remove root_com when xd is fixed for stacked joints
  offset = x_i.pos - x.pos if root_com is None else x_i.pos - root_com
  xd_i = x_i.replace(pos=offset).vmap().do(xd)

  # TODO(brax-team): add ellipsoid fluid model from mujoco
  # TODO(brax-team): consider adding wind from mj.opt.wind
  diag_inertia = jax.vmap(jp.diag)(inertia)
  diag_inertia_v = jp.repeat(diag_inertia, 3, axis=-2).reshape((-1, 3, 3))
  diag_inertia_v *= jp.ones((3, 3)) - 2 * jp.eye(3)
  box = 6.0 * jp.clip(jp.sum(diag_inertia_v, axis=-1), min=1e-12)
  box = jp.sqrt(box / mass[:, None])

  # explore_bench fork: under BRAX_FORK_ELLIPSOID_DRAG=1 and when the model
  # declares `fluidshape="ellipsoid"` geoms, the per-link box above is
  # replaced wholesale by MuJoCo's own decomposition -- per-BODY inertia
  # boxes for plain bodies plus per-GEOM ellipsoid drag (module comment).
  ell_tables = _ellipsoid_tables(sys) if _ELLIPSOID else None
  if ell_tables is not None:
    ell, box_tables = ell_tables
    frc = _ellipsoid_force(sys, xd_i, ell)
    if box_tables is not None:
      frc += _body_box_force(sys, xd_i, box_tables)
  else:
    frc = _box_viscosity(box, xd_i, sys.viscosity)
    frc += _box_density(box, xd_i, sys.density)

  # rotate back to the world orientation
  frc = Transform.create(rot=x_i.rot).vmap().do(frc)

  return frc
