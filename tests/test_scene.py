"""Scene tests: the properties checked by hand while building sim/scenes/dinner_table.xml.

Each test names the bug or decision it guards (docs/debug-log.md, docs/design-decisions.md).
`scene` is the scene file alone (no items, D17). Checks that depend on what is attached
per episode run for the scene alone and for each task's build. Checks specific to one
task are in tests/test_task_<task>.py.
Run:  uv run pytest            (all)
      uv run pytest -m "not slow"   (skip the IK reach searches)
"""

import importlib.util
import os
import time
from pathlib import Path

import mujoco
import numpy as np
import pytest

from sim.ik import ArmIK
from sim.waste_env import ITEMS_DIR, TASKS, build_model, resting_height
from reach import CAN_CENTRE, TIP, reachable

# Override to test a modified copy (must sit in sim/scenes/ so relative asset paths resolve).
SCENE = Path(os.environ.get("SCENE_XML", "sim/scenes/dinner_table.xml"))
TABLE_TOP = 0.315
BINS = ["plastic", "metal"]
BIN_OPEN = 0.18
ALL_ITEMS = sorted(p.parent.name.split("_", 1)[1] for p in ITEMS_DIR.glob("*/model.xml"))
# what the environment builds: the scene alone, and the scene with each task's items
BUILDS = {"scene": (), **{t: tuple(task.items) for t, task in TASKS.items()}}


@pytest.fixture(scope="module")
def scene():
    return mujoco.MjModel.from_xml_path(str(SCENE))


@pytest.fixture(scope="module", params=sorted(BUILDS))
def built(request):
    """The model the environment compiles for one entry of BUILDS."""
    return build_model(BUILDS[request.param], SCENE)


def settle(m, key, seconds):
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key(key).id)
    for _ in range(int(seconds / m.opt.timestep)):
        mujoco.mj_step(m, d)
    return d


def arm_qadr(m):
    return np.array([m.jnt_qposadr[m.actuator_trnid[i, 0]] for i in range(m.nu)])


def contacts_with(m, d, prefixes):
    """Body-name pairs of contacts where either body name starts with one of `prefixes`."""
    pairs = set()
    for c in d.contact[: d.ncon]:
        a, b = m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name
        if a.startswith(prefixes) or b.startswith(prefixes):
            pairs.add((a, b))
    return sorted(pairs)


def free_joints(m):
    return [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]


# --- structure -----------------------------------------------------------------

def test_counts_and_names(scene):
    m = scene
    assert m.nu == 12
    assert [m.actuator(i).name for i in range(m.nu)] == [
        f"{s}_{j}" for s in ("left", "right")
        for j in ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")]
    assert [m.camera(i).name for i in range(m.ncam)] == ["overview", "left_wrist", "right_wrist"]
    assert [m.key(i).name for i in range(m.nkey)] == ["home", "bins_open"]
    # state layout: bins first, then arms; no items in the scene file (D17)
    assert [m.joint(j).name for j in range(2)] == ["plastic_bin_slide", "metal_bin_slide"]
    assert free_joints(m) == []
    assert m.nq == 2 + 12


def test_physics_options_declared_in_scene(built):
    """An attached model's <option> is not inherited; the scene must set its own, and attaching
    items must not change it."""
    o = built.opt
    assert o.timestep == pytest.approx(0.002)
    assert o.integrator == mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    assert o.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    assert o.impratio == pytest.approx(10)
    assert o.solver == mujoco.mjtSolver.mjSOL_CG


def test_items_have_ycb_masses():
    """Every imported item's mass equals the YCB Table II value in import_ycb.py."""
    spec = importlib.util.spec_from_file_location("import_ycb", "scripts/import_ycb.py")
    imp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(imp)
    for oid, (mass, _, _) in imp.ITEMS.items():
        path = ITEMS_DIR / oid / "model.xml"
        if not path.exists():
            continue  # e.g. the wine glass: no scan on the server
        m = mujoco.MjModel.from_xml_path(str(path))
        assert m.body_mass[m.body(oid.split("_", 1)[1]).id] == pytest.approx(mass, abs=1e-4), oid


@pytest.mark.parametrize("item", ALL_ITEMS)
def test_each_item_stands_on_the_table(item):
    """Built into the scene on its own and set on the table, each item settles without sinking."""
    m = build_model((item,), SCENE)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    a = m.jnt_qposadr[m.joint(f"ycb_{item}_free").id]
    d.qpos[a:a + 7] = [0.0, 0.10, TABLE_TOP + resting_height(m, item) + 0.002, 1, 0, 0, 0]
    for _ in range(int(1.0 / m.opt.timestep)):
        mujoco.mj_step(m, d)
    b = m.body(f"ycb_{item}").id
    assert np.isfinite(d.qpos).all()
    assert d.xpos[b][2] > TABLE_TOP
    assert np.hypot(*(d.xpos[b][:2] - [0.0, 0.10])) < 0.05, "slid or rolled more than 5 cm"


def test_assets_have_license_and_provenance():
    assert Path("sim/assets/so101/LICENSE").exists() and Path("sim/assets/so101/SOURCE.md").exists()
    assert Path("sim/assets/objects/ycb/LICENSE").exists()
    for folder in Path("sim/assets/objects/ycb").iterdir():
        if folder.is_dir():
            assert (folder / "SOURCE.md").exists(), folder


# --- keyframes ---------------------------------------------------------------------

def test_keyframes_hold_only_bins_and_arms(scene):
    """With items attached per episode (D17), keys must not list items: a long key is a load error,
    and a short one pads free joints with the world origin (debug-log #6, #7)."""
    assert scene.key_qpos.shape[1] == scene.nq == 14


def test_keyframe_ctrl_matches_arm_qpos(scene):
    """Servo targets must equal the arm pose, or the arms move on reset."""
    m, adr = scene, arm_qadr(scene)
    for k in range(m.nkey):
        np.testing.assert_allclose(m.key_ctrl[k], m.key_qpos[k, adr], atol=1e-6)


def test_bins_open_key(scene):
    m = scene
    for b in BINS:
        a = m.jnt_qposadr[m.joint(f"{b}_bin_slide").id]
        assert m.key_qpos[m.key("home").id, a] == 0
        assert m.key_qpos[m.key("bins_open").id, a] == pytest.approx(BIN_OPEN)


# --- physics at rest -----------------------------------------------------------------

@pytest.mark.parametrize("key", ["home", "bins_open"])
def test_arms_hold_without_contacts(scene, key):
    d = settle(scene, key, 3.0)
    assert np.isfinite(d.qpos).all()
    assert np.abs(d.qpos[arm_qadr(scene)] - d.ctrl).max() < 1e-3
    assert contacts_with(scene, d, ("left_", "right_")) == []


def test_bins_stay_closed_at_rest(scene):
    d = settle(scene, "home", 3.0)
    for b in BINS:
        assert abs(d.qpos[scene.jnt_qposadr[scene.joint(f"{b}_bin_slide").id]]) < 1e-3


@pytest.mark.parametrize("b", BINS)
def test_bin_opens_under_small_pull_and_stops(scene, b):
    m = scene
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    body = m.body(f"{b}_bin").id
    for _ in range(int(3.0 / m.opt.timestep)):
        d.xfrc_applied[body, :3] = [0, -0.5, 0]
        mujoco.mj_step(m, d)
    assert d.qpos[m.jnt_qposadr[m.joint(f"{b}_bin_slide").id]] == pytest.approx(BIN_OPEN, abs=2e-3)
    other = "metal" if b == "plastic" else "plastic"
    assert abs(d.qpos[m.jnt_qposadr[m.joint(f"{other}_bin_slide").id]]) < 1e-3


def test_arm_bases_rest_on_the_tabletop(scene):
    """Bases and table are both welded to the world, so MuJoCo never reports contacts between them;
    check the geometry instead (the base mesh once sank 2.4 mm into the table)."""
    m = scene
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    for side in ("left", "right"):
        b = m.body(f"{side}_base").id
        lowest = np.inf
        for g in range(m.ngeom):
            if m.geom_bodyid[g] == b and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]
                v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
                lowest = min(lowest, (v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g])[:, 2].min())
        assert lowest == pytest.approx(TABLE_TOP, abs=1e-3), f"{side} base bottom at {lowest:.4f}"


# --- rendering -------------------------------------------------------------------------

def test_near_plane_closer_than_the_jaws(built):
    """Items grow the model extent, and the near plane scales with it; it once clipped the jaws
    (5 cm vs 3.5 cm)."""
    m = built
    near = m.vis.map.znear * m.stat.extent
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    mujoco.mj_forward(m, d)
    for side in ("left", "right"):
        cam = d.cam_xpos[m.camera(f"{side}_wrist").id]
        closest = np.inf
        for g in range(m.ngeom):
            if m.geom_group[g] == 2 and m.body(m.geom_bodyid[g]).name.startswith(f"{side}_") \
                    and m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]
                v = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
                v = v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
                closest = min(closest, np.linalg.norm(v - cam, axis=1).min())
        assert near < closest, f"{side}: near plane {near:.3f} m, closest jaw vertex {closest:.3f} m"


def test_every_camera_renders_something(scene):
    d = settle(scene, "home", 0.2)
    r = mujoco.Renderer(scene, 120, 160)
    for i in range(scene.ncam):
        r.update_scene(d, camera=i)
        img = r.render()
        assert img.std() > 5, scene.camera(i).name  # not a blank frame
    r.close()


def test_faster_than_real_time(scene):
    d = mujoco.MjData(scene)
    mujoco.mj_resetDataKeyframe(scene, d, scene.key("home").id)
    n = 500
    t = time.perf_counter()
    for _ in range(n):
        mujoco.mj_step(scene, d)
    assert n * scene.opt.timestep / (time.perf_counter() - t) > 2.0  # measured 10.8x on an M-series Mac


# --- reach: both arms can open and drop into both bins (D10, D11) ---

@pytest.mark.slow
@pytest.mark.parametrize("side, b", [("right", "metal"), ("left", "plastic")])
def test_arm_opens_its_bin(scene, side, b):
    """Fingertip into the gap behind the handle, then follow it 0.18 m (handle must stand 5 cm off)."""
    m = scene
    ik = ArmIK(m, side)
    home = m.key_qpos[m.key("home").id].copy()
    slide = m.jnt_qposadr[m.joint(f"{b}_bin_slide").id]
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
    mujoco.mj_forward(m, d)
    hook = d.site(f"{b}_bin_hook").xpos.copy()
    prev = None
    for s, dz in [(0.0, 0.055), (0.0, 0.0), (0.06, 0.0), (0.12, 0.0), (BIN_OPEN, 0.0)]:
        base = home.copy()
        base[slide] = s
        prev = reachable(ik, base, hook + np.array([0, -s, dz]), TIP, prev=prev)
        assert prev is not None, f"{side} arm loses the {b} handle at slide {s}"


@pytest.mark.slow
@pytest.mark.parametrize("side, b", [("left", "metal"), ("left", "plastic"), ("right", "metal"), ("right", "plastic")])
def test_arm_can_drop_into_open_bin(scene, side, b):
    """Held can's centre over the open bin with the gripper at most 75 deg from pointing down.
    At bins x = +-0.20, arm A reached the metal bin only upside down (172 deg)."""
    m = scene
    openq = m.key_qpos[m.key("bins_open").id].copy()
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, m.key("bins_open").id)
    mujoco.mj_forward(m, d)
    x = d.site(f"{b}_bin_interior").xpos[0]
    assert reachable(ArmIK(m, side), openq, np.array([x, -0.38, 0.36]), CAN_CENTRE, max_tilt=75) is not None
