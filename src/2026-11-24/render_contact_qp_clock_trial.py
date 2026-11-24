"""Render human motion and measured robot state on the contact-QP sim clock."""
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("LP_NUM_THREADS", "2")
import cv2
import mujoco
import numpy as np

ROOT = Path("/mnt/d/Github/PhaseMend")
WORKSPACE = ROOT / "workspace"
RUNS = ROOT / "sonic_takeover_wsl/takeover/runs"
OUT = WORKSPACE / (sys.argv[1] if len(sys.argv) > 1 else
                   "wholebody_sonic_trial_20261122_scale0100_slow9x_rate50hz_humanref_swinggain200_simclock_20261123_contactqpgait")
sys.path[:0] = [str(WORKSPACE), "/home/wangrunqi/sonic_takeover_share_20261114/human_motion/scripts"]
from retarget import SCENE  # noqa: E402

def read_rows(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def nearest_index(times, value):
    i = int(np.searchsorted(times, value))
    if i <= 0:
        return 0
    if i >= len(times):
        return len(times) - 1
    return i - 1 if value - times[i - 1] <= times[i] - value else i

exp = json.loads((OUT / "experiment.json").read_text(encoding="utf-8"))
review = json.loads((OUT / "review.json").read_text(encoding="utf-8"))
pre_dir = Path(exp["source_contract"]["preflight_directory"])
pre = json.loads((pre_dir / "preflight.json").read_text(encoding="utf-8"))
run = Path(exp["run"])
qp = read_rows(run / "contact_qp_gait.jsonl")
sim = read_rows(run / "sim.jsonl")
qp = [r for r in qp if r.get("source_clock_mode") == "mujoco_sim_time_with_transfer_pause"]
qp_t = np.asarray([float(r["sim_time"]) for r in qp])
qp_elapsed = np.asarray([float(r["source_elapsed_s"]) for r in qp])
motion_start = next(float(r["motion_start_sim_time"]) for r in qp if r.get("motion_start_sim_time") is not None)
source_duration = float(exp["preflight_limits"]["duration_s"])
motion_end = next(float(r["sim_time"]) for r in reversed(qp) if float(r["source_elapsed_s"]) >= source_duration - 1e-6)
duration = motion_end - motion_start
sim_t = np.asarray([float(r["sim_time"]) for r in sim])
sim_qp = np.asarray([r["qpos"] for r in sim], dtype=float)
sim_qv = np.asarray([r["qvel"] for r in sim], dtype=float)

record = WORKSPACE / "human_input_candidates/cmu13_30/human_skeleton_candidate.npz"
with np.load(record, allow_pickle=False) as z:
    names = list(z["joint_names"])
    human_p = z["p_world_m"]
    human_t = np.asarray(z["t"], dtype=float)
    edges = z["edges"]
    floor = float(np.min(human_p[:, [names.index("lfoot"), names.index("rfoot")], 2]))
start_source_t = float(pre["rows"][0]["source_t"])
anchor_idx = int(np.argmin(np.abs(human_t - start_source_t)))
anchor = human_p[anchor_idx, 0].copy()
slowdown = float(exp["case"]["playback_slowdown"])
scales = exp["source_contract"]["part_motion_scales"]
scale_pct = float(exp["case"]["motion_scale"]) * 100
arm_l_pct = float(scales.get("arms_left", scales.get("arms", 0))) * 100
arm_r_pct = float(scales.get("arms_right", scales.get("arms", 0))) * 100

model = mujoco.MjModel.from_xml_path(str(SCENE))
data = mujoco.MjData(model)
for g in range(model.ngeom):
    if model.geom_bodyid[g] == 0:
        model.geom_group[g] = 2
opt = mujoco.MjvOption()
opt.geomgroup[0] = 0
renderer = mujoco.Renderer(model, height=480, width=500)
cam = mujoco.MjvCamera()
cam.elevation = -12
px_per_m = 220.0
fovy_px = renderer.height / (2.0 * np.tan(np.deg2rad(float(model.vis.global_.fovy)) / 2.0))
cam.distance = fovy_px * np.cos(np.deg2rad(cam.elevation)) / px_per_m

out = OUT / "真人到SONIC全身动态对齐_QP仿真时钟.mp4"
raw = out.with_suffix(".raw.mp4")
fps = 8
frames = int(np.ceil(duration * fps)) + 1
writer = cv2.VideoWriter(str(raw), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1120, 620))
assert writer.isOpened(), raw
left_px_per_m = 220.0
sim_axis_t = sim_t
torso = model.body("torso_link").id
root_drift = float(review["metrics"]["actual_root_xy_drift_m"])
drift_limit = float(exp["guards"]["root_xy_drift_max_m"])

for k in range(frames):
    rel_sim = min(duration, k / fps)
    sim_time = motion_start + rel_sim
    # QP logs carry the human source clock, including pauses while COM transfers.
    source_elapsed = float(np.interp(sim_time, qp_t, qp_elapsed))
    source_t = float(np.clip(start_source_t + source_elapsed / slowdown,
                             start_source_t, start_source_t + source_duration / slowdown))
    h_idx = int(np.argmin(np.abs(human_t - source_t)))
    pts = human_p[h_idx].copy()
    pts[:, :2] -= anchor[:2]

    canvas = np.full((620, 1120, 3), 246, np.uint8)
    cv2.putText(canvas, "HUMAN SOURCE  |  CMU 13_30 prerecorded motion", (18, 28), 0, .52, (20, 35, 50), 1, cv2.LINE_AA)
    cv2.putText(canvas, "not live camera  |  playback slowed 9x", (18, 53), 0, .40, (55, 65, 75), 1, cv2.LINE_AA)
    cv2.putText(canvas, "ROBOT  |  measured MuJoCo physics state", (575, 28), 0, .50, (20, 35, 50), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"retarget scale lower {scale_pct:.0f}%  arms L/R {arm_l_pct:.0f}/{arm_r_pct:.0f}%", (575, 53), 0, .39, (55, 65, 75), 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Exploratory trial | no fall | root drift {root_drift*100:.1f}/{drift_limit*100:.0f} cm", (18, 82), 0, .40, (25, 65, 95), 1, cv2.LINE_AA)

    sx = 225 + left_px_per_m * (.75 * pts[:, 1] + .65 * pts[:, 0])
    sy = 535 - left_px_per_m * (pts[:, 2] - floor)
    for a, b in edges:
        if any(part in names[b] for part in ("fingers", "thumb")):
            continue
        col = (190, 80, 35) if names[b].startswith("l") else (30, 100, 210) if names[b].startswith("r") else (55, 55, 55)
        cv2.line(canvas, (int(sx[a]), int(sy[a])), (int(sx[b]), int(sy[b])), col, 3, cv2.LINE_AA)

    human_bar = int(round(left_px_per_m * .5))
    cv2.line(canvas, (535, 535-human_bar), (535, 535), (25, 25, 25), 2, cv2.LINE_AA)
    cv2.line(canvas, (529, 535-human_bar), (541, 535-human_bar), (25, 25, 25), 2, cv2.LINE_AA)
    cv2.line(canvas, (529, 535), (541, 535), (25, 25, 25), 2, cv2.LINE_AA)
    cv2.putText(canvas, "0.5 m", (485, 555), 0, .38, (25, 25, 25), 1, cv2.LINE_AA)

    robot_idx = nearest_index(sim_axis_t, sim_time)
    data.qpos[:] = sim_qp[robot_idx]
    data.qvel[:] = sim_qv[robot_idx]
    mujoco.mj_forward(model, data)
    cam.lookat[:] = data.xpos[torso]
    cam.lookat[2] = max(.4, float(data.qpos[2]) - .1)
    rot = data.xmat[torso].reshape(3, 3)
    cam.azimuth = np.degrees(np.arctan2(rot[1, 0], rot[0, 0])) + 145
    renderer.update_scene(data, camera=cam, scene_option=opt)
    canvas[95:575, 570:1070] = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
    cv2.putText(canvas, f"sim {sim_time:.2f}s  |  human source {source_t:.2f}s  |  robot t={sim_t[robot_idx]:.2f}s",
                (575, 592), 0, .37, (20, 35, 50), 1, cv2.LINE_AA)
    writer.write(canvas)
    if k % 40 == 0:
        print(f"FRAME {k}/{frames} sim={sim_time:.2f} source={source_t:.2f}", flush=True)

writer.release()
renderer.close()
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(raw), "-c:v", "libx264", "-preset", "fast",
                "-crf", "22", "-threads", "2", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(out)], check=True)
raw.unlink()
cap = cv2.VideoCapture(str(out))
count = 0
while cap.read()[0]:
    count += 1
cap.release()
assert count == frames, (count, frames)
print(json.dumps({"video": str(out), "frames": count, "duration_s": duration,
                  "sim_start_s": motion_start, "sim_end_s": motion_end,
                  "source_t_start_s": start_source_t, "source_t_end_s": start_source_t + source_duration / slowdown,
                  "root_drift_m": root_drift, "root_drift_guard_m": drift_limit}, ensure_ascii=False))
