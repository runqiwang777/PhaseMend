"""One guarded exploratory SONIC replay of the offline full-body retarget.

Run in the SONIC WSL simulation environment. No gate file is changed and no
hardware transport is used. The session aborts on stale physics, low root
height, excessive root drift, or a policy/session fault.
"""
from __future__ import annotations

import json
import http.client
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from scipy.interpolate import PchipInterpolator

R = Path('/home/wangrunqi/sonic_takeover_share_20261114')
S = Path('/mnt/d/Github/PhaseMend/workspace')
DEFAULT_SCALE = 0.05
DEFAULT_SLOWDOWN = 3.0
DEFAULT_ROOT_MIN = 0.70
DEFAULT_XY_MAX = 0.15
DEFAULT_SAMPLE_PERIOD = 0.02
REQUESTED_SCALE = float(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_SCALE
SLOWDOWN = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_SLOWDOWN
ROOT_MIN = float(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_ROOT_MIN
XY_MAX = float(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_XY_MAX
SAMPLE_PERIOD = float(sys.argv[5]) if len(sys.argv) > 5 else DEFAULT_SAMPLE_PERIOD
RUN_MODE = sys.argv[6] if len(sys.argv) > 6 else 'standing'
OVERLAY_GAIN = float(sys.argv[7]) if len(sys.argv) > 7 else 1.0
RUN_LABEL = sys.argv[8] if len(sys.argv) > 8 else ''
GAIT_SPEED_SCALE = float(sys.argv[9]) if len(sys.argv) > 9 else 1.0
PART_ARM_SCALE = float(os.environ.get('PHASEMEND_ARM_SCALE', REQUESTED_SCALE))
if not 0.0 < PART_ARM_SCALE <= 1.0:
    raise ValueError('Upper-arm motion scale must be in (0, 1].')
if not (0.0 < REQUESTED_SCALE <= 1.0 and SLOWDOWN >= 1.0 and ROOT_MIN > 0.0
        and XY_MAX > 0.0 and 0.01 <= SAMPLE_PERIOD <= 0.1
        and RUN_MODE in ('standing', 'gait_residual', 'human_reference') and 0.0 < OVERLAY_GAIN <= 1.0
        and 0.05 <= GAIT_SPEED_SCALE <= 1.0):
    raise ValueError('Expected valid scale, slowdown, guards, sample period, run mode and overlay gain')
SCALE_TAG = int(round(REQUESTED_SCALE * 1000))
SAMPLE_HZ = 1.0 / SAMPLE_PERIOD
MODE_TAG = ('' if RUN_MODE == 'standing' else
            f'_gaitres{int(round(OVERLAY_GAIN * 100)):03d}' if RUN_MODE == 'gait_residual' else
            '_humanref')
if RUN_MODE == 'gait_residual' and GAIT_SPEED_SCALE != 1.0:
    MODE_TAG += f'_gait{int(round(GAIT_SPEED_SCALE * 100)):03d}'
MODE_TAG += f'_{RUN_LABEL}' if RUN_LABEL else ''
CONTACT_QP_GAIT = RUN_MODE == 'human_reference' and os.environ.get('PHASEMEND_CONTACT_QP_GAIT') == '1'
if CONTACT_QP_GAIT:
    MODE_TAG += '_contactqpgait'
O = S / f'wholebody_sonic_trial_20261122_scale{SCALE_TAG:04d}_slow{SLOWDOWN:g}x_rate{SAMPLE_HZ:g}hz{MODE_TAG}'
PRE = (Path(os.environ['PHASEMEND_REFERENCE_DIR']) if os.environ.get('PHASEMEND_REFERENCE_DIR')
       else (S / 'wholebody_pose_retarget_preflight_20261122'
             if abs(REQUESTED_SCALE - DEFAULT_SCALE) < 1e-9
             else S / f'wholebody_pose_retarget_scale_{SCALE_TAG:03d}'))
SIM_PYTHON = Path(os.environ.get('PHASEMEND_PYTHON', str(R / '.runtime/.venv_sim/bin/python')))
if not SIM_PYTHON.exists():
    SIM_PYTHON = Path('/home/wangrunqi/.venvs/phasemend-sim/bin/python')
sys.path[:0] = [str(S), str(R / 'human_motion' / 'scripts')]
from reliable_control_api import ReliableControlAPI  # noqa: E402
from parameter_adapter import ParameterAdapter  # noqa: E402


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def jsonl_rows(path):
    if not path.exists():
        return []
    out = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            try:
                if line.strip():
                    out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def latest_row(path):
    if not path.exists():
        return None
    with path.open('rb') as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(max(0, size - 16384), os.SEEK_SET)
        lines = f.read().splitlines()
    for line in reversed(lines):
        try:
            if line.strip():
                return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def appended_rows(path, offset):
    if not path.exists():
        return [], offset
    with path.open('rb') as f:
        f.seek(offset)
        data = f.read()
        new_offset = f.tell()
    rows = []
    for line in data.splitlines():
        try:
            if line.strip():
                rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows, new_offset


def main():
    O.mkdir(parents=True, exist_ok=True)
    source = np.load(PRE / 'joint_reference_candidate.npz', allow_pickle=False)
    preflight_meta = json.loads((PRE / 'preflight.json').read_text(encoding='utf-8'))
    part_motion_scales = preflight_meta.get('chosen_part_motion_scales', {
        'lower': REQUESTED_SCALE, 'torso': REQUESTED_SCALE, 'root': REQUESTED_SCALE,
        'feet': REQUESTED_SCALE, 'arms': PART_ARM_SCALE})
    arm_scale_left = float(part_motion_scales.get('arms_left', part_motion_scales.get('arms', PART_ARM_SCALE)))
    arm_scale_right = float(part_motion_scales.get('arms_right', part_motion_scales.get('arms', PART_ARM_SCALE)))
    times = (source['source_t'] - source['source_t'][0]) * SLOWDOWN
    q_points = source['q']
    base = source['reference_anchor_q']
    interpolator = PchipInterpolator(times, q_points, axis=0)
    duration = float(times[-1])
    dense_t = np.arange(0.0, duration, SAMPLE_PERIOD)
    if not np.isclose(dense_t[-1], duration):
        dense_t = np.append(dense_t, duration)
    dense_q = interpolator(dense_t)
    dense_v = interpolator.derivative(1)(dense_t)
    dense_a = interpolator.derivative(2)(dense_t)
    if RUN_MODE == 'gait_residual':
        dense_reference_q = base[None, :] + OVERLAY_GAIN * (dense_q - base[None, :])
        dense_reference_v = OVERLAY_GAIN * dense_v
        dense_reference_a = OVERLAY_GAIN * dense_a
        dense_operator_q = dense_reference_q - base[None, :]
    else:
        dense_reference_q = dense_q
        dense_reference_v = dense_v
        dense_reference_a = dense_a
        dense_operator_q = dense_q
    limit_summary = {
        'duration_s': duration,
        'sample_period_s': SAMPLE_PERIOD,
        'sample_rate_hz': SAMPLE_HZ,
        'sample_count': len(dense_t),
        'operator_update_hz': SAMPLE_HZ,
        'run_mode': RUN_MODE,
        'overlay_gain': OVERLAY_GAIN if RUN_MODE == 'gait_residual' else 1.0,
        'max_reference_speed_rad_s': float(np.max(np.abs(dense_reference_v))),
        'max_reference_acceleration_rad_s2': float(np.max(np.abs(dense_reference_a))),
        'first_reference_max_delta_from_base_rad': float(np.max(np.abs(dense_reference_q[0] - base))),
        'max_reference_step_rad': float(np.max(np.abs(np.diff(dense_reference_q, axis=0)))),
    }
    if (limit_summary['max_reference_speed_rad_s'] > 1.5
            or limit_summary['max_reference_acceleration_rad_s2'] > 8.0
            or limit_summary['max_reference_step_rad'] > 0.05):
        save(O / 'preflight_limits.json', limit_summary)
        raise RuntimeError('Resampled reference exceeds a speed, acceleration, or 0.05 rad step guard')
    save(O / 'preflight_limits.json', limit_summary)

    api = ReliableControlAPI(8792)
    try:
        previous = api.state()
    except OSError:
        previous = None
    if previous and previous.get('phase') == 'running':
        raise RuntimeError('A SONIC session is already running; refusing to take over an existing session')

    case = {
        'id': f'human_gait_fullbody_retarget_l{SCALE_TAG:03d}_aL{int(round(arm_scale_left * 1000)):03d}_R{int(round(arm_scale_right * 1000)):03d}_{RUN_MODE}',
        'group': 'full_body', 'pose': 'B_balanced_2cm', 'repetition': 1,
        'comparison_method': 'wholebody_contact_phase_kinematic_retarget',
                'source': f'CMU 13_30 prerecorded motion; {float(source["source_t"][0]):.3f}–{float(source["source_t"][-1]):.3f} s; {SLOWDOWN:g}x slowed; {SAMPLE_HZ:g}Hz operator updates; mode={RUN_MODE}',
        'motion_scale': REQUESTED_SCALE,
        'part_motion_scales': part_motion_scales,
        'operator_update_hz': SAMPLE_HZ,
        'controller_mode': ('cyclic_gait' if RUN_MODE == 'gait_residual' else
                            'human_reference' if RUN_MODE == 'human_reference' else 'stationary'),
        'motion_overlay_gain': OVERLAY_GAIN if RUN_MODE == 'gait_residual' else 1.0,
        'gait_speed_scale': GAIT_SPEED_SCALE if RUN_MODE == 'gait_residual' else None,
        'pose_file': str(R / 'human_motion/runs/balanced_standing_pilot_20261119/pose_B_balanced_2cm.json'),
        'record': str(R / 'human_motion/data/cmu13_30/human_skeleton_candidate.npz'),
        'preflight_directory': str(PRE), 'playback_slowdown': SLOWDOWN,
        'playback_trigger_file': str(O / 'human_reference.start'),
    }
    save(O / 'case.json', case)
    log = (O / 'session.log').open('w', encoding='utf-8', buffering=1)
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', PYTHONDONTWRITEBYTECODE='1')
    if RUN_MODE == 'human_reference':
        env['PHASEMEND_REFERENCE_DIR'] = str(PRE)
        env['PHASEMEND_PLAYBACK_TRIGGER_FILE'] = str(O / 'human_reference.start')
        env['PHASEMEND_PLAYBACK_SLOWDOWN'] = str(SLOWDOWN)
    session_script = (S / 'sonic_locomotion_session_worker.py' if RUN_MODE == 'gait_residual'
                      else S / 'sonic_human_retarget_gaitqp_session_worker.py' if RUN_MODE == 'human_reference' and os.environ.get('PHASEMEND_CONTACT_QP_GAIT') == '1'
                      else S / 'sonic_human_retarget_session_worker.py' if RUN_MODE == 'human_reference'
                      else R / 'human_motion/scripts/human_mocap_session.py')
    if not session_script.exists():
        log.close()
        raise FileNotFoundError(f'Existing SONIC session entry point missing: {session_script}')
    proc = subprocess.Popen(
        [str(SIM_PYTHON), '-u', str(session_script), 'session',
         '--case', str(O / 'case.json'), '--output', str(O), '--port', '8792'],
        cwd=R, env=env, stdout=log, stderr=subprocess.STDOUT)

    run_path = None
    started = ended = None
    failure_at = None
    failure = None
    events = []
    max_joint_clip_rad = 0.0
    last_q = np.zeros_like(base, dtype=float) if RUN_MODE == 'gait_residual' else np.asarray(base, float)
    initial_xy = None
    lower_guard = {'root_height_min_m': ROOT_MIN,
                   'root_xy_drift_max_m': XY_MAX if RUN_MODE in ('standing', 'human_reference') else None,
                   'root_xy_speed_max_m_s': 2.5 if RUN_MODE == 'gait_residual' else None,
                   'physics_stale_max_s': .25}
    try:
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError('SONIC session exited during initialization')
            try:
                state = api.state()
            except (OSError, http.client.HTTPException):
                time.sleep(.1)
                continue
            if state.get('phase') == 'running':
                run_path = Path(state['run'])
                break
            time.sleep(.1)
        if run_path is None:
            raise TimeoutError('SONIC session startup exceeded 240 seconds')
        time.sleep(2)
        state = api.state()
        phys = latest_row(run_path / 'sim.jsonl')
        if phys is None:
            raise RuntimeError('No completed physics sample before takeover')
        if time.monotonic() - float(phys['t']) > lower_guard['physics_stale_max_s']:
            raise RuntimeError('Physics feedback already stale before takeover')
        initial_xy = np.asarray(phys['qpos'][:2], float)
        if RUN_MODE == 'human_reference':
            # The human retarget is the SONIC base reference; duplicate HTTP
            # operator updates would add latency without changing that target.
            started = time.monotonic()
            Path(case['playback_trigger_file']).touch()
            events.append({'action': 'human_reference_started', 't': started})
        else:
            takeover_operator_anchor = np.zeros_like(base, dtype=float) if RUN_MODE == 'gait_residual' else base
            ack = api.post('takeover', takeover_operator_anchor,
                           method='bounded',
                           group='full_body', target_mode='relative' if RUN_MODE == 'gait_residual' else 'absolute')
            api.post('operator', last_q)
            events.append({'action': 'takeover', 't': time.monotonic(), 'ack': ack})
            started = time.monotonic()
        tick = started
        next_guard = started
        rows = []
        input_file = (O / 'input.jsonl').open('w', encoding='utf-8', buffering=1)
        k = 0
        next_physics_poll = started
        next_state_poll = started + .4
        event_offset = 0
        fall_seen = False
        contact_qp_deadline = started + max(120.0, duration * 4.0) if CONTACT_QP_GAIT else None
        while True:
            now = time.monotonic()
            replay_done = k >= len(dense_t)
            if replay_done and not CONTACT_QP_GAIT:
                break
            qp_complete = False
            if replay_done and run_path is not None:
                qp_row = latest_row(run_path / 'contact_qp_gait.jsonl')
                qp_complete = bool(qp_row and float(qp_row.get('source_elapsed_s', 0.0)) >= duration - SAMPLE_PERIOD)
                if contact_qp_deadline is not None and now > contact_qp_deadline and not qp_complete:
                    raise TimeoutError('Contact-QP simulation-time reference did not finish before the runtime safety limit')
            elapsed = now - started
            source_lag = elapsed - float(dense_t[k]) if not replay_done else None
            if source_lag is not None and source_lag > .25:
                raise TimeoutError(f'Retarget reference became stale: {source_lag:.3f} s')
            if proc.poll() is not None:
                raise RuntimeError('SONIC session exited during retarget replay')
            if now >= next_physics_poll:
                phys = latest_row(run_path / 'sim.jsonl')
                if phys is None or now - float(phys['t']) > lower_guard['physics_stale_max_s']:
                    raise RuntimeError('Physics feedback stale during retarget replay')
                qpos = np.asarray(phys['qpos'], float)
                xy_drift = float(np.linalg.norm(qpos[:2] - initial_xy))
                if qpos[2] < lower_guard['root_height_min_m']:
                    raise RuntimeError(f'Root height guard triggered: {qpos[2]:.4f} m')
                if RUN_MODE == 'gait_residual':
                    root_speed = float(np.linalg.norm(np.asarray(phys.get('qvel', [0., 0.]), float)[:2]))
                    if root_speed > lower_guard['root_xy_speed_max_m_s']:
                        raise RuntimeError(f'Root horizontal speed guard triggered: {root_speed:.4f} m/s')
                elif xy_drift > lower_guard['root_xy_drift_max_m']:
                    raise RuntimeError(f'Root drift guard triggered: {xy_drift:.4f} m')
                new_events, event_offset = appended_rows(run_path / 'physics_events.jsonl', event_offset)
                fall_seen = fall_seen or any(x.get('kind') == 'fall' for x in new_events)
                if fall_seen:
                    raise RuntimeError('Fall event recorded during retarget replay')
                if RUN_MODE == 'gait_residual':
                    latest_reference = latest_row(run_path / 'published.jsonl')
                    if latest_reference and latest_reference.get('state') == 'active':
                        baseline_q = np.asarray(latest_reference['base_q'], float)
                        operator_q = np.asarray(latest_reference['operator_q'], float)
                        applied_q = np.asarray(latest_reference['q'], float)
                        requested_q = baseline_q + operator_q
                        max_joint_clip_rad = max(max_joint_clip_rad,
                                                 float(np.max(np.abs(requested_q - applied_q))))
                        if max_joint_clip_rad > .05:
                            raise RuntimeError(f'Joint-limit projection exceeded 0.05 rad: {max_joint_clip_rad:.4f} rad')
                next_physics_poll = now + .1
            if now >= next_state_poll:
                state = api.state()
                if state.get('error') or state.get('phase') != 'running':
                    raise RuntimeError('SONIC publisher/session left running state')
                next_state_poll = time.monotonic() + .5
                if k < len(dense_t):
                    source_lag = time.monotonic() - started - float(dense_t[k])
                    if source_lag > .25:
                        raise TimeoutError(f'Retarget reference became stale during state check: {source_lag:.3f} s')

            if replay_done:
                if qp_complete:
                    break
                time.sleep(.02)
                continue

            q = dense_operator_q[k]
            post_before = time.monotonic()
            if RUN_MODE != 'human_reference':
                api.post('operator', q)
            post_after = time.monotonic()
            last_q = q.copy()
            row = {'replay_t_s': float(dense_t[k]), 't': post_after,
                   'source_t_s': float(source['source_t'][0] + dense_t[k] / SLOWDOWN),
                   'q': dense_reference_q[k].tolist(), 'operator_q': q.tolist(),
                   'qdot': dense_reference_v[k].tolist(), 'qddot': dense_reference_a[k].tolist(),
                   'source_lag_s': source_lag,
                   'operator_post_ms': ((post_after - post_before) * 1000.
                                        if RUN_MODE != 'human_reference' else 0.0)}
            rows.append(row)
            input_file.write(json.dumps(row, allow_nan=False) + '\n')
            k += 1
            if k < len(dense_t):
                tick = started + float(dense_t[k])
                delay = tick - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
        ended = time.monotonic()
        events.append({'action': 'human_end', 't': ended})
        if RUN_MODE == 'human_reference':
            state = api.state()
            if state.get('editor_state') != 'automatic':
                raise RuntimeError('SONIC editor left AUTO during direct human-reference playback')
            events.append({'action': 'automatic_reference_end', 't': time.monotonic()})
        else:
            until = time.monotonic() + .4
            while time.monotonic() < until:
                api.post('operator', last_q)
                time.sleep(.02)
            api.post('release')
            events.append({'action': 'release', 't': time.monotonic()})
            deadline = time.monotonic() + 8
            while api.state().get('editor_state') != 'automatic':
                if time.monotonic() > deadline:
                    raise TimeoutError('SONIC did not return to AUTO after release')
                time.sleep(.05)
            events.append({'action': 'returned_to_AUTO', 't': time.monotonic()})
        api.post('stop')
        proc.wait(timeout=25)
        input_file.close()
    except Exception as exc:
        failure = str(exc)
        failure_at = time.monotonic()
        if 'input_file' in locals() and not input_file.closed:
            input_file.close()
        try:
            if started is not None:
                if RUN_MODE == 'human_reference':
                    events.append({'action': 'automatic_reference_abort', 't': time.monotonic(), 'reason': failure})
                else:
                    until = time.monotonic() + .2
                    while time.monotonic() < until:
                        api.post('operator', last_q)
                        time.sleep(.02)
                    api.post('release')
                    events.append({'action': 'emergency_release', 't': time.monotonic(), 'reason': failure})
                    release_deadline = time.monotonic() + 8
                    while time.monotonic() < release_deadline:
                        state = api.state()
                        if state.get('editor_state') == 'automatic':
                            events.append({'action': 'returned_to_AUTO', 't': time.monotonic()})
                            break
                        time.sleep(.05)
        except Exception:
            pass
    finally:
        if proc.poll() is None:
            try:
                api.post('stop')
                proc.wait(timeout=12)
            except Exception:
                proc.send_signal(signal.SIGINT)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.terminate()
                    proc.wait(timeout=5)
        log.close()
        if run_path is None and (O / 'session_result.json').exists():
            try:
                result = json.loads((O / 'session_result.json').read_text(encoding='utf-8'))
                run_path = Path(result['run'])
            except Exception:
                pass

    experiment = {
        'case': case, 'run': str(run_path) if run_path else None,
        'failure': failure, 'session_exit': proc.returncode,
        'start': started, 'end': ended, 'failure_at': failure_at, 'events': events,
        'guards': lower_guard, 'preflight_limits': limit_summary,
        'scheduler': {'type': 'fixed source-time grid', 'operator_update_hz': SAMPLE_HZ,
                      'max_reference_step_rad': limit_summary['max_reference_step_rad']},
        'max_joint_clip_rad_observed': max_joint_clip_rad,
        'input_samples': len(jsonl_rows(O / 'input.jsonl')),
        'source_contract': {'dataset': 'CMU 13_30', 'live_camera': False,
                            'motion_scale': REQUESTED_SCALE,
                            'part_motion_scales': case['part_motion_scales'],
                            'preflight_directory': str(PRE),
                            'overlay_gain': OVERLAY_GAIN if RUN_MODE == 'gait_residual' else 1.0,
                            'controller_mode': ('cyclic_gait' if RUN_MODE == 'gait_residual' else
                                                'human_reference' if RUN_MODE == 'human_reference' else 'stationary'),
                            'method': 'wholebody joint-pose retarget with stance anchors and swing-foot targets'}
    }
    save(O / 'experiment.json', experiment)
    print(json.dumps({'failure': failure, 'session_exit': proc.returncode,
                      'run': str(run_path) if run_path else None,
                      'input_samples': experiment['input_samples'],
                      'start': started, 'end': ended,
                      'events': [x['action'] for x in events],
                      'preflight_limits': limit_summary}, ensure_ascii=False, indent=2))

    if run_path is not None:
        observed_until = ended if ended is not None else failure_at
        sim = [x for x in jsonl_rows(run_path / 'sim.jsonl')
               if started is not None and observed_until is not None
               and started <= x.get('t', -1) <= observed_until + .2]
        events_phys = jsonl_rows(run_path / 'physics_events.jsonl')
        metrics = {'sim_samples': len(sim), 'fall_events': [x for x in events_phys if x.get('kind') == 'fall']}
        if sim:
            qpos = np.asarray([x['qpos'] for x in sim], float)
            actual_root = qpos[:, :3]
            adapter = ParameterAdapter()
            knees = []
            feet = []
            hands = []
            for row in sim:
                adapter.data.qpos[:] = row['qpos']
                mujoco.mj_forward(adapter.model, adapter.data)
                knees.append(adapter.data.qpos[adapter.addresses[[3, 9]]].copy())
                feet.append(adapter.data.xpos[[adapter.model.body(s + '_ankle_roll_link').id for s in ['left', 'right']]].copy())
                hands.append(adapter.data.xpos[[adapter.model.body(s + '_wrist_roll_link').id for s in ['left', 'right']]].copy())
            knees = np.asarray(knees)
            feet = np.asarray(feet)
            hands = np.asarray(hands)
            metrics.update({
                'actual_root_min_z_m': float(actual_root[:, 2].min()),
                'actual_root_drop_m': float(actual_root[0, 2] - actual_root[:, 2].min()),
                'actual_root_xy_drift_m': float(np.max(np.linalg.norm(actual_root[:, :2] - actual_root[0, :2], axis=1))),
                'actual_root_xy_speed_max_m_s': float(max(np.linalg.norm(np.asarray(x.get('qvel', [0., 0.]), float)[:2]) for x in sim)),
                'actual_knee_excursion_rad': np.ptp(knees, axis=0).tolist(),
                'actual_foot_xy_drift_m': np.max(np.linalg.norm(feet[:, :, :2] - feet[0:1, :, :2], axis=2), axis=0).tolist(),
                'actual_wrist_xyz_excursion_m': np.ptp(hands, axis=0).tolist(),
                'double_support_fraction': float(np.mean([all(x.get('foot_contact', {}).values()) for x in sim])),
            })
        checks = {
            'no_runner_failure': failure is None,
            'session_exit_zero': proc.returncode == 0,
            'returned_to_AUTO': (any(x['action'] == 'returned_to_AUTO' for x in events)
                                 or any(x['action'] == 'automatic_reference_end' for x in events)),
            'no_fall_events': not metrics['fall_events'],
            'root_height_guard_passed': not sim or metrics['actual_root_min_z_m'] > lower_guard['root_height_min_m'],
            'root_drift_guard_passed': (not sim or RUN_MODE == 'gait_residual'
                                        or metrics['actual_root_xy_drift_m'] < lower_guard['root_xy_drift_max_m']),
            'root_speed_guard_passed': (not sim or RUN_MODE != 'gait_residual'
                                        or metrics['actual_root_xy_speed_max_m_s'] < lower_guard['root_xy_speed_max_m_s']),
            'joint_limit_projection_guard_passed': max_joint_clip_rad <= .05,
            'reference_limits_preflight_passed': limit_summary['max_reference_speed_rad_s'] <= 1.5 and limit_summary['max_reference_acceleration_rad_s2'] <= 8.0 and limit_summary['max_reference_step_rad'] <= .05,
        }
        save(O / 'review.json', {'case': case['id'], 'passed': bool(all(checks.values())),
                                 'checks': checks, 'metrics': metrics,
                                 'scope': 'one simulation exploratory trial; not reliability gate acceptance'})
        print(json.dumps({'review_checks': checks, 'metrics': metrics}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()


