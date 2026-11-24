"""SONIC replay worker using the human-retargeted pose as the moving reference."""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation, Slerp
ROOT=Path('/home/wangrunqi/sonic_takeover_share_20261114')
S=Path('/mnt/d/Github/PhaseMend/workspace')
MOTION_START_MARKER=None
SESSION_OBJ=None
USE_CONTACT_GAIT=os.environ.get('PHASEMEND_CONTACT_QP_GAIT')=='1'
sys.path[:0]=[str(ROOT/'takeover'),str(ROOT/'human_motion'/'scripts')]
import takeover.session_staged_standing as sm
from safe_reference_editor import SafeReferenceEditor

class HumanRetargetMotion:
    """C1 human-to-G1 trajectory, held at frame zero until the runner triggers it."""
    def __init__(self,directory,loop=False):
        ref=Path(os.environ['PHASEMEND_REFERENCE_DIR'])
        slowdown=float(os.environ.get('PHASEMEND_PLAYBACK_SLOWDOWN','9'))
        self.trigger=Path(os.environ['PHASEMEND_PLAYBACK_TRIGGER_FILE'])
        pre=json.loads((ref/'preflight.json').read_text(encoding='utf-8'))
        candidate=np.load(ref/'joint_reference_candidate.npz',allow_pickle=False)
        source_t=np.asarray(candidate['source_t'],float)
        q=np.asarray(candidate['q'],float)
        self.source_times=(source_t-source_t[0])*slowdown
        self.duration=float(self.source_times[-1])
        qcurve=PchipInterpolator(self.source_times,q,axis=0)
        euler=np.asarray([r['root_target_rpy_deg'] for r in pre['rows']],float)
        rotations=Rotation.from_euler('xyz',euler,degrees=True)
        orientation=Slerp(self.source_times,rotations)
        self.times=np.arange(0.,self.duration+.009999,.02)
        self.q=np.asarray(qcurve(self.times),float)
        self.dq=np.asarray(qcurve.derivative()(self.times),float)
        self.quat=np.asarray([np.r_[rot.as_quat()[3],rot.as_quat()[:3]] for rot in orientation(self.times)])
        self.dq[-1]=0.
        self.loop=False
        self.loop_start=self.loop_end=None
        self.started_frame=None
    def describe(self):
        return {'loop':False,'loop_start':None,'loop_end':None,'period_s':None,
                'source':'synchronized human-to-G1 retarget; fixed source-time playback'}
    @staticmethod
    def cycle_index(frame):
        return 0
    def sample(self,frame,moving=True):
        if not moving:
            return self.q[0].copy(),np.zeros(29),self.quat[0].copy()
        if self.started_frame is None:
            if not self.trigger.exists():
                return self.q[0].copy(),np.zeros(29),self.quat[0].copy()
            self.started_frame=int(frame)
            if MOTION_START_MARKER is not None and not MOTION_START_MARKER.exists():
                temp=MOTION_START_MARKER.with_suffix('.tmp')
                debug_index=SESSION_OBJ.debug_latest.get('index') if SESSION_OBJ is not None else None
                temp.write_text(json.dumps({'frame_index':self.started_frame,'debug_index':debug_index,'wall_time':time.monotonic()}),encoding='utf-8')
                temp.replace(MOTION_START_MARKER)
        idx=int(np.clip(int(frame)-self.started_frame,0,len(self.q)-1))
        vel=self.dq[idx].copy() if idx<len(self.q)-1 else np.zeros(29)
        return self.q[idx].copy(),vel,self.quat[idx].copy()

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('session',choices=['session'])
    parser.add_argument('--case',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--port',type=int,default=8792)
    args=parser.parse_args()
    case=json.loads(Path(args.case).read_text(encoding='utf-8'))
    os.environ['PHASEMEND_REFERENCE_DIR']=case['preflight_directory']
    os.environ['PHASEMEND_PLAYBACK_TRIGGER_FILE']=case['playback_trigger_file']
    os.environ['PHASEMEND_PLAYBACK_SLOWDOWN']=str(case['playback_slowdown'])
    Path(case['playback_trigger_file']).unlink(missing_ok=True)
    sm.Motion=HumanRetargetMotion
    sm.Editor=SafeReferenceEditor
    session_args=argparse.Namespace(headless=True,web=True,port=args.port,duration=0.0,
        scenario='interactive',method='bounded',moving=True,once=True,amplitude=0.0,
        window_frames=51,exercise_api=False,exercise_viewer=False,exercise_walking=False)
    if USE_CONTACT_GAIT:
        class ContactGaitSession(sm.Session):
            def launch(self,name,argv,terminal=False):
                if name=='sim':argv=[*argv[:-1],'takeover.range_contact_qp_gait']
                return super().launch(name,argv,terminal)
        sm.Session=ContactGaitSession
    session=sm.Session(session_args)
    if USE_CONTACT_GAIT:
        global MOTION_START_MARKER,SESSION_OBJ
        MOTION_START_MARKER=session.run/'human_motion_start.json'
        SESSION_OBJ=session
        session.env['PHASEMEND_CONTACT_QP_GAIT']='1'
        session.env['PHASEMEND_ROOT_XY_WEIGHT']=os.environ.get('PHASEMEND_ROOT_XY_WEIGHT','0.5')
        (session.run/'contact_qp_gait.request').touch()
    session.cfg['standing_contact_release']=True
    session.env['D1_CONTACT_INITIALIZATION']='1'
    session.event('wholebody_human_reference_case',case=case,
                  editor='human retarget trajectory is the SONIC base reference')
    try:
        session.start()
    except KeyboardInterrupt:
        session.phase='interrupted'
    except Exception as exc:
        session.error=str(exc);session.phase='failed'
        print(f'SESSION ERROR {exc}',flush=True)
    finally:
        session.close()
    try:
        from takeover.analyze import analyze
        metrics=analyze(session.run)
    except Exception as exc:
        metrics={'analysis_error':str(exc)}
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    result={'run':str(session.run),'phase':session.phase,'error':session.error,'case':case,'metrics':metrics}
    (out/'session_result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'run':str(session.run),'phase':session.phase,'error':session.error,
                      'metrics':metrics},ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':
    main()





