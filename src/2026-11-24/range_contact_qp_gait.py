"""Hybrid contact-phase whole-body tracking QP. Independent controller; SONIC initializes the robot only."""
import json, os, time, threading
from pathlib import Path
import numpy as np, mujoco, osqp
from scipy import sparse
from scipy.interpolate import PchipInterpolator
from scipy.spatial.transform import Rotation, Slerp

class ContactGaitQP:
 def __init__(self,m,d,limit,reference,preflight_dir,start_marker):
  self.m=m;self.d=d;self.nv=m.nv;self.nu=m.nu;self.n=self.nv+self.nu+12
  self.limit=np.asarray(limit,float);self.jids=m.actuator_trnid[:,0].copy();self.qa=m.jnt_qposadr[self.jids];self.va=m.jnt_dofadr[self.jids]
  assert m.nv==49 and m.nu==43 and np.all(m.actuator_gear[:,0]==1.)
  from takeover.motion import NAMES
  self.body_act=np.array([int(np.flatnonzero(self.jids==m.joint(n).id)[0]) for n in NAMES])
  self.anchor=d.qpos[self.qa].copy();self.reference=np.asarray(reference,float).copy();self.desired=self.anchor.copy();self.vdes=np.zeros(self.nu)
  self.qrange=m.jnt_range[self.jids];self.root=d.qpos[:7].copy();self.rootR=Rotation.from_quat(self.root[[4,5,6,3]])
  self.feet=[m.body(s+'_ankle_roll_link').id for s in ['left','right']]
  self.footpos=d.xpos[self.feet].copy();self.footrot=[Rotation.from_matrix(d.xmat[b].reshape(3,3)) for b in self.feet]
  self.foot_anchor_pos=self.footpos.copy();self.foot_anchor_rot=list(self.footrot)
  self.bounds=[]
  for i,b in enumerate(self.feet):
   pts=[]
   for c in d.contact:
    bodies=m.geom_bodyid[[c.geom1,c.geom2]]
    if b in bodies and 0 in bodies and c.dist<.002:pts.append(c.pos.copy())
   if len(pts)<3:raise ValueError('QP entry needs >=3 measured sole contacts on both feet')
   local=(np.asarray(pts)-self.footpos[i])@self.footrot[i].as_matrix()
   lo=local.min(axis=0);hi=local.max(axis=0)
   if hi[0]-lo[0]<.10 or hi[1]-lo[1]<.03:raise ValueError('Measured sole support polygon too small')
   self.bounds.append([lo[0]+.006,hi[0]-.006,lo[1]+.006,hi[1]-.006,float(local[:,2].mean())])
  self.swing_height_gain=float(os.environ.get("PHASEMEND_SWING_HEIGHT_GAIN","1.0"))
  if not 1.<=self.swing_height_gain<=2.:raise ValueError('PHASEMEND_SWING_HEIGHT_GAIN must be in [1,2]')
  self.root_xy_weight=float(os.environ.get("PHASEMEND_ROOT_XY_WEIGHT","0.5"));self.com_entry=d.subtree_com[0,:2].copy();self.com_support_local=None;self.com_support_raw=None;self.com_support_bounds=None;self.com_support_speed=None;self.previous_com_J=None;self.last_sim=None;self.last_tau=np.zeros(self.nu);self.solver=None;self.last_info=None
  self.S=np.zeros((self.nv,self.nu));self.S[self.va,np.arange(self.nu)]=1.
  self.start_marker=Path(start_marker);self.motion_start_index=None;self.motion_started=False;self.startup_root_xy=self.root[:2].copy()
  pf=json.loads((Path(preflight_dir)/'preflight.json').read_text(encoding='utf-8'));rows=pf['rows']
  self.source_t=np.asarray([r['source_t'] for r in rows],float);self.playback_slowdown=float(os.environ.get('PHASEMEND_PLAYBACK_SLOWDOWN',pf['source_playback_slowdown']))
  self.times=(self.source_t-self.source_t[0])*self.playback_slowdown
  self.contact_schedule=np.asarray([r['human_foot_contact'] for r in rows],bool)
  self.ref_root_xyz=np.asarray([r['root_target_xyz_m'] for r in rows],float)
  self.ref_root_rot=Rotation.from_euler('xyz',np.asarray([r['root_target_rpy_deg'] for r in rows],float),degrees=True)
  self.root_xyz_curve=PchipInterpolator(self.times,self.ref_root_xyz,axis=0)
  self.root_rot_curve=Slerp(self.times,self.ref_root_rot)
  self.root_rot_offset=self.rootR*self.ref_root_rot[0].inv()
  self.root_xyz_offset=self.root[:3]-self.ref_root_xyz[0]
  self.ready=False;self.prepare_error=None
  self._prepare_thread=threading.Thread(target=self._prepare_targets,args=(rows,),daemon=True)
  self._prepare_thread.start()
  self.begin=float(d.time)
  self.modes=['stance','stance'];self.source_contact=np.array([True,True]);self.source_elapsed=0.;self.pre_shift_active=True;self.contact_false_cycles=np.zeros(2,dtype=int)
  self.transfer_target_idx=None;self.transfer_start_time=float(d.time);self.transfer_start_com=self.com_entry.copy()
  self.last_index=0;self.last_ref=self.reference.copy();self.last_contact=[True,True]

 def _prepare_targets(self,rows):
  try:
   td=mujoco.MjData(self.m);foot_ref=[];ref_foot_rot=[]
   for r in rows:
    td.qpos[:7]=np.r_[r['root_target_xyz_m'],np.asarray(r['root_target_quat_xyzw'])[[3,0,1,2]]]
    td.qpos[self.qa[self.body_act]]=np.asarray(r['q'],float)
    mujoco.mj_forward(self.m,td)
    foot_ref.append(td.xpos[self.feet].copy())
    ref_foot_rot.append([Rotation.from_matrix(td.xmat[b].reshape(3,3)) for b in self.feet])
   self.foot_ref=np.asarray(foot_ref)
   self.foot_curves=[PchipInterpolator(self.times,self.foot_ref[:,i,:],axis=0) for i in range(2)]
   ref_foot_rot=[[ref_foot_rot[k][i] for k in range(len(rows))] for i in range(2)]
   self.foot_rot_curves=[Slerp(self.times,Rotation.concatenate(ref_foot_rot[i])) for i in range(2)]
   self.foot_offset=self.footpos-self.foot_ref[0]
   self.foot_rot_offsets=[self.footrot[i]*ref_foot_rot[i][0].inv() for i in range(2)]
   self.swing_start_times=[]
   for i in range(2):
    starts=np.flatnonzero((~self.contact_schedule[:,i]) & np.r_[True,self.contact_schedule[:-1,i]])
    self.swing_start_times.append(self.times[starts])
   self.ready=True
  except Exception as exc:self.prepare_error=str(exc)

 def _phase(self,index):
  if self.motion_start_index is None and self.start_marker.exists():
   try:
    marker=json.loads(self.start_marker.read_text())
    self.motion_start_index=int(marker.get('debug_index') if marker.get('debug_index') is not None else marker['frame_index'])
   except (OSError,ValueError,KeyError,json.JSONDecodeError):pass
   if self.motion_start_index is not None:self.motion_started=True
  if not self.motion_started:
   self.pre_shift_active=True
   return 0.,np.array([False,True])
  t=float(np.clip((int(index)-self.motion_start_index)*.02,0.,self.times[-1]))
  k=int(np.clip(np.searchsorted(self.times,t,side='right')-1,0,len(self.times)-1))
  return t,self.contact_schedule[k].copy()

 def target(self,ref,index):
  self.last_index=int(index);self.source_elapsed,self.source_contact=self._phase(index)
  r=np.asarray(ref,float)
  goal=self.anchor.copy();goal[self.body_act]=self.anchor[self.body_act]+r-self.reference
  goal=np.clip(goal,self.qrange[:,0]+.01,self.qrange[:,1]-.01)
  vel=np.clip((goal-self.desired)/.02,-2.,2.);self.desired=goal;self.vdes=.5*self.vdes+.5*vel
  self.last_ref=r.copy()

 def _com_in_support(self,i,margin=.005):
  b=self.feet[i];R=self.d.xmat[b].reshape(3,3);raw_local=R.T@(self.d.subtree_com[0]-self.d.xpos[b])
  vertical=R.T@np.array([0.,0.,1.]);plane_z=self.bounds[i][4];local=raw_local.copy();local[:2]-=((raw_local[2]-plane_z)/vertical[2])*vertical[:2]
  lo_x,hi_x,lo_y,hi_y,_=self.bounds[i]
  vel=np.zeros((3,self.nv));mujoco.mj_jacSubtreeCom(self.m,self.d,vel,0);v=(vel@self.d.qvel)[:2]
  self.com_support_local=local[:2].copy();self.com_support_raw=raw_local[:2].copy();self.com_support_bounds=np.array([lo_x,hi_x,lo_y,hi_y,plane_z]);self.com_support_speed=float(np.linalg.norm(v))
  return bool(lo_x+margin<=local[0]<=hi_x-margin and lo_y+margin<=local[1]<=hi_y-margin and self.com_support_speed<.10)
 def _actual_contacts(self):
  found=[]
  for b in self.feet:
   count=0;normal=0.
   for k,c in enumerate(self.d.contact):
    bodies=self.m.geom_bodyid[[c.geom1,c.geom2]]
    if b in bodies and 0 in bodies and c.dist<.003:
     w=np.zeros(6);mujoco.mj_contactForce(self.m,self.d,k,w);count+=1;normal+=max(0.,float(w[0]))
   found.append(count>=1)
  return np.asarray(found,bool)

 def _foot_target(self,i,t,scale_swing=False):
  p=np.asarray(self.foot_curves[i](t)+self.foot_offset[i]);v=np.asarray(self.foot_curves[i].derivative(1)(t));a=np.asarray(self.foot_curves[i].derivative(2)(t))
  if scale_swing and self.swing_height_gain>1. and len(self.swing_start_times[i]):
   k=int(np.searchsorted(self.swing_start_times[i],t,side='right')-1)
   if k>=0:
    t0=float(self.swing_start_times[i][k]);base_z=float(self.foot_curves[i](t0)[2]+self.foot_offset[i,2]);height=float(p[2]-base_z)
    if height>0.:
     p[2]=base_z+self.swing_height_gain*height;v[2]*=self.swing_height_gain;a[2]*=self.swing_height_gain
  r=self.foot_rot_offsets[i]*self.foot_rot_curves[i](t)
  return p,v,a,r

 def solve(self):
  m,d=self.m,self.d;start=time.perf_counter()
  if self.last_sim is not None and d.time-self.last_sim<.0099:return self.last_tau,self.last_info,False
  mujoco.mj_forward(m,d);dt=float(d.time-self.last_sim) if self.last_sim is not None else .01;self.last_sim=float(d.time)
  nv,nu,n=self.nv,self.nu,self.n;M=np.empty((nv,nv));mujoco.mj_fullM(m,M,d.qM);h=d.qfrc_bias.copy()-d.qfrc_passive
  actual=self._actual_contacts();Jfeet=[];rhs=[];foot_errors=[];modes=[]
  for i in range(2):
   self.contact_false_cycles[i]=0 if actual[i] else self.contact_false_cycles[i]+1
   actual[i]=bool(actual[i] or self.contact_false_cycles[i]<3)
  planned=self.source_contact.copy();effective=planned.copy();was_pre_shift=self.pre_shift_active;com_in_support=None;self.com_support_local=None;self.com_support_raw=None;self.com_support_bounds=None;self.com_support_speed=None
  supporters=np.flatnonzero(planned)
  if not self.motion_started:
   effective=np.array([True,True]);self.pre_shift_active=True
  elif len(supporters)==1:
   si=int(supporters[0]);com_in_support=self._com_in_support(si)
   if not actual[si]:
    effective=np.array([bool(actual[1-si]),True]);effective[si]=True;self.pre_shift_active=True
   elif not com_in_support:
    effective=np.array([True,True]);self.pre_shift_active=True
   else:
    self.pre_shift_active=False
  else:
   self.pre_shift_active=False
  if self.pre_shift_active and len(supporters)==1 and actual[int(supporters[0])]:
   si=int(supporters[0])
   if self.transfer_target_idx!=si or not was_pre_shift:
    self.transfer_start_time=float(d.time);self.transfer_start_com=d.subtree_com[0,:2].copy()
   self.transfer_target_idx=si
  else:self.transfer_target_idx=None
  if was_pre_shift and not self.pre_shift_active:
   for i in range(2):
    if not planned[i] and actual[i]:
     rot_now=Rotation.from_matrix(d.xmat[self.feet[i]].reshape(3,3))
     self.foot_offset[i]=d.xpos[self.feet[i]]-self.foot_curves[i](self.source_elapsed)
     self.foot_rot_offsets[i]=rot_now*self.foot_rot_curves[i](self.source_elapsed).inv()
  target_ps=[];target_rs=[]
  for i,b in enumerate(self.feet):
   jp=np.zeros((3,nv));jr=jp.copy();dp=jp.copy();dr=jp.copy()
   mujoco.mj_jacBody(m,d,jp,jr,b);mujoco.mj_jacDot(m,d,dp,dr,d.xpos[b],b)
   jac=np.vstack([jp,jr]);jdot=np.vstack([dp,dr]);pos=d.xpos[b];rot=Rotation.from_matrix(d.xmat[b].reshape(3,3))
   want=bool(effective[i])
   if want and actual[i]:
    mode='stance'
    if self.modes[i]!='stance':
     self.foot_anchor_pos[i]=pos.copy();self.foot_anchor_rot[i]=rot
   elif want:
    mode='landing'
   elif actual[i]:
    mode='liftoff'
   else:
    mode='swing'
   self.modes[i]=mode;modes.append(mode)
   if mode=='stance':
    pt=self.foot_anchor_pos[i];rt=self.foot_anchor_rot[i];vr=np.zeros(3);ar=np.zeros(3)
   else:
    pt,vr,ar,rt=self._foot_target(i,self.source_elapsed,scale_swing=mode in ('liftoff','swing'))
   target_ps.append(pt);target_rs.append(rt)
   perr=pt-pos;ore=(rt*rot.inv()).as_rotvec();v=jac@d.qvel
   kp=np.array([180.,180.,220.,100.,100.,100.]);kd=np.array([25.,25.,28.,20.,20.,20.])
   acc_ref=np.r_[ar,np.zeros(3)]
   rhs.append(acc_ref+kp*np.r_[perr,ore]+kd*(np.r_[vr,np.zeros(3)]-v)-jdot@d.qvel)
   Jfeet.append(jac);foot_errors.append(float(np.linalg.norm(perr)))
  J=np.vstack(Jfeet);contact_rhs=np.concatenate(rhs)
  P=np.eye(n)*1e-6;q=np.zeros(n)
  def task(A,b,weight):
   nonlocal P,q
   weights=np.broadcast_to(np.asarray(weight),(len(b),));P[:nv,:nv]+=A.T@(weights[:,None]*A);q[:nv]-=A.T@(weights*np.asarray(b))
  A=np.zeros((nu,nv));A[np.arange(nu),self.va]=1.
  kp=np.full(nu,50.);weight=np.full(nu,.2)
  kp[self.body_act[:12]]=65.;weight[self.body_act[:12]]=3.5
  kp[self.body_act[12:15]]=50.;weight[self.body_act[12:15]]=1.5
  kp[self.body_act[15:]]=70.;weight[self.body_act[15:]]=10.
  acc=kp*(self.desired-d.qpos[self.qa])+2*np.sqrt(kp)*(self.vdes-d.qvel[self.va])
  task(A,np.clip(acc,-80,80),weight)
  root_goal=self.rootR*self.ref_root_rot[0].inv()*self.root_rot_curve(self.source_elapsed)
  root_error=(root_goal*Rotation.from_quat(d.qpos[[4,5,6,3]]).inv()).as_rotvec()
  task(np.eye(nv)[3:6],80*root_error-18*d.qvel[3:6],25.)
  target_root_xyz=self.root_xyz_offset+self.root_xyz_curve(self.source_elapsed)
  task(np.eye(nv)[2:3],np.array([80*(target_root_xyz[2]-d.qpos[2])-18*d.qvel[2]]),25.)
  task(np.eye(nv)[:2],8*(target_root_xyz[:2]-d.qpos[:2])-5*d.qvel[:2],self.root_xy_weight)
  jc=np.zeros((3,nv));mujoco.mj_jacSubtreeCom(m,d,jc,0)
  bias=(jc-self.previous_com_J)@d.qvel/dt if self.previous_com_J is not None else np.zeros(3);self.previous_com_J=jc.copy()
  support=[i for i,mode in enumerate(modes) if mode=='stance']
  if not support:support=[i for i,v in enumerate(actual) if v]
  if not support:support=[int(np.argmax(self.source_contact))]
  if self.pre_shift_active and len(supporters)==1:
   si=int(supporters[0]);center_id=si if actual[si] else (int(np.flatnonzero(actual)[0]) if np.any(actual) else si);support=[center_id]
  center=np.mean([d.xpos[self.feet[i]][:2] for i in support],axis=0)
  transfer_progress=1.;com_goal=center.copy()
  if self.pre_shift_active and len(supporters)==1 and actual[int(supporters[0])]:
   transfer_progress=float(np.clip((d.time-self.transfer_start_time)/2.5,0.,1.));u=transfer_progress;blend=u**3*(10-15*u+6*u*u)
   com_goal=self.transfer_start_com+blend*(center-self.transfer_start_com)
  com_acc=np.clip(45*(com_goal-d.subtree_com[0,:2])-14*(jc@d.qvel)[:2]-bias[:2],-3.,3.);task(jc[:2],com_acc,500.)
  P[nv:nv+nu,nv:nv+nu]+=np.eye(nu)*.0002;q[nv:nv+nu]-=.0002*self.last_tau
  P[-12:,-12:]+=np.diag([.0001]*3+[.002]*3+[.0001]*3+[.002]*3)
  blocks=[];lower=[];upper=[]
  def constraint(mat,lo,hi):
   blocks.append(mat);lower.extend(np.broadcast_to(lo,(len(mat),)));upper.extend(np.broadcast_to(hi,(len(mat),)))
  constraint(np.c_[M,-self.S,-J.T],-h,-h)
  constraint(np.c_[J,np.zeros((12,nu+12))],contact_rhs,contact_rhs)
  torque=np.zeros((nu,n));torque[:,nv:nv+nu]=np.eye(nu);constraint(torque,-self.limit,self.limit)
  aa=np.zeros((nv,n));aa[:,:nv]=np.eye(nv);constraint(aa,-200.,200.)
  for i in range(2):
   rot=d.xmat[self.feet[i]].reshape(3,3);transform=np.zeros((6,6));transform[:3,:3]=rot.T;transform[3:,3:]=rot.T
   loc=np.zeros((6,n));loc[:,nv+nu+6*i:nv+nu+6*i+6]=transform
   fx,fy,fz,tx,ty,tz=loc;lo_x,hi_x,lo_y,hi_y,z=self.bounds[i]
   if modes[i]=='stance':constraint(fz[None],20.,500.)
   elif modes[i] in ('landing','liftoff'):constraint(fz[None],0.,500.)
   else:constraint(fz[None],0.,0.)
   friction=np.array([fx-.5*fz,-fx-.5*fz,fy-.5*fz,-fy-.5*fz]);constraint(friction,-np.inf,0.)
   copx=-ty+z*fx;copy=tx+z*fy
   constraint(np.array([copx-lo_x*fz,hi_x*fz-copx,copy-lo_y*fz,hi_y*fz-copy]),0.,np.inf)
   constraint(np.array([tz-.025*fz,-tz-.025*fz]),-np.inf,0.)
  dense=np.vstack(blocks);l=np.asarray(lower);u=np.asarray(upper)
  if self.solver is None:
   pmask=np.triu(np.ones((n,n),dtype=bool));self.pmask=pmask;row,col=np.nonzero(pmask);order=np.lexsort((row,col));self.p_rows=row[order];self.p_cols=col[order]
   mask=np.zeros(dense.shape,dtype=bool);mask[:nv,:nv]=True;mask[self.va,nv+np.arange(nu)]=True
   for side in range(2):
    ancestors=np.r_[np.arange(6),self.va[self.body_act[6*side:6*(side+1)]]]
    wcols=np.arange(nv+nu+6*side,nv+nu+6*side+6)
    mask[np.ix_(ancestors,wcols)]=True
    mask[np.ix_(np.arange(nv+6*side,nv+6*side+6),ancestors)]=True
    startrow=nv+12+nu+nv+11*side;mask[np.ix_(np.arange(startrow,startrow+11),wcols)]=True
   mask[nv+12+np.arange(nu),nv+np.arange(nu)]=True
   mask[nv+12+nu+np.arange(nv),np.arange(nv)]=True
   self.amask=mask;rr,cc=np.nonzero(mask);order=np.lexsort((rr,cc));self.a_rows=rr[order];self.a_cols=cc[order]
   Ps=sparse.csc_matrix((P[self.p_rows,self.p_cols],(self.p_rows,self.p_cols)),shape=(n,n))
   As=sparse.csc_matrix((dense[self.a_rows,self.a_cols],(self.a_rows,self.a_cols)),shape=dense.shape)
   self.solver=osqp.OSQP();self.solver.setup(P=Ps,q=q,A=As,l=l,u=u,verbose=False,eps_abs=1e-4,eps_rel=1e-4,max_iter=1400,polishing=False,check_termination=10,adaptive_rho=True)
  else:
   if np.max(np.abs(dense[~self.amask]))>1e-10:raise RuntimeError('Unrepresented contact-QP coefficient')
   self.solver.update(Px=P[self.p_rows,self.p_cols],q=q,Ax=dense[self.a_rows,self.a_cols],l=l,u=u)
  res=self.solver.solve(raise_error=False)
  if res.x is None:raise RuntimeError('Contact gait QP returned no solution: '+res.info.status)
  x=res.x;val=dense@x;violation=float(max(np.max(l-val),np.max(val-u),0.))
  if res.info.status_val not in (1,2) or violation>.08:raise RuntimeError('Contact gait QP rejected '+res.info.status+' residual='+str(violation))
  tau=x[nv:nv+nu];self.last_tau=tau.copy()
  self.last_info=dict(sim_time=float(d.time),compute_ms=1000*(time.perf_counter()-start),solver_status=res.info.status,solver_iter=res.info.iter,
    max_constraint_violation=violation,source_elapsed_s=self.source_elapsed,source_contact=self.source_contact.tolist(),effective_contact=effective.tolist(),actual_contact=actual.tolist(),foot_mode=modes,pre_shift_active=self.pre_shift_active,com_in_support=com_in_support,com_support_local_m=None if self.com_support_local is None else self.com_support_local.tolist(),com_support_local_raw_m=None if self.com_support_raw is None else self.com_support_raw.tolist(),com_support_bounds_m=None if self.com_support_bounds is None else self.com_support_bounds.tolist(),com_support_speed_m_s=self.com_support_speed,
    foot_target_error_m=foot_errors,foot_position_m=d.xpos[self.feet].tolist(),foot_target_position_m=np.asarray(target_ps).tolist(),swing_height_gain=self.swing_height_gain,com_xy=d.subtree_com[0,:2].tolist(),support_center_xy=center.tolist(),com_goal_xy=com_goal.tolist(),transfer_progress=transfer_progress,
    root_height_m=float(d.qpos[2]),root_xy_m=d.qpos[:2].tolist(),root_target_xy_m=target_root_xyz[:2].tolist(),root_xy_error_m=float(np.linalg.norm(d.qpos[:2]-target_root_xyz[:2])),root_xy_task_weight=self.root_xy_weight,desired_q=self.desired[self.body_act].tolist(),actual_q=d.qpos[self.qa[self.body_act]].tolist())
  return tau,self.last_info,True

def _contact_counts(m,d,feet):
 out=[]
 for b in feet:
  count=0
  for c in d.contact:
   if b in m.geom_bodyid[[c.geom1,c.geom2]] and 0 in m.geom_bodyid[[c.geom1,c.geom2]] and c.dist<.003:count+=1
  out.append(count)
 return out

def main():
 from takeover.runtime import config
 import sys
 sys.path.insert(0,str(config()['vendor']))
 from gear_sonic.utils.mujoco_sim import base_sim
 from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import UnitreeSdk2Bridge
 from takeover.range_impedance import tail
 original_body=base_sim.DefaultEnv.compute_body_torques;original_hand=base_sim.DefaultEnv.compute_hand_torques
 original_publish=UnitreeSdk2Bridge.PublishLowState
 def publish(self,obs):
  if getattr(self,'contact_gait_owner',False):
   if obs['time']-getattr(self,'contact_gait_last_publish',-1.)<.0099:return
   self.contact_gait_last_publish=obs['time']
  return original_publish(self,obs)
 UnitreeSdk2Bridge.PublishLowState=publish
 run=Path(os.environ['TAKEOVER_RUN']);log=(run/'contact_qp_gait.jsonl').open('w',buffering=65536)
 controller=None;last_read=0.;ref_t=None;stable_since=None;startup_logged=False;handed_off=False;log_count=0
 request_enabled=(run/'contact_qp_gait.request').exists()
 def body(self):
  nonlocal controller,last_read,ref_t,stable_since,startup_logged,handed_off,log_count
  if not request_enabled:return original_body(self)
  if controller is None:
   band=getattr(self,'elastic_band',None)
   if band is not None and getattr(band,'enable',False):return original_body(self)
   mujoco.mj_forward(self.mj_model,self.mj_data)
   feet=[self.mj_model.body(s+'_ankle_roll_link').id for s in ['left','right']]
   counts=_contact_counts(self.mj_model,self.mj_data,feet)
   stable=(min(counts)>=3 and .74<=self.mj_data.qpos[2]<=.86 and abs(self.mj_data.qvel[2])<.12)
   if not stable:stable_since=None;return original_body(self)
   if stable_since is None:stable_since=float(self.mj_data.time)
   if self.mj_data.time-stable_since<.5:return original_body(self)
   row=tail(run/'debug.jsonl')
   if not row or 'body_q_target' not in row.get('data',{}):return original_body(self)
   try:
    controller=ContactGaitQP(self.mj_model,self.mj_data,self.torque_limit,row['data']['body_q_target'],
       os.environ['PHASEMEND_REFERENCE_DIR'],run/'human_motion_start.json')
   except Exception as exc:
    if not startup_logged:log.write(json.dumps(dict(event='controller_init_wait',error=str(exc)))+'\n');startup_logged=True
    return original_body(self)
   log.write(json.dumps(dict(event='controller_reference_precompute_started',sim_time=float(self.mj_data.time),contact_counts=counts))+'\n')
  if controller.prepare_error:raise RuntimeError('Contact gait QP reference precompute failed: '+controller.prepare_error)
  if not controller.ready:return original_body(self)
  if not handed_off:
   bridge=self.unitree_bridge
   for sub in [bridge.low_cmd_suber,bridge.left_hand_cmd_suber,bridge.right_hand_cmd_suber]:sub.Close()
   bridge.contact_gait_owner=True;handed_off=True
   (run/'motor_authority_handoff.json').write_text(json.dumps(dict(mode='independent_hybrid_contact_QP',nominal_SONIC_motor_commands_bypassed=True,
     source='same SONIC human-retarget target stream; preflight foot contact labels; measured MuJoCo contact transitions'),indent=2)+'\n')
   (run/'contact_qp_gait_anchor.json').write_text(json.dumps(dict(root=controller.root.tolist(),motor_q=controller.anchor.tolist(),
     reference=controller.reference.tolist(),feet=controller.footpos.tolist(),support_bounds=controller.bounds,
     preflight_directory=os.environ['PHASEMEND_REFERENCE_DIR'],source='SONIC initialization followed by independent hybrid contact inverse dynamics QP'),indent=2)+'\n')
   log.write(json.dumps(dict(event='controller_initialized',sim_time=float(self.mj_data.time),contact_modes=controller.modes))+'\n');log.flush()
  now=time.monotonic()
  if now-last_read>=.02:
   row=tail(run/'debug.jsonl');last_read=now
   if row and 'body_q_target' in row.get('data',{}):
    controller.target(row['data']['body_q_target'],row['data'].get('index',0));ref_t=row['t']
  tau,info,updated=controller.solve()
  if updated:
   log.write(json.dumps(dict(t=time.monotonic(),reference_age_s=time.monotonic()-ref_t if ref_t else None,**info))+'\n');log_count+=1
   if log_count%25==0:log.flush()
  return tau[self.body_joint_index-1]
 def hand(self):
  if controller is None or not controller.ready or not handed_off:return original_hand(self)
  return np.r_[controller.last_tau[self.left_hand_index-1],controller.last_tau[self.right_hand_index-1]]
 base_sim.DefaultEnv.compute_body_torques=body;base_sim.DefaultEnv.compute_hand_torques=hand
 try:
  from takeover.simulator import main as sim_main
  sim_main()
 finally:log.close()
if __name__=='__main__':main()









