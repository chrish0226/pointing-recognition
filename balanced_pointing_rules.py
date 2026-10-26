"""RTMW arm rules using a refreshed anatomical torso coordinate frame.

Optional existing head landmarks resolve body heading. Arm decisions use only
shoulders, elbows and wrists. No separate face model is used.
No video-specific timestamps, person IDs, ROI or forced direction completion.
The camera/body reference remains a monocular estimate, not calibration.
"""
from collections import Counter,deque
import math
import numpy as np
from arm_geometry import ForwardGeometry,apply_arm_motion,direction_evidence

REQUIRED=('forward','left','right')
PROFILE='arm_geometry_v9'
DEFAULTS={'smooth_frames':3,'reference_frames':60,'min_reference_frames':12,'min_joint_score':.35,
          'min_wrist_score':.45,'min_mean_score':.50,'raise_enter':-.64,
          'raise_exit':-.76,'min_reach':.18,'min_forearm':.06,
          'max_elbow_angle':157.,'vote_frames':9,'side_votes':4,'forward_votes':4,
          'vote_fraction':.67,'stable_radius':.17,'pose_change':.23,
          'forearm_turn_degrees':30.,'forward_sector':43.,'lateral_sector':50.,
          'full_shoulder_torso_ratio':.70,'max_gap_s':.20,'rest_frames':4,
          'forward_min_lift':-.48,'forward_forearm_rise':.03,
          'raise_history_frames':12,'recent_raise_delta':.14,
          'low_forward_hand_separation':.80,
          'reference_window_frames':12,'reference_stale_s':.6,
          'spatial_reach':.30,'spatial_forearm':.14,
          'forward_min_component':.22,'side_forearm_fraction':.60,
          'side_min_width':.35,'frontal_right_depth_limit':.42,
          'side_min_lateral':.24,'side_min_forearm_lateral':.06,'side_min_fraction':.45,
          'body_turn_window_s':.25,'body_turn_degrees':20.,'weak_after_turn_s':.65,
          'walking_hip_speed':1.05,'walking_foot_speed':.50,
          'orientation_assist':True,'profile_front_settle_s':.22,'forward_height_spread':.12,
          'arm_geometry':True,'vote_max_age_s':.32,'confirmation_min_s':.08,
          'compact_forward_min_s':.12,'bone_forward_min_s':.12}


def unit(v):
    n=float(np.linalg.norm(v))
    return v/n if n>1e-8 else None


def reference_sample(p,q,s):
    if min(s[[5,6,11,12]])<.30:return None
    u2=q[[5,6]].mean(0)-q[[11,12]].mean(0);torso=np.linalg.norm(u2)
    if torso<8:return None
    u2/=torso
    shoulder2=q[6]-q[5];shoulder2-=shoulder2@u2*u2
    hip2=q[12]-q[11];hip2-=hip2@u2*u2
    width=float(np.linalg.norm(shoulder2)/torso)
    if not 0<=width<1.8:return None
    # An isolated anatomical shoulder-label flip must not reverse body left/right.
    if np.linalg.norm(shoulder2)>.10*torso and np.linalg.norm(hip2)>.10*torso and shoulder2@hip2<0:return None
    u3=unit(p[[5,6]].mean(0)-p[[11,12]].mean(0))
    if u3 is None:return None
    shoulder3=p[6]-p[5];shoulder3-=shoulder3@u3*u3
    hip3=p[12]-p[11];hip3-=hip3@u3*u3
    r3=unit(shoulder3);hip_right=unit(hip3)
    if r3 is None:return None
    if hip_right is not None and r3@hip_right<-.1:return None
    if hip_right is not None:r3=unit(.8*r3+.2*hip_right)
    r2=unit(shoulder2)
    if r2 is None:r2=unit(hip2)
    if r2 is None:r2=np.array([-u2[1],u2[0]])
    return {'up2':u2,'right2':r2,'up3':u3,'right3':r3,'width':width,'right_z':p[6,2]-p[5,2]}


def make_reference(samples,config):
    if len(samples)<config['min_reference_frames']:return None
    samples=list(samples)[-config['reference_window_frames']:]
    u2=unit(np.median([a['up2'] for a in samples],axis=0))
    up=unit(np.median([a['up3'] for a in samples],axis=0))
    r2=unit(np.median([a['right2'] for a in samples],axis=0))
    right=unit(np.median([a['right3'] for a in samples],axis=0))
    if r2 is None or up is None or right is None:return None
    width=float(np.median([a['width'] for a in samples]))
    # Broad, frontal shoulder projections contradict extreme predicted shoulder
    # depth. Constrain that depth instead of allowing it to rotate the axes.
    cosine=min(1.,width/config['full_shoulder_torso_ratio'])
    max_depth=math.sqrt(max(0.,1-cosine*cosine))
    # Width only limits implausible depth; it never invents body yaw from a
    # narrowed shoulder projection during an arm movement.
    if abs(right[2])>max_depth and width>=.55:
        xy=unit(right[:2])
        if xy is not None:right=np.r_[xy*math.sqrt(1-max_depth*max_depth),np.sign(right[2])*max_depth]
    right=unit(right-(right@up)*up)
    if right is None:return None
    forward=unit(np.cross(up,right))
    return {'right':right,'up':up,'forward':forward,'right2':r2,'width':width,
            'sample_count':len(samples),'calibrated':False}


def observe_arms(p,q,s,ref,config,previous_active,current_sample=None):
    shoulder=q[[5,6]].mean(0);hip=q[[11,12]].mean(0)
    torso2=float(np.linalg.norm(shoulder-hip));up2=unit(shoulder-hip)
    torso3=float(np.linalg.norm(p[[5,6]].mean(0)-p[[11,12]].mean(0)))
    if up2 is None or torso2<8 or torso3<1e-5:return []
    arms=[]
    for side,sh,el,wr in [('L',5,7,9),('R',6,8,10)]:
        if (not np.isfinite(p[[sh,el,wr]]).all() or
            not np.isfinite(q[[sh,el,wr]]).all() or
            not np.isfinite(s[[sh,el,wr]]).all()):continue
        quality=float(np.mean(s[[sh,el,wr]]));confidence=float(min(s[[sh,el,wr]]))
        visible=(confidence>=config['min_joint_score'] and s[wr]>=config['min_wrist_score'] and quality>=config['min_mean_score'])
        if not visible:continue
        v2=(q[wr]-q[sh])/torso2;upper=(q[el]-q[sh])/torso2;fore=(q[wr]-q[el])/torso2
        lift=float(v2@up2);elbow_lift=float(upper@up2);reach=float(np.linalg.norm(v2))
        cosine=float((-upper@fore)/max(np.linalg.norm(upper)*np.linalg.norm(fore),1e-8))
        elbow_angle=float(np.degrees(np.arccos(np.clip(cosine,-1,1))))
        bent_or_lifted=elbow_angle<=config['max_elbow_angle'] or elbow_lift>-.36
        forearm_level=lift-elbow_lift>=-.16 or elbow_lift>-.36
        v3=(p[wr]-p[sh])/torso3;fore3=(p[wr]-p[el])/torso3
        space_up=ref['up'] if ref else unit(p[[5,6]].mean(0)-p[[11,12]].mean(0))
        horizontal=np.linalg.norm(v3-(v3@space_up)*space_up)
        spatially_extended=(horizontal>=config['spatial_reach'] and np.linalg.norm(fore3)>=config['spatial_forearm'] and lift>=-.45)
        projected_extended=reach>=config['min_reach'] and np.linalg.norm(fore)>=config['min_forearm']
        entry=(lift>=config['raise_enter'] and (projected_extended or spatially_extended) and bent_or_lifted and forearm_level)
        exit_pose=lift<config['raise_exit'] or (lift<-.66 and elbow_angle>165)
        active=bool(entry or (previous_active.get(side,False) and not exit_pose))
        direction=None;heading=None;ambiguity=0.
        direction_scores={};forearm_side=False
        legacy_direction=None;spatial_conflict=False
        if ref is not None and entry:
            v=p[wr]-p[sh];f=p[wr]-p[el]
            vw=np.array([v@ref['right'],v@ref['forward']])/torso3
            fw=np.array([f@ref['right'],f@ref['forward']])/torso3
            v_unit=unit(vw);f_unit=unit(fw)
            if v_unit is not None:
                mixed=.70*v_unit+.30*f_unit if f_unit is not None else v_unit
                heading=float(np.degrees(np.arctan2(mixed[0],mixed[1])))
                lateral2=float(v2@ref['right2'])
                fore_lateral2=float(fore@ref['right2'])
                clear_projected_side=(abs(ref['right'][2])<=config['frontal_right_depth_limit'] and ref['width']>=config['side_min_width'] and abs(lateral2)>=config['side_min_lateral'] and
                    abs(lateral2)/max(reach,1e-8)>=config['side_min_fraction'] and lateral2*fore_lateral2>0 and abs(fore_lateral2)>=config['side_min_forearm_lateral'] and
                    abs(fore_lateral2)/max(np.linalg.norm(fore),1e-8)>=config['side_forearm_fraction'])
                if reach<.36 and spatially_extended and abs(heading)<=config['forward_sector'] and vw[1]>config['forward_min_component']:
                    direction='forward';ambiguity=.5
                elif clear_projected_side:
                    direction='right' if lateral2>0 else 'left';ambiguity=.35
                elif abs(heading)<=config['forward_sector'] and vw[1]>config['forward_min_component']:
                    direction='forward';ambiguity=(config['forward_sector']-abs(heading))/config['forward_sector']
                elif config['lateral_sector']<=abs(heading)<=155 and abs(vw[0])>.15 and horizontal>=config['spatial_reach']:
                    direction='right' if heading>0 else 'left'
                    ambiguity=min(abs(heading)-config['lateral_sector'],155-abs(heading))/45
                if config['arm_geometry']:
                    previous_direction=direction
                    legacy_direction=direction
                    w_body=np.array([v@ref['right'],v@ref['up'],v@ref['forward']])/torso3
                    f_body=np.array([f@ref['right'],f@ref['up'],f@ref['forward']])/torso3
                    scored,direction_scores,score_margin,forearm_side=direction_evidence(
                        w_body,f_body,np.array([lateral2,lift]),
                        np.array([fore_lateral2,lift-elbow_lift]),elbow_angle,ref,side)
                    spatial_conflict=(abs(ref['right'][2])<.45 and elbow_angle>110 and
                                      previous_direction in ('left','right') and scored=='forward' and
                                      abs(heading)<=config['forward_sector'] and
                                      direction_scores['forward']>direction_scores[previous_direction]+.12)
                    if forearm_side or spatial_conflict:
                        direction=scored;ambiguity=score_margin
                    elif (direction is None and scored in ('left','right') and
                          abs(ref['right'][2])<.45 and ref['width']>.35 and
                          reach>.40 and lift>-.66 and
                          lateral2*fore_lateral2>0 and abs(fore_lateral2)>.20 and
                          abs(fore_lateral2)/max(np.linalg.norm(fore),1e-8)>.70 and
                          score_margin>.25):
                        direction=scored;ambiguity=score_margin
        instant_up=current_sample['up3'] if current_sample is not None else (ref['up'] if ref else space_up)
        instant_right=current_sample['right3'] if current_sample is not None else (ref['right'] if ref else np.array([-1.,0.,0.]))
        instant_right=unit(instant_right-(instant_right@instant_up)*instant_up)
        if instant_right is None:
            instant_right=np.zeros(3);instant_forward=np.zeros(3)
        else:instant_forward=unit(np.cross(instant_up,instant_right))
        feature3=np.array([v3@instant_right,v3@instant_up,v3@instant_forward,
                          fore3@instant_right,fore3@instant_up,fore3@instant_forward])
        arms.append({'side':side,'visible':visible,'raised':active,'resting':bool(exit_pose),
            'entry_pose':bool(entry),'direction_stages':{'geometry':direction},
            'direction':direction,'confidence':confidence,'quality':quality,'lift':lift,
            'elbow_lift':elbow_lift,'reach':reach,'elbow_angle':elbow_angle,'heading':heading,
            'margin':max(0.,float(ambiguity)),'feature':np.r_[v2,fore].tolist(),
            'direction_scores':direction_scores,'forearm_side_evidence':forearm_side,
            'legacy_direction':legacy_direction,'spatial_conflict':bool(spatial_conflict),
            'reach_3d_horizontal':float(horizontal),'forearm_3d':float(np.linalg.norm(fore3)),
            'projection_shortening_supported':bool(spatially_extended and not projected_extended),
            'feature_3d':feature3.tolist(),'body_right':instant_right.tolist(),
            'strength':quality*(1+.25*min(max(reach,horizontal),1.5)),'selected':False})
    return arms


def repair_arm_depth(p,q,s):
    """Repair a short raised arm only when bilateral bone consistency improves.

    This tests an alternative depth correspondence; it does not validate Z.
    The original model arrays are preserved.
    """
    p=p.copy()
    torso=np.linalg.norm(q[[5,6]].mean(0)-q[[11,12]].mean(0))
    if torso<8 or min(s[[5,6,7,8,9,10]])<.4:return p
    up=unit(q[[5,6]].mean(0)-q[[11,12]].mean(0))
    lifts=[float((q[w]-q[sh])@up/torso) for sh,w in ((5,9),(6,10))]
    reaches=[float(np.linalg.norm(q[w]-q[sh])/torso) for sh,w in ((5,9),(6,10))]
    if not any(lifts[i]>-.2 and reaches[i]<.36 and lifts[1-i]<-.7 for i in (0,1)):return p
    metric=p.copy();metric[:,:2]=(q-q[[11,12]].mean(0))*.5/torso
    swapped=metric.copy()
    for sh,el,wr,osh,oel,owr in ((5,7,9,6,8,10),(6,8,10,5,7,9)):
        swapped[[el,wr],2]=metric[[oel,owr],2]-metric[osh,2]+metric[sh,2]
    def loss(v):
        return sum(float(np.sum(np.log(np.maximum([np.linalg.norm(v[el]-v[sh]),np.linalg.norm(v[wr]-v[el])],1e-5)/[.275,.25])**2)) for sh,el,wr in ((5,7,9),(6,8,10)))
    original_cost=loss(metric);swap_cost=loss(swapped)
    if original_cost-swap_cost>.75 and swap_cost<.5*original_cost:p[[7,8,9,10],2]=swapped[[7,8,9,10],2]
    return p


class HeadingFilter:
    def __init__(self):
        self.widths=deque(maxlen=90);self.depths=deque(maxlen=60)
        self.depth_sign=None;self.right2=None;self.ready=False;self.cosine=1.
    def observe(self,p,q,s,sample,t):
        # Learn this person's frontal proportions; head points remain optional.
        up2=sample['up2'];size=np.linalg.norm(q[[5,6]].mean(0)-q[[11,12]].mean(0))
        hip=q[12]-q[11];hip-=hip@up2*up2
        hip_width=np.linalg.norm(hip)/size;sw=sample['width']
        ears_ok=min(s[[3,4]])>=.45 and np.isfinite(q[[0,3,4]]).all() and s[0]>=.5
        lateral=np.array([up2[1],-up2[0]])
        offset=float((q[0]-q[[3,4]].mean(0))@lateral/size) if ears_ok else 0.
        ears_width=np.linalg.norm(q[3]-q[4])/size if ears_ok else 0.
        frontal=(ears_ok and abs(offset)<.04 and ears_width>.18 and sw>.5)
        if frontal:self.widths.append((sw,hip_width))
        self.ready=len(self.widths)>=12
        if not self.ready:return sample
        shoulder_full=float(np.percentile([v[0] for v in self.widths],85))
        hip_full=float(np.percentile([v[1] for v in self.widths],85))
        if hip_full<.15 or shoulder_full<.45:return sample
        shoulder_ratio=sw/shoulder_full;hip_ratio=hip_width/hip_full
        cosine=float(np.clip(max(shoulder_ratio,hip_ratio),0,1))
        self.cosine=cosine
        if cosine>=.94:return sample
        sd=p[6,2]-p[5,2];hd=p[12,2]-p[11,2]
        while self.depths and t-self.depths[0][0]>1.5:self.depths.popleft()
        if abs(sd)>.07 and abs(hd)>.035 and sd*hd>0:self.depths.append((t,np.sign(sd),abs(sd)+abs(hd)))
        if self.depths:
            signed=sum(v[1]*v[2] for v in self.depths)
            if abs(signed)>.08:self.depth_sign=1. if signed>0 else -1.
        if self.depth_sign is None:self.depth_sign=1. if sample['right3'][2]>=0 else -1.
        if ears_ok and abs(offset)>.035:self.depth_sign=1. if offset>0 else -1.
        r2=sample['right2']
        if cosine<.65 and ears_ok:
            ear_right=q[4]-q[3];ear_right-=ear_right@up2*up2
            if np.linalg.norm(ear_right)>.04*size and ear_right@r2<0:r2=-r2
        if self.right2 is not None and sw<.10 and hip_width<.10:r2=self.right2
        elif sw>.10:self.right2=r2.copy()
        right=np.r_[r2*cosine,self.depth_sign*np.sqrt(max(0.,1-cosine*cosine))]
        right=unit(right-(right@sample['up3'])*sample['up3'])
        if right is None:return sample
        return {**sample,'right3':right}


class BalancedPointing:
    def __init__(self,config=None):
        self.config={**DEFAULTS,**(config or {})};self.people={}
    def _state(self,tid,time_s):
        if int(tid) in self.people:return self.people[int(tid)]
        return self.people.setdefault(int(tid),{'done':{},'passed_s':None,'events':[],
            'natural_history':{side:deque(maxlen=128) for side in ('L','R')},
            'forward_geometry':ForwardGeometry(),'forward_votes':{side:deque(maxlen=128) for side in ('L','R')},
            'profile_front_since':{side:None for side in ('L','R')},'heading_filter':HeadingFilter(),'alternate_refs':deque(maxlen=12),'last_frame':None,'last_seen_s':time_s,'last_reference_s':None,
            'reference_history':deque(maxlen=60),'weak_pose_after_s':0.,'motion_history':deque(maxlen=60),
            'samples':deque(maxlen=self.config['smooth_frames']),
            'references':deque(maxlen=self.config['reference_frames']),'reference':None,
            'provisional':None,'active':{},'votes':deque(maxlen=max(self.config['vote_frames'],128)),
            'lift_history':{side:deque(maxlen=self.config['raise_history_frames']) for side in ('L','R')},
            'phase':None,'rest_count':0,'last_observation':None})
    def snapshot(self,tid):
        st=self.people[int(tid)]
        return {'track_id':int(tid),'done':list(st['done']),'passed':st['passed_s'] is not None,'passed_s':st['passed_s']}
    def _end_phase(self,st,frame,t,reason):
        if st['phase']:
            st['events'].append({**st['phase'],'ended_s':t,'end_frame':frame,'end_reason':reason})
        st['phase']=None;st['votes'].clear()
        for pool in st['forward_votes'].values():pool.clear()
    def _stable_candidate(self,votes,phase=None,bone_duration=None):
        cfg=self.config
        if not votes or not votes[-1].get('direction'):return None
        now=votes[-1]['time_s']
        recent_votes=[v for v in votes if now-v.get('time_s',-math.inf)<=cfg['vote_max_age_s']]
        times=[v['time_s'] for v in recent_votes[-12:] if 'time_s' in v]
        intervals=np.diff(times)
        intervals=intervals[intervals>1e-6]
        interval=float(np.median(intervals)) if len(intervals) else 1/30
        window_duration=cfg['confirmation_min_s']
        if votes[-1].get('compact_forward'):window_duration=max(window_duration,cfg['compact_forward_min_s'])
        if votes[-1].get('bone_forward'):
            window_duration=max(window_duration,cfg['bone_forward_min_s'] if bone_duration is None else bone_duration)
        window=max(cfg['vote_frames'],int(math.ceil(window_duration/interval))+3)
        recent_votes=recent_votes[-window:]
        known=[v for v in recent_votes if v['direction']]
        if phase:known=[v for v in known if v['direction']!=phase['direction']]
        if not known:return None
        counts=Counter(v['direction'] for v in known);direction,count=counts.most_common(1)[0]
        if votes[-1]['direction']!=direction:return None
        required=cfg['forward_votes'] if direction=='forward' else cfg['side_votes']
        tail=recent_votes[-3:]
        # Three stable frames can resolve a brief, extended spatial gesture.
        # This exception requires agreement in all three observations, rather
        # than lowering the global vote count for preparation movements.
        if (direction=='forward' and phase and len(tail)==3 and
            all(v.get('direction')==direction and v.get('spatial_conflict') and v.get('margin',0.)>.16 and
                v.get('quality',0.)>.60 for v in tail) and
            tail[-1]['time_s']-tail[0]['time_s']>=.06):required=3
        min_duration=.06 if required==3 else cfg['confirmation_min_s']
        if direction=='forward' and any(v.get('compact_forward') for v in known[-3:]):
            min_duration=max(min_duration,cfg['compact_forward_min_s'])
        if direction=='forward' and any(v.get('bone_forward') for v in known[-3:]):
            min_duration=max(min_duration,cfg['bone_forward_min_s'] if bone_duration is None else bone_duration)
        required=max(required,int(math.ceil(min_duration/interval-1e-6))+1)
        if count<required or count/len(known)<cfg['vote_fraction']:return None
        support=[v for v in known if v['direction']==direction]
        if direction=='forward' and (votes[-1].get('compact_forward') or votes[-1].get('bone_forward')):
            run=[]
            for v in reversed(recent_votes):
                if v.get('direction')!='forward' or not (v.get('compact_forward') or v.get('bone_forward')):break
                run.append(v)
            if len(run)<required:return None
            support=list(reversed(run))
        if direction=='forward':support=support[-required:]
        if support[-1]['time_s']-support[0]['time_s']<min_duration-1e-8:return None
        recent=support[-3:]
        if len(recent)>=3:
            dt=recent[-1]['time_s']-recent[0]['time_s']
            change=np.asarray(recent[-1]['feature'])[:2]-np.asarray(recent[0]['feature'])[:2]
            assisted_side=direction in ('left','right') and any(v.get('orientation_agrees') or v.get('orientation_preserved') for v in recent)
            compact_vote=any(v.get('compact_forward') or v.get('bone_forward') for v in recent)
            speed=(np.linalg.norm(change) if compact_vote or (phase is None and (direction=='forward' or assisted_side)) else abs(change[1]))/max(dt,1e-8)
            gate=(direction=='forward' or any(v.get('lowering_exempt') for v in recent) or
                  (phase is None and (recent[-1]['lift']<-.15 or assisted_side)))
            if gate and dt>0 and speed>1.0:return None
            if (assisted_side and
                np.ptp([v['elbow_angle'] for v in recent])>12.):return None
        if direction=='forward' and np.ptp([v['lift'] for v in support[-required:]])>cfg['forward_height_spread']:return None
        if len({v['side'] for v in support})!=1:return None
        features=np.asarray([v['feature'] for v in support]);median=np.median(features,axis=0)
        spread=np.linalg.norm(features[:,:2]-median[:2],axis=1)
        if np.percentile(spread,80)>cfg['stable_radius']:return None
        body_right=unit(np.median([v['body_right'] for v in support],axis=0))
        if body_right is None:return None
        return {'direction':direction,'side':support[-1]['side'],'support':count,
                'lift':float(np.median([v['lift'] for v in support])),
                'recent_raise':any(v['recent_raise'] for v in support),
                'natural_transition':sum(bool(v.get('natural_transition')) for v in support)>=required,
                'compact_forward':sum(bool(v.get('compact_forward')) for v in support)>=required,
                'bone_forward':sum(bool(v.get('bone_forward')) for v in support)>=required,
                'direction_margin':float(np.mean([v.get('margin',0.) for v in support])),
                'forward_score':float(np.mean([v.get('direction_scores',{}).get('forward',0.) for v in support])),
                'feature':median.tolist(),'start_frame':support[0]['frame'],'started_s':support[0]['time_s'],
                'feature_3d':np.median([v['feature_3d'] for v in support],axis=0).tolist(),
                'body_right':body_right.tolist(),
                'quality':float(np.mean([v['quality'] for v in support]))}
    def _is_new_pose(self,phase,candidate):
        if phase['side']==candidate['side'] and 'feature_3d' in phase:
            old3=np.asarray(phase['feature_3d']);new3=np.asarray(candidate['feature_3d'])
            turn_body=float(np.degrees(np.arccos(np.clip(np.asarray(phase['body_right'])@candidate['body_right'],-1,1))))
            wrist3=float(np.linalg.norm(new3[:3]-old3[:3]))
            old_fore=unit(old3[3:]);new_fore=unit(new3[3:]);fore_turn=0.
            if old_fore is not None and new_fore is not None:
                fore_turn=float(np.degrees(np.arccos(np.clip(old_fore@new_fore,-1,1))))
            # Rotating the body with an unchanged body-relative arm pose is
            # not a second arm gesture, even though its image projection moves.
            if turn_body>15 and wrist3<.15 and fore_turn<20:return False
            if (candidate['direction']=='forward' and turn_body<15 and
                candidate.get('forward_score',0.)>.75 and candidate.get('direction_margin',0.)>.16 and
                new3[2]-old3[2]>.30):return True
        if candidate.get('natural_transition'):
            return True
        # Lowering a lateral arm often projects briefly as forward. A forward
        # gesture after a lateral one requires a new raise, separated by rest.
        if phase['direction'] in ('left','right') and candidate['direction']=='forward':
            # A new spatial pose may turn forward without raising higher.
            if not candidate['recent_raise'] and candidate['lift']<phase['lift']+.06:return False
        if phase['side']!=candidate['side']:return True
        a=np.asarray(phase['feature']);b=np.asarray(candidate['feature'])
        wrist_change=float(np.linalg.norm(a[:2]-b[:2]))
        ua=unit(a[2:]);ub=unit(b[2:]);turn=0.
        if ua is not None and ub is not None:turn=float(np.degrees(np.arccos(np.clip(ua@ub,-1,1))))
        return wrist_change>=self.config['pose_change'] or (wrist_change>=.12 and turn>=self.config['forearm_turn_degrees'])
    def update(self,tid,frame,time_s,xyz,points_xy,scores):
        cfg=self.config;st=self._state(tid,time_s)
        all_p=np.asarray(xyz,float);all_q=np.asarray(points_xy,float);all_s=np.asarray(scores,float)
        p=all_p[:17];q=all_q[:17];s=all_s[:17]
        if p.shape!=(17,3) or q.shape!=(17,2) or s.shape!=(17,):raise ValueError('Expected COCO body points')
        if st['last_frame'] is not None and (time_s-st['last_seen_s']>cfg['max_gap_s'] or frame<=st['last_frame']):
            self._end_phase(st,st['last_frame'],st['last_seen_s'],'TRACK_GAP')
            st['samples'].clear();st['active']={};st['rest_count']=0
            st['references'].clear();st['reference']=None;st['last_reference_s']=None
            st['reference_history'].clear();st['motion_history'].clear()
            st['weak_pose_after_s']=0.
            st['alternate_refs'].clear();st['profile_front_since']={side:None for side in ('L','R')}
            st['heading_filter'].depths.clear();st['heading_filter'].depth_sign=None;st['heading_filter'].right2=None
            for history in st['lift_history'].values():history.clear()
            for history in st['natural_history'].values():history.clear()
            st['forward_geometry']=ForwardGeometry()
        st['last_frame']=frame;st['last_seen_s']=time_s
        torso_ids=[5,6,11,12]
        if (not np.isfinite(p[torso_ids]).all() or not np.isfinite(q[torso_ids]).all() or
            not np.isfinite(s[torso_ids]).all() or min(s[torso_ids])<.30):
            for pool in st['forward_votes'].values():pool.append({'direction':None,'time_s':time_s})
            st['votes'].append({'direction':None,'time_s':time_s});return self.snapshot(tid),[]
        p=repair_arm_depth(p,q,s)
        st['samples'].append((p,q,s))
        # Short median rejects one-frame keypoint spikes without a long delay.
        p=np.median([v[0] for v in st['samples']],axis=0)
        q=np.median([v[1] for v in st['samples']],axis=0)
        s=np.minimum(s,np.median([v[2] for v in st['samples']],axis=0))
        sample=reference_sample(p,q,s)
        if sample:
            st['references'].append(sample)
            alternate=st['heading_filter'].observe(p,q,s,sample,time_s)
            st['alternate_refs'].append(alternate)
            current_reference=make_reference(st['references'],cfg)
            if current_reference is not None:
                st['reference']=current_reference;st['last_reference_s']=time_s
        ref=st['reference'] if st['last_reference_s'] is not None and time_s-st['last_reference_s']<=cfg['reference_stale_s'] else None
        reference_history=st['reference_history']
        while reference_history and time_s-reference_history[0][0]>cfg['body_turn_window_s']:reference_history.popleft()
        if ref is not None:
            if reference_history and time_s-reference_history[0][0]>=.10:
                turn=float(np.degrees(np.arccos(np.clip(ref['right']@reference_history[0][1],-1,1))))
                projection_change=abs(ref['width']-reference_history[0][2])>.08
                if turn>=cfg['body_turn_degrees'] and projection_change:st['weak_pose_after_s']=time_s+cfg['weak_after_turn_s']
            reference_history.append((time_s,ref['right'].copy(),ref['width']))
        recent_reference=[v for v in reference_history if time_s-v[0]<=.16]
        reference_settled=bool(ref is not None and len(recent_reference)>=4 and
            time_s-recent_reference[0][0]>=.10 and
            np.ptp([v[2] for v in recent_reference])<.045 and
            min(float(ref['right']@v[1]) for v in recent_reference)>math.cos(math.radians(8)))
        arms=observe_arms(p,q,s,ref,cfg,st['active'],sample)
        alternate=make_reference(st['alternate_refs'],cfg)
        tracker=st['heading_filter']
        for arm in arms:
            if tracker.ready and arm.get('spatial_conflict'):
                arm['direction']=arm['legacy_direction']
        if cfg['orientation_assist'] and ref is not None and tracker.ready and alternate is not None and tracker.cosine<.94:
            alternatives=observe_arms(p,q,s,alternate,{**cfg,'arm_geometry':False},st['active'],sample)
            shoulder=q[[5,6]].mean(0);hip=q[[11,12]].mean(0)
            size=np.linalg.norm(shoulder-hip);up=unit(shoulder-hip)
            head_ok=s[0]>=.50 and min(s[[3,4]])>=.20 and max(s[[3,4]])>=.65 and np.isfinite(q[[0,3,4]]).all()
            raw_width=np.linalg.norm((q[6]-q[5])-((q[6]-q[5])@up)*up)/size
            if head_ok:
                face=q[0]-q[[3,4]].mean(0);face-=face@up*up
                ear_right=q[4]-q[3];ear_right-=ear_right@up*up
                body_right2=ref['right2'] if ref else sample['right2']
                coherent=ear_right@body_right2>0 and np.linalg.norm(ear_right)>.06*size
                for arm in arms:
                    alt=next((a for a in alternatives if a['side']==arm['side']),None)
                    if alt is None:continue
                    sh,el,wr=(5,7,9) if arm['side']=='L' else (6,8,10)
                    fore=(q[wr]-q[el])/size;v=(q[wr]-q[sh])/size
                    h=np.array([up[1],-up[0]])
                    follows_face=(v@h)*(face@h)>0 and abs(face@h)>.035*size
                    horizontal=abs(fore@h);fraction=horizontal/max(np.linalg.norm(fore),1e-8)
                    profile=bool(tracker.cosine<.80 and raw_width<.40)
                    medium=(not profile and coherent and arm['reach']<.70 and raw_width<.60)
                    if not follows_face or not (profile or medium):continue
                    if profile and (horizontal<.12 or fraction<.55 or (fore@h)*(face@h)<=0):continue
                    dz=float(p[wr,2]-p[sh,2])
                    depth_agrees=dz*alternate['forward'][2]>0 and abs(dz)>.06
                    if medium and not depth_agrees and abs(dz)>.06:continue
                    if (arm['direction'] in ('left','right') and
                        alt['direction']==arm['direction']):
                        arm['orientation_agrees']=True
                        continue
                    # Preserve a clear bent lateral arm in a broad torso view.
                    # The alternative heading must not erase this solely from
                    # a head turn; transitional poses still need stable votes.
                    clear_side=(arm['direction'] in ('left','right') and
                        ref['width']>.35 and abs(ref['right'][2])<.45 and
                        arm['elbow_angle']<135 and arm['reach']>.40 and
                        arm.get('direction_scores',{}).get(arm['direction'],0.)>.75 and
                        horizontal>.18 and fraction>.65)
                    if clear_side:
                        arm['orientation_preserved']=True
                        continue
                    # Heading is only an aid. A folded arm with no forward
                    # spatial support cannot become forward solely because
                    # its wrist happens to lie on the face-facing side.
                    if (alt['direction']=='forward' and arm['elbow_angle']<135 and
                        arm['direction']!='forward' and
                        arm.get('direction_scores',{}).get('forward',0.)<.50):
                        arm['direction']=None
                        continue
                    arm['direction']=alt['direction'] if alt['direction']=='forward' else None
                    arm['orientation_assisted']=True
                    arm['profile_front']=profile and arm['direction']=='forward'

        for arm in arms:arm['direction_stages']['heading']=arm['direction']
        if cfg['arm_geometry']:
            apply_arm_motion(arms,p,q,ref,st['phase'],st['natural_history'],time_s)
            st['forward_geometry'].observe(arms,q,ref,st['phase'],time_s)
        for arm in arms:arm['direction_stages']['temporal']=arm['direction']

        torso_size=float(np.linalg.norm(q[[5,6]].mean(0)-q[[11,12]].mean(0)))
        hip=q[[11,12]].mean(0)
        feet=(q[[15,16]]-hip)/max(torso_size,1.)
        foot_scores=s[[15,16]].copy()
        history=st['motion_history']
        while history and time_s-history[0][0]>.24:history.popleft()
        walking=False;hip_speed=0.;foot_speed=0.
        if history and time_s-history[0][0]>=.12 and np.isfinite(feet).all():
            old_t,old_hip,old_feet,old_size,old_scores=history[0]
            dt=time_s-old_t
            valid=(np.minimum(foot_scores,old_scores)>=.45)
            hip_speed=float(np.linalg.norm(hip-old_hip)/max((torso_size+old_size)*.5,1.)/dt)
            if np.any(valid):
                foot_speed=float(np.median(np.linalg.norm(feet[valid]-old_feet[valid],axis=1))/dt)
                walking=hip_speed>=cfg['walking_hip_speed'] and foot_speed>=cfg['walking_foot_speed']
        if np.isfinite(feet).all():history.append((time_s,hip.copy(),feet.copy(),torso_size,foot_scores))
        hands_close=(min(s[[9,10]])>=cfg['min_wrist_score'] and
            np.linalg.norm(q[9]-q[10])<cfg['low_forward_hand_separation']*torso_size and
            ref is not None and ref['width']>=.55)
        both_hands_raised=all((q[wr]-q[sh])@(q[[5,6]].mean(0)-hip)/max(torso_size**2,1.)>-.72
                             for sh,wr in ((5,9),(6,10)))
        for arm in arms:
            arm['rejections']=[]
            if (arm['direction']=='forward' and st['phase'] and
                st['phase']['direction'] in ('left','right') and not tracker.ready and
                not arm.get('natural_transition') and arm.get('direction_scores') and
                arm['direction_scores']['forward']-max(arm['direction_scores']['left'],arm['direction_scores']['right'])<.16):
                arm['direction']=None;arm['rejections'].append('AMBIGUOUS_SPATIAL_DIRECTION')
            history=st['lift_history'][arm['side']]
            while history and frame-history[0][0]>cfg['raise_history_frames']:history.popleft()
            low=min((lift for _,lift in history),default=arm['lift'])
            arm['recent_raise']=bool(low<=cfg['raise_enter'] and arm['lift']-low>=cfg['recent_raise_delta'])
            peak=max((lift for _,lift in history),default=arm['lift'])
            falling=(len(history)>=3 and arm['lift']<history[-3][1]-.08)
            history.append((frame,arm['lift']))
            if st['phase'] and st['phase']['side']==arm['side']:
                st['phase']['peak_lift']=max(st['phase'].get('peak_lift',st['phase']['lift']),arm['lift'])
                phase_peak=st['phase']['peak_lift']
            else:phase_peak=arm['lift']
            wrist=np.asarray(arm['feature'])[:2]
            fore=np.asarray(arm['feature'])[2:]
            side_extension=(arm['direction'] in ('left','right') and arm['reach']>.40 and
                abs(wrist@ref['right2'])>.22 and abs(fore@ref['right2'])>.18 and
                abs(fore@ref['right2'])/max(np.linalg.norm(fore),1e-8)>.65 and
                arm['elbow_angle']>55 and arm['lift']>-.66 and
                arm.get('direction_scores',{}).get(arm['direction'],0.)>.70) if ref is not None else False
            arm['lowering_exempt']=bool(side_extension and st['phase'] and
                arm['side']==st['phase']['side'] and arm['direction']!=st['phase']['direction'])
            if st['phase'] and not arm['lowering_exempt'] and arm['direction']!=st['phase']['direction'] and arm['lift']<phase_peak-.12 and falling:
                arm['direction']=None;arm['rejections'].append('LOWERING_FROM_POINT_PEAK')
            if st['phase'] and not arm['lowering_exempt'] and arm['direction']!=st['phase']['direction'] and arm['lift']<st['phase']['lift']-.12 and falling:
                arm['direction']=None;arm['rejections'].append('LOWERING_AFTER_POINT')
            if arm.get('orientation_assisted') and arm['direction']=='forward' and st['phase'] and st['phase']['direction']!='forward' and arm['lift']<peak-.10:
                arm['direction']=None;arm['rejections'].append('LOWERING_AFTER_POINT')
            # A low forward-looking arm needs evidence of an actual raise;
            # static glove handling around the waist does not provide it.
            if (arm['direction']=='forward' and arm['lift']<cfg['forward_min_lift'] and
                arm['lift']-arm['elbow_lift']<cfg['forward_forearm_rise'] and
                (not arm['recent_raise'] or hands_close)):
                arm['direction']=None;arm['rejections'].append('LOW_FORWARD_WITHOUT_EXTENSION')
            weak_low_pose=(arm['lift']<-.28 and arm['elbow_lift']<-.30 and
                (arm['reach']<.70 or arm['elbow_angle']<120))
            deliberate_side=(arm.get('forearm_side_evidence') and not arm.get('bimanual_suppressed') and
                             (arm['recent_raise'] or st['phase'] is not None))
            scores3=arm.get('direction_scores',{})
            strong_side=(arm['direction'] in ('left','right') and arm['recent_raise'] and
                         scores3.get(arm['direction'],0.)>.80 and
                         abs(arm['feature_3d'][3])>.30 and arm['reach']>.60)
            strong_front=(arm['direction']=='forward' and not hands_close and
                          (arm['recent_raise'] or st['phase'] is not None) and
                          arm['lift']>-.48 and arm['lift']-arm['elbow_lift']>.08 and
                          scores3.get('forward',0.)>.78 and
                          scores3.get('forward',0.)-max(scores3.get('left',0.),scores3.get('right',0.))>.16 and
                          arm['feature_3d'][2]>.30 and arm['feature_3d'][5]>.25)
            deliberate=deliberate_side or strong_side or strong_front
            arm['walking_exemption']=bool(walking and weak_low_pose and deliberate)
            blocked_walking=walking and not deliberate
            blocked_reference=time_s<st['weak_pose_after_s']
            settled_point=(reference_settled and not walking and not (hands_close and both_hands_raised) and
                arm['direction'] in ('left','right') and arm['recent_raise'] and
                arm['quality']>.65 and arm['lift']>-.60 and arm['reach']>.40 and
                arm['lift']-arm['elbow_lift']>-.10 and
                abs(fore@ref['right2'])>.25 and
                abs(fore@ref['right2'])/max(np.linalg.norm(fore),1e-8)>.80 and
                scores3.get(arm['direction'],0.)>.80) if ref is not None else False
            arm['settling_exempt']=bool(blocked_reference and settled_point)
            blocked_reference=blocked_reference and not settled_point
            if weak_low_pose and (blocked_walking or blocked_reference):
                arm['direction']=None
                if blocked_walking:arm['rejections'].append('WALKING_WEAK_ARM')
                if blocked_reference:arm['rejections'].append('BODY_REFERENCE_SETTLING')
            timer=st['profile_front_since']
            if arm['direction']=='forward' and arm.get('profile_front'):
                if timer[arm['side']] is None:timer[arm['side']]=time_s
                if time_s-timer[arm['side']]<cfg['profile_front_settle_s']:
                    arm['direction']=None;arm['rejections'].append('PROFILE_DIRECTION_SETTLING')
            else:timer[arm['side']]=None
            arm['direction_stages']['filtered']=arm['direction']
            if arm['direction'] is None and not arm['rejections']:
                if ref is None:reason='BODY_REFERENCE_NOT_READY'
                elif not arm['entry_pose']:reason='ARM_POSE_NOT_ELIGIBLE'
                elif (arm['direction_stages']['geometry'] is not None and
                      arm['direction_stages']['heading'] is None):reason='ORIENTATION_REJECTED_DIRECTION'
                else:reason='AMBIGUOUS_DIRECTION'
                arm['rejections'].append(reason)
        for side in ('L','R'):
            if not any(a['side']==side for a in arms):st['profile_front_since'][side]=None
        st['active']={a['side']:a['raised'] for a in arms}
        candidates=[a for a in arms if a['raised'] and a['direction']]
        # Prefer the current hand so a hidden second hand cannot add a direction.
        side=st['phase']['side'] if st['phase'] else None
        candidates.sort(key=lambda a:a['strength']*(1.25 if a['side']==side else 1),reverse=True)
        selected=candidates[0] if candidates else None
        if len(candidates)>1 and candidates[0]['direction']!=candidates[1]['direction'] and candidates[0]['strength']<1.20*candidates[1]['strength']:
            selected=None
        if any(a['raised'] for a in arms) and st['provisional'] is None and st['reference'] is None:
            st['provisional']=ref
        if selected:
            selected['selected']=True
            st['votes'].append({**selected,'frame':frame,'time_s':time_s})
        else:st['votes'].append({'direction':None,'time_s':time_s})
        for arm_side,pool in st['forward_votes'].items():
            arm=next((a for a in arms if a['side']==arm_side),None)
            eligible=bool(arm and arm.get('bone_forward_proposal') and st['phase'] and
                arm['lift']>st['phase'].get('peak_lift',st['phase']['lift'])-.12 and
                not any(r in arm['rejections'] for r in ('WALKING_WEAK_ARM','BODY_REFERENCE_SETTLING',
                                                        'LOW_FORWARD_WITHOUT_EXTENSION')))
            if eligible:
                pool.append({**arm,'direction':'forward','natural_transition':True,'bone_forward':True,
                             'frame':frame,'time_s':time_s})
            else:pool.append({'direction':None,'time_s':time_s})
        phase=st['phase'];active_side=phase['side'] if phase else None
        tracked=next((a for a in arms if a['side']==active_side),None)
        resting=bool((tracked and tracked['resting']) or (len(arms)==2 and all(a['resting'] for a in arms)))
        st['rest_count']=st['rest_count']+1 if resting else 0
        if st['rest_count']>=cfg['rest_frames']:
            self._end_phase(st,frame,time_s,'ARM_LOWERED');st['provisional']=None
        candidate=self._stable_candidate(st['votes'],st['phase'])
        if st['phase'] and st['phase']['direction'] in ('left','right'):
            forward_pool=st['forward_votes'][st['phase']['side']]
            inferred=self._stable_candidate(forward_pool,st['phase'])
            if inferred:candidate=inferred
            elif candidate and candidate['direction'] in ('left','right') and self._stable_candidate(
                    forward_pool,st['phase'],bone_duration=cfg['confirmation_min_s']):
                # A stable geometric front hypothesis needs a little longer
                # than a side vote. Allow that bounded observation time so a
                # flat-depth lateral label cannot win one frame earlier.
                candidate=None
        if candidate and (phase is None or (candidate['direction']!=phase['direction'] and self._is_new_pose(phase,candidate))):
            if phase:self._end_phase(st,frame,time_s,'STABLE_NEW_POSE')
            phase={**candidate,'confirmed_frame':frame,'confirmed_s':time_s}
            st['phase']=phase
            if st['reference'] is None:st['reference']=ref
            direction=candidate['direction']
            if direction not in st['done']:
                st['done'][direction]={'frame':frame,'time_s':time_s,'gesture_started_s':candidate['started_s']}
            if len(st['done'])==3 and st['passed_s'] is None:st['passed_s']=time_s
        st['last_observation']={'arms':arms,'reference_ready':ref is not None,
            'reference_width':ref['width'] if ref else None,'walking':walking,
            'hip_speed':hip_speed,'foot_speed':foot_speed,
            'reference_settling':time_s<st['weak_pose_after_s'],
            'orientation_assist_ready':tracker.ready,'orientation_cosine':tracker.cosine,
            'orientation_reference':{key:value.tolist() if isinstance(value,np.ndarray) else value for key,value in alternate.items()} if alternate is not None else None}
        return self.snapshot(tid),arms
    def close(self):
        for st in self.people.values():self._end_phase(st,st['last_frame'],st['last_seen_s'],'VIDEO_OR_TRACK_END')
    def summary(self):
        result=[]
        for tid,st in self.people.items():
            ref=st['reference'];ref_json={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in ref.items()} if ref else None
            result.append({'track_id':tid,'done':st['done'],'passed':st['passed_s'] is not None,
                           'passed_s':st['passed_s'],'events':st['events'],'reference':ref_json})
        return result
