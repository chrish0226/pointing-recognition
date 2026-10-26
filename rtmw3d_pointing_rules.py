"""3D direction adapter; the existing Completion state machine is unchanged."""
import numpy as np


def unit(v):
    length=float(np.linalg.norm(v))
    return v/length if length>1e-8 else None


def arm_directions_3d(xyz,points_xy,scores,keypoint_conf=.3):
    p=np.asarray(xyz,float)[:17];q=np.asarray(points_xy,float)[:17];s=np.asarray(scores,float)[:17]
    if p.shape!=(17,3) or q.shape!=(17,2) or s.shape!=(17,):raise ValueError('Expected COCO body subset')
    if not np.isfinite(p).all() or not np.isfinite(q).all() or not np.isfinite(s).all():return set(),[]
    if min(s[[5,6,11,12]])<keypoint_conf:return set(),[]
    shoulder=p[[5,6]].mean(0);hip=p[[11,12]].mean(0);torso=float(np.linalg.norm(shoulder-hip))
    up=unit(shoulder-hip)
    if up is None or torso<1e-5:return set(),[]
    r=p[6]-p[5];r-=float(r@up)*up;right=unit(r)
    if right is None or np.linalg.norm(r)<.15*torso:return set(),[]
    # Camera XYZ: x right, y down, z away. For a front-facing person,
    # anatomical right points image-left and forward points toward camera (-z).
    forward=unit(np.cross(up,right))
    torso2=float(np.linalg.norm(q[[5,6]].mean(0)-q[[11,12]].mean(0)))
    up2=unit(q[[5,6]].mean(0)-q[[11,12]].mean(0))
    if up2 is None or torso2<1:return set(),[]
    arms=[]
    for side,sh,el,wr in [('L',5,7,9),('R',6,8,10)]:
        conf=float(min(s[[sh,el,wr]]))
        if conf<keypoint_conf:continue
        v=p[wr]-p[sh];fore=p[wr]-p[el];v2=q[wr]-q[sh]
        lift2=float(v2@up2/torso2);extension2=float(np.linalg.norm(v2)/torso2)
        raised=lift2>-.55 and extension2>.18 and np.linalg.norm(q[wr]-q[el])>.10*torso2
        lateral=float(v@right/torso);front=float(v@forward/torso)
        fore_right=float(fore@right/torso);fore_front=float(fore@forward/torso)
        heading=float(np.degrees(np.arctan2(fore_right,fore_front)))
        direction=None
        if raised and conf>=.60:
            if 60<=abs(heading)<=120 and abs(lateral)>=.25 and lateral*fore_right>0:
                direction='right' if fore_right>0 else 'left'
            elif abs(heading)<=35 and front>=.20 and fore_front>=.15:
                direction='forward'
        strength=float(np.linalg.norm(v)/torso*conf)
        arms.append({'side':side,'direction':direction,'raised':bool(raised),'confidence':conf,
            'selected':False,'strength':strength,'lift':lift2,'extension':extension2,
            'body_lateral':lateral,'body_forward':front,'forearm_lateral':fore_right,
            'forearm_forward':fore_front,'heading_degrees':heading})
    candidates=[a for a in arms if a['direction'] in ('left','right')]
    if not candidates:candidates=[a for a in arms if a['direction']=='forward']
    candidates.sort(key=lambda a:a['strength'],reverse=True)
    if not candidates:return set(),arms
    best=candidates[0]
    conflict=next((a for a in candidates[1:] if a['direction']!=best['direction']),None)
    if conflict and best['strength']<1.3*conflict['strength']:return set(),arms
    best['selected']=True
    return {best['direction']},arms
