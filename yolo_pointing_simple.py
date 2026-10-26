"""YOLO-only person-relative front/left/right arm-raise accumulation.

No region, face model, head confirmation, crossing line, timeout, or old rules.
YOLO supplies 2D keypoints, so forward is a projection heuristic, not measured 3D.
"""
from pathlib import Path
from collections import Counter
import argparse
import hashlib
import json
import os
import subprocess
import time

ROOT=Path(__file__).resolve().parent
# Keep Ultralytics runtime settings inside this workspace.
RUNTIME=ROOT/'yolo_runtime';RUNTIME.mkdir(exist_ok=True)
os.environ['YOLO_CONFIG_DIR']=str(RUNTIME)

import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont

REQUIRED=('forward','left','right')
NAMES={'forward':'前','left':'左','right':'右'}
EDGES=[(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),(11,13),(13,15),(12,14),(14,16)]


def unit(v):
    n=float(np.linalg.norm(v))
    return v/n if n>1e-6 else None


def arm_directions(points,confidence,keypoint_conf=.3):
    """Relative lateral axis uses anatomical shoulder/hip labels, not image x.

    Forward = raised, small lateral projection, and foreshortened arm. A single
    monocular 2D skeleton cannot uniquely identify all 3D pointing directions.
    """
    p=np.asarray(points,float);s=np.asarray(confidence,float)
    if p.shape!=(17,2) or s.shape!=(17,):raise ValueError('Expected 17 COCO keypoints')
    if min(s[[5,6,11,12]])<keypoint_conf:return set(),[]
    shoulder=p[[5,6]].mean(0);hip=p[[11,12]].mean(0)
    torso=float(np.linalg.norm(shoulder-hip))
    up=unit(shoulder-hip)
    if up is None or torso<1:return set(),[]
    raw_right=p[6]-p[5]
    # Remove apparent torso tilt while preserving anatomical right/left sign.
    right=unit(raw_right-np.dot(raw_right,up)*up)
    if right is None:
        raw_right=p[12]-p[11]
        right=unit(raw_right-np.dot(raw_right,up)*up)
    if right is None:return set(),[]
    arms=[]
    shoulder_width=float(np.linalg.norm(p[6]-p[5])/torso)
    for side,sh,el,wr in [('L',5,7,9),('R',6,8,10)]:
        if min(s[[sh,el,wr]])<keypoint_conf:continue
        v=p[wr]-p[sh];fore=p[wr]-p[el]
        lateral=float(v@right/torso);lift=float(v@up/torso)
        extension=float(np.linalg.norm(v)/torso)
        upper_len=float(np.linalg.norm(p[el]-p[sh]));fore_len=float(np.linalg.norm(fore))
        elbow_lift=float((p[el]-p[sh])@up/torso)
        arm_conf=float(min(s[[sh,el,wr]]))
        lateral_fraction=abs(lateral)/max(extension,1e-6)
        direction=None
        raised=lift>-.55 and extension>.18 and fore_len>.10*torso
        if raised and arm_conf>=.60 and abs(lateral)>=.40 and lateral_fraction>=.60:
            direction='right' if lateral>0 else 'left'
        elif (raised and abs(lateral)<=.25 and -.35<=lift<=.40 and
              shoulder_width>=.55 and arm_conf>=.60):
            # Forward needs positive evidence, not merely absence of lateral
            # extension. Exclude low-confidence hidden arms and side-on torsos.
            foreshortened=fore_len/max(upper_len,1e-6)<=.65
            wrist_above_elbow=lift-elbow_lift>=.12
            if foreshortened and wrist_above_elbow:direction='forward'
        arms.append({'side':side,'direction':direction,'lateral':lateral,'lift':lift,
                     'extension':extension,'forearm_upperarm_ratio':fore_len/max(upper_len,1e-6),
                     'raised':raised,'confidence':arm_conf,'elbow_lift':elbow_lift,
                     'strength':extension*arm_conf,'selected':False})
    # One person, one direction per frame. Clear lateral pointing suppresses a
    # competing forward guess from the other (often hidden) arm.
    candidates=[a for a in arms if a['direction'] in ('left','right')]
    if not candidates:candidates=[a for a in arms if a['direction']=='forward']
    candidates.sort(key=lambda a:a['strength'],reverse=True)
    found=set()
    if candidates:
        best=candidates[0]
        conflicting=[a for a in candidates[1:] if a['direction']!=best['direction']]
        ambiguous=bool(conflicting and best['strength']<1.3*conflicting[0]['strength'])
        if not ambiguous:
            best['selected']=True;found={best['direction']}
    return found,arms


class Completion:
    """One direction per gesture; stable opposite lateral holds can split a raise."""
    def __init__(self,stable_frames=3):
        if stable_frames<1:raise ValueError('stable_frames must be positive')
        self.stable_frames=stable_frames;self.people={}
    def _finish(self,st,frame,time_s,reason):
        event=st['event']
        if event is None:return
        selected=None
        if event.get('locked_lateral'):
            selected=event['locked_lateral']
        elif (event['max_streak'].get('forward',0)>=max(6,2*self.stable_frames) and
              max(event['max_streak'].get('left',0),event['max_streak'].get('right',0))<2):
            selected='forward'
        completed={**event,'ended_s':time_s,'end_frame':frame,'selected_direction':selected,'end_reason':reason}
        st['events'].append(completed)
        if selected and selected not in st['done']:
            st['done'][selected]={'time_s':time_s,'frame':frame,'gesture_started_s':event['started_s']}
        st['event']=None;st['neutral_frames']=0
        self._latch(st,time_s)

    def _latch(self,st,time_s):
        if len(st['done'])==3 and st['passed_s'] is None:st['passed_s']=time_s

    def snapshot(self,tid):
        st=self.people[int(tid)]
        return {'track_id':int(tid),'done':list(st['done']),
                'passed':st['passed_s'] is not None,'passed_s':st['passed_s']}

    def update(self,tid,frame,time_s,directions,arms=None):
        st=self.people.setdefault(int(tid),{'done':{},'last_frame':None,'first_seen_s':time_s,
            'passed_s':None,'event':None,'events':[],'neutral_frames':0})
        # Defensive: a multi-label observation never confirms multiple directions.
        direction=next(iter(directions)) if len(directions)==1 else None
        continuous=st['last_frame'] is not None and frame==st['last_frame']+1
        if arms is None:
            active=bool(directions);resting=not active
        else:
            active=any(a.get('raised',False) for a in arms)
            event_side=st['event'].get('active_side') if st['event'] else None
            previous_arm=next((a for a in arms if a['side']==event_side),None)
            resting=(len(arms)>=2 and not active) or bool(previous_arm and not previous_arm.get('raised',False))
        if not continuous and st['event']:
            st['event']['streak']={};st['event']['switch_streak']=0;st['neutral_frames']=0
            if st['last_frame'] is not None and frame-st['last_frame']>3:
                # Do not complete or join an action across a long tracking gap.
                st['event']=None
        if resting:
            st['neutral_frames']+=1
            if st['event']:
                st['event']['streak']={};st['event']['switch_streak']=0
            if st['neutral_frames']>=3:self._finish(st,frame,time_s,'ARM_LOWERED')
        elif active:
            st['neutral_frames']=0
            previous=st['event']
            # A clear left/right change is a new gesture even without lowering.
            # Forward guesses during that sweep never create another gesture.
            locked=previous.get('locked_lateral') if previous else None
            opposite=direction in ('left','right') and locked and direction!=locked
            if previous:
                if opposite and not previous.get('switch_streak'):
                    previous['switch_start_frame']=frame;previous['switch_started_s']=time_s
                previous['switch_streak']=previous.get('switch_streak',0)+1 if opposite else 0
            split=bool(previous and previous['switch_streak']>=self.stable_frames)
            carry=previous['switch_streak'] if split else 0
            if split:self._finish(st,frame,time_s,'STABLE_LATERAL_CHANGE')
            if st['event'] is None:
                st['event']={'started_s':time_s,'start_frame':frame,'votes':{},'streak':{},'max_streak':{},
                             'active_side':None,'locked_lateral':None,'switch_streak':0}
                if split:
                    st['event']['started_s']=previous['switch_started_s']
                    st['event']['start_frame']=previous['switch_start_frame']
            event=st['event']
            if arms:
                selected=next((a for a in arms if a.get('selected')),None)
                if selected and event['active_side'] is None:event['active_side']=selected['side']
            for d in REQUIRED:
                event['streak'][d]=event['streak'].get(d,0)+1 if d==direction else 0
                if carry and d==direction:event['streak'][d]=carry
                event['max_streak'][d]=max(event['max_streak'].get(d,0),event['streak'][d])
            if direction:event['votes'][direction]=event['votes'].get(direction,0)+(carry or 1)
            if direction in ('left','right') and event['streak'][direction]>=self.stable_frames and not event['locked_lateral']:
                event['locked_lateral']=direction
        elif st['event']:
            st['event']['streak']={};st['event']['switch_streak']=0;st['neutral_frames']=0
        st['last_frame']=frame;st['last_seen_s']=time_s
        return self.snapshot(tid)

    def close(self,frame,time_s):
        for st in self.people.values():
            self._finish(st,st['last_frame'],st['last_seen_s'],
                         'VIDEO_END' if st['last_frame']==frame else 'TRACK_END')


def render(frame,detections,font,label_fn=None):
    vis=frame.copy()
    for d in detections:
        k=np.array(d['keypoints_xy']);s=np.array(d['keypoint_confidence'])
        for a,b in EDGES:
            if min(s[[a,b]])>=.3:
                cv2.line(vis,tuple(np.rint(k[a]).astype(int)),tuple(np.rint(k[b]).astype(int)),(0,220,255),2,cv2.LINE_AA)
        box=np.rint(d['bbox']).astype(int);color=(0,255,0) if d['state']['passed'] else (0,0,255)
        cv2.rectangle(vis,tuple(box[:2]),tuple(box[2:]),color,3)
    im=Image.fromarray(cv2.cvtColor(vis,cv2.COLOR_BGR2RGB));draw=ImageDraw.Draw(im)
    for d in detections:
        st=d['state'];label=label_fn(st) if label_fn else f'ID{st["track_id"]} '+ ' '.join(f'{NAMES[x]}{int(x in st["done"])}' for x in REQUIRED)
        x=max(0,min(frame.shape[1]-220,int(d['bbox'][0])));y=max(0,int(d['bbox'][1])-29)
        if label_fn:
            bounds=draw.textbbox((0,0),label,font=font,stroke_width=1)
            x=max(0,min(frame.shape[1]-(bounds[2]-bounds[0])-4,int(d['bbox'][0])))
        draw.text((x,y),label,font=font,fill=(0,255,0) if st['passed'] else (255,70,70),
                  stroke_width=1,stroke_fill=(0,0,0))
    return cv2.cvtColor(np.array(im),cv2.COLOR_RGB2BGR)


def save_image(path,img):
    ok,b=cv2.imencode('.jpg',img)
    if not ok:raise RuntimeError('Image encode failed')
    Path(path).write_bytes(b.tobytes())


def convert(raw,final):
    p=subprocess.run(['ffmpeg','-hide_banner','-loglevel','error','-y','-i',str(raw),
        '-c:v','libx264','-crf','20','-preset','fast','-pix_fmt','yuv420p','-movflags','+faststart',str(final)],
        capture_output=True,text=True)
    if p.returncode:raise RuntimeError(p.stderr)
    raw.unlink()


def process(video,out,args):
    replay_root=getattr(args,'replay_from',None)
    cached=None
    if replay_root:
        cached=json.loads((replay_root/video.stem/'summary.json').read_text(encoding='utf-8'))
        if hashlib.sha256(video.read_bytes()).hexdigest()!=cached['source_sha256']:
            raise ValueError('Replay source video hash differs from cached YOLO inference')
        replay_rows=iter((replay_root/video.stem/'frame_results.jsonl').read_text(encoding='utf-8').splitlines())
        model_metadata={key:cached[key] for key in ('model','ultralytics','gpu','imgsz','precision')}
    else:
        import torch
        from ultralytics import YOLO
        # New tracker and direction states for each independent source video.
        model=YOLO(str(args.model));model.predict(np.zeros((640,640,3),np.uint8),device=0,
            imgsz=args.imgsz,half=True,verbose=False)
        model_metadata={'model':str(args.model),'ultralytics':__import__('ultralytics').__version__,
                        'gpu':torch.cuda.get_device_name(0),'imgsz':args.imgsz,'precision':'FP16'}
    cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS)
    frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));w,h=int(cap.get(3)),int(cap.get(4))
    if fps<=0 or not cap.isOpened():raise RuntimeError(f'Cannot read {video}')
    out.mkdir(parents=True,exist_ok=True)
    raw=out/'raw.mp4';writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    if not writer.isOpened():raise RuntimeError('Cannot create output')
    completion=Completion(args.stable_frames);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20)
    preview_at=set(np.linspace(0,frames-1,12,dtype=int));previews=[];counts=[];latencies=[];fi=0
    start=time.perf_counter()
    with (out/'frame_results.jsonl').open('w',encoding='utf-8') as f:
        while True:
            ok,frame=cap.read()
            if not ok:break
            t=time.perf_counter()
            observations=[]
            if cached:
                row=json.loads(next(replay_rows))
                if row['frame']!=fi:raise ValueError('Noncontiguous cached frames')
                observations=row['people']
            else:
                result=model.track(frame,persist=True,tracker='bytetrack.yaml',device=0,half=True,
                    imgsz=args.imgsz,conf=.25,iou=.7,max_det=32,verbose=False)[0]
                if result.boxes is not None and result.boxes.id is not None and result.keypoints is not None:
                    ids=result.boxes.id.int().cpu().numpy();boxes=result.boxes.xyxy.cpu().numpy()
                    keypoints=result.keypoints.xy.cpu().numpy();scores=result.keypoints.conf.cpu().numpy()
                    box_conf=result.boxes.conf.cpu().numpy()
                    for tid,b,k,s,bc in zip(ids,boxes,keypoints,scores,box_conf):
                        observations.append({'track_id':int(tid),'bbox':b.round(2).tolist(),
                            'detection_confidence':float(bc),'keypoints_xy':k.round(3).tolist(),
                            'keypoint_confidence':s.round(4).tolist()})
            detections=[]
            for obs in observations:
                directions,arms=arm_directions(obs['keypoints_xy'],obs['keypoint_confidence'])
                state=completion.update(obs['track_id'],fi,fi/fps,directions,arms)
                detections.append({**obs,'directions':sorted(directions),'arms':arms,'state':state})
            if fi==frames-1:
                completion.close(fi,fi/fps)
                for d in detections:d['state']=completion.snapshot(d['track_id'])
            latencies.append((time.perf_counter()-t)*1000);counts.append(len(detections))
            canvas=render(frame,detections,font);writer.write(canvas)
            f.write(json.dumps({'frame':fi,'time_s':fi/fps,'people':detections},ensure_ascii=False)+'\n')
            if fi in preview_at:
                p=out/f'preview_{fi:05d}.jpg';save_image(p,canvas);previews.append(p)
            if fi%100==0:print(f'{video.name} {fi}/{frames}, people={len(detections)}, green={sum(x["passed_s"] is not None for x in completion.people.values())}',flush=True)
            fi+=1
    seconds=time.perf_counter()-start;writer.release();cap.release()
    assert fi==frames,(fi,frames)
    convert(raw,out/'yolo_red_green.mp4')
    people=[]
    for tid,st in completion.people.items():
        people.append({'track_id':tid,'done':st['done'],'passed':st['passed_s'] is not None,
            'passed_s':st['passed_s'],'first_seen_s':st['first_seen_s'],'events':st['events']})
    summary={'video':str(video),'source_sha256':hashlib.sha256(video.read_bytes()).hexdigest(),
        **model_metadata,'inference_mode':'cached_yolo_keypoints' if cached else 'fresh_yolo',
        'replay_from':str(replay_root) if cached else None,
        'frames':fi,'width':w,'height':h,'source_fps':fps,
        'rules':{'required':['forward','left','right'],'region':None,'face_model':None,
                 'head_confirmation':False,'timeout':None,'hold_seconds':None,
                 'keypoint_debounce_frames':args.stable_frames,'forward_debounce_frames':max(6,2*args.stable_frames),
                 'exclusive_per_frame':True,'exclusive_per_gesture':True,
                 'stable_lateral_change_frames':args.stable_frames,'pass_latched':True},
        'people':people,'green_track_count':sum(p['passed'] for p in people),
        'red_track_count':sum(not p['passed'] for p in people),
        'max_simultaneous_people':max(counts,default=0),'processing_s':seconds,'processing_fps':fi/seconds,
        'pose_tracking_ms_mean':None if cached else float(np.mean(latencies)),
        'note':'Forward is inferred approximately from monocular 2D arm projection. Red means not all three directions detected, not a violation. IDs are tracker trajectories, not verified person counts.'}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
    thumbs=[]
    for p in previews:
        im=cv2.imdecode(np.frombuffer(p.read_bytes(),np.uint8),1);thumbs.append(cv2.resize(im,(360,round(im.shape[0]*360/im.shape[1]))))
    th=max(x.shape[0] for x in thumbs);sheet=np.full((((len(thumbs)+2)//3)*th,1080,3),25,np.uint8)
    for i,im in enumerate(thumbs):sheet[i//3*th:i//3*th+im.shape[0],i%3*360:(i%3+1)*360]=im
    save_image(out/'contact_sheet.jpg',sheet)
    print(json.dumps({'video':video.name,'green':summary['green_track_count'],'red':summary['red_track_count'],
        'fps':summary['processing_fps'],'people':[{k:v for k,v in p.items() if k!='events'} for p in people]},ensure_ascii=False),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--videos',nargs='+')
    parser.add_argument('--model',type=Path,default=ROOT/'models/yolo26m_pose/yolo26m-pose.pt')
    parser.add_argument('--out',type=Path,default=ROOT/'yolo_exclusive_test_20261024')
    parser.add_argument('--replay-from',type=Path,help='Re-evaluate cached YOLO keypoints against the identical source video')
    parser.add_argument('--imgsz',type=int,default=1280);parser.add_argument('--stable-frames',type=int,default=3)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    videos=[Path(x) for x in args.videos] if args.videos else [Path(json.loads((ROOT/'rtmw3d_l_test_20261024'/n/'summary.json').read_text(encoding='utf-8'))['video']) for n in ['3','4','9']]
    summaries=[process(v,args.out/v.stem,args) for v in videos]
    (args.out/'summaries.json').write_text(json.dumps(summaries,ensure_ascii=False,indent=2),encoding='utf-8')
