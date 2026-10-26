"""Fresh RTMW3D-L TensorRT FP16 inference, shared red/green completion rules.

The balanced profile uses a stable reference and temporal arm segmentation.
The legacy profile also exports the previous 2D-rule control video.
No manually configured region or separate face/head detector is used.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import ImageFont
from rtmw_trt_runtime import Engine,TRT_DIR,torch,trt
from rtmw3d_pointing_rules import arm_directions_3d
from yolo_pointing_simple import Completion,arm_directions,render,save_image,convert
from balanced_pointing_rules import BalancedPointing

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'rtmlib-main'))
from rtmlib.tools.pose_estimation.pre_processings import bbox_xyxy2cs,top_down_affine


def stat(values):
    return {'mean':float(np.mean(values)),'p50':float(np.percentile(values,50)),
            'p95':float(np.percentile(values,95))} if values else None


def pose_input(frame,box):
    center,scale=bbox_xyxy2cs(np.asarray(box,np.float32),padding=1.25)
    crop,scale=top_down_affine((288,384),scale,center,frame)
    crop=(crop[:,:,::-1].astype(np.float32)-np.array([123.675,116.28,103.53],np.float32))/np.array([58.395,57.12,57.375],np.float32)
    return np.ascontiguousarray(crop.transpose(2,0,1)[None]),center,scale


def decode_pose(raw,center,scale,shape,fx=1145.04940459,fy=1143.78109572,root_depth=5.14388):
    simcc=np.stack([np.argmax(a,axis=-1) for a in raw],axis=-1)[0].astype(np.float32)/2
    score=np.minimum(raw[0].max(-1),raw[1].max(-1))[0]
    xy=simcc[:,:2]/np.array([288,384],np.float32)*scale+center-scale/2
    z=(simcc[:,2]/144-1)*2.1744869
    depth=z+root_depth
    xyz=np.column_stack(((xy[:,0]-shape[1]/2)/fx*depth,(xy[:,1]-shape[0]/2)/fy*depth,depth))
    xyz-=xyz[[11,12]].mean(0)
    return xy,score,z,xyz,simcc


class PoseTRT:
    def __init__(self):self.net=Engine(TRT_DIR/'rtmw3d_l_fp16.engine')
    def raw(self,x):
        values=dict(zip(self.net.outputs,self.net(x)))
        return [values[n] for n in ('simcc_x','simcc_y','simcc_z')]
    def __call__(self,frame,box):
        x,center,scale=pose_input(frame,box)
        return decode_pose(self.raw(x),center,scale,frame.shape)


class DetectorTRT:
    def __init__(self):self.net=Engine(TRT_DIR/'yolox_s_fp16.engine')
    def preprocess(self,frame):
        h,w=frame.shape[:2];ratio=min(640/h,640/w)
        canvas=np.full((640,640,3),114,np.uint8)
        resized=cv2.resize(frame,(int(w*ratio),int(h*ratio)),interpolation=cv2.INTER_LINEAR)
        canvas[:resized.shape[0],:resized.shape[1]]=resized
        x=np.ascontiguousarray(canvas.transpose(2,0,1)[None],dtype=np.float32)
        return x,ratio,w,h
    @staticmethod
    def postprocess(output,ratio,w,h):
        boxes=next(a for a in output if a.shape[-1]==4)[0]
        scores=next(a for a in output if a.shape[-1]!=4)[0,0]
        good=scores>.30;boxes=boxes[good];scores=scores[good]
        if not len(boxes):return np.empty((0,5),np.float32)
        tlwh=boxes.copy();tlwh[:,2:]-=tlwh[:,:2]
        keep=cv2.dnn.NMSBoxes(tlwh.tolist(),scores.tolist(),.30,.65)
        if not len(keep):return np.empty((0,5),np.float32)
        keep=np.asarray(keep).reshape(-1);boxes=boxes[keep]/ratio;scores=scores[keep]
        boxes[:,[0,2]]=np.clip(boxes[:,[0,2]],0,w-1);boxes[:,[1,3]]=np.clip(boxes[:,[1,3]],0,h-1)
        good=((boxes[:,2:]-boxes[:,:2])>2).all(axis=1)
        return np.column_stack((boxes[good],scores[good])).astype(np.float32)
    def __call__(self,frame):
        x,ratio,w,h=self.preprocess(frame)
        return self.postprocess(self.net(x),ratio,w,h)


def make_tracker(fps):
    from ultralytics.trackers.byte_tracker import BYTETracker
    return BYTETracker(SimpleNamespace(track_high_thresh=.25,track_low_thresh=.10,
        new_track_thresh=.25,track_buffer=30,match_thresh=.80,fuse_score=True))


def render_2d_control(video,out,rows):
    cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS);w,h=int(cap.get(3)),int(cap.get(4))
    raw=out/'control_raw.mp4';writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20)
    for row in rows:
        ok,frame=cap.read()
        if not ok:raise RuntimeError('Control source read failed')
        dets=[{**d,'state':d['state_same_2d_rules']} for d in row['people']]
        writer.write(render(frame,dets,font))
    writer.release();cap.release();convert(raw,out/'rtmw3d_l_fp16_same_rules_2d.mp4')


def process(video,out,detector,pose,rule_mode='legacy'):
    from ultralytics.engine.results import Boxes
    out.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(str(video));fps=cap.get(cv2.CAP_PROP_FPS);w,h=int(cap.get(3)),int(cap.get(4))
    count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or fps<=0:raise RuntimeError(f'Cannot read {video}')
    ok,first=cap.read()
    if not ok:raise RuntimeError('Empty source video')
    boxes=detector(first)
    for _ in range(3):
        detector(first)
        if len(boxes):pose(first,boxes[0,:4])
    cap.set(cv2.CAP_PROP_POS_FRAMES,0)
    balanced=rule_mode=='balanced'
    tracker=make_tracker(fps);completion=BalancedPointing() if balanced else Completion(3);control=Completion(3)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',20)
    raw=out/'raw.mp4';writer=cv2.VideoWriter(str(raw),cv2.VideoWriter_fourcc(*'mp4v'),fps,(w,h))
    if not writer.isOpened():raise RuntimeError('Cannot create output video')
    preview_at=set(np.linspace(0,count-1,12,dtype=int));previews=[];rows=[]
    times={k:[] for k in ('detector_ms','pose_all_people_ms','tracking_rules_ms','pipeline_frame_ms')}
    person_instances=0;start=time.perf_counter()
    with (out/'frame_results.jsonl').open('w',encoding='utf8') as log:
        for fi in range(count):
            ft=time.perf_counter();ok,frame=cap.read()
            if not ok:raise RuntimeError(f'Truncated input at frame {fi}')
            t=time.perf_counter();det=detector(frame);times['detector_ms'].append((time.perf_counter()-t)*1000)
            t=time.perf_counter()
            detections=np.column_stack((det,np.zeros(len(det),np.float32)))
            tracks=tracker.update(Boxes(detections,(h,w)),frame)
            track_ms=(time.perf_counter()-t)*1000;pt=0;rt=track_ms;people=[]
            for track in tracks:
                b=track[:4];tid=int(track[4]);confidence=float(track[5])
                t=time.perf_counter();xy,s,z,xyz,simcc=pose(frame,b);pt+=(time.perf_counter()-t)*1000
                t=time.perf_counter()
                if balanced:
                    state,arms=completion.update(tid,fi,fi/fps,xyz,xy,s)
                    directions={a['direction'] for a in arms if a['selected']}
                else:
                    directions,arms=arm_directions_3d(xyz,xy,s)
                    state=completion.update(tid,fi,fi/fps,directions,arms)
                d2,a2=arm_directions(xy[:17],s[:17]);state2=control.update(tid,fi,fi/fps,d2,a2)
                rt+=(time.perf_counter()-t)*1000
                people.append({'track_id':tid,'bbox':b.round(2).tolist(),'detection_confidence':confidence,
                    'keypoints_xy':xy[:17].round(3).tolist(),'keypoint_confidence':s[:17].round(5).tolist(),
                    'keypoints_3d_assumed_camera':xyz[:17].round(6).tolist(),
                    'wholebody_keypoints_xy':xy.round(3).tolist(),'wholebody_scores_xy':s.round(5).tolist(),
                    'wholebody_depth_offset_m':z.round(6).tolist(),
                    'directions':sorted(directions),'arms':arms,'state':state,
                    'directions_same_2d_rules':sorted(d2),'state_same_2d_rules':state2})
            if fi==count-1:
                if balanced:completion.close()
                else:completion.close(fi,fi/fps)
                control.close(fi,fi/fps)
                for d in people:
                    d['state']=completion.snapshot(d['track_id']);d['state_same_2d_rules']=control.snapshot(d['track_id'])
            person_instances+=len(people)
            row={'frame':fi,'time_s':fi/fps,'people':people};rows.append(row)
            log.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')
            canvas=render(frame,people,font);writer.write(canvas)
            if fi in preview_at:
                path=out/f'preview_{fi:05d}.jpg';save_image(path,canvas);previews.append(path)
            times['pose_all_people_ms'].append(pt);times['tracking_rules_ms'].append(rt)
            times['pipeline_frame_ms'].append((time.perf_counter()-ft)*1000)
            if fi%100==0:print(f'{video.name} {fi}/{count} people={len(people)} det={times["detector_ms"][-1]:.2f}ms pose={pt:.2f}ms',flush=True)
    elapsed=time.perf_counter()-start;writer.release();cap.release()
    convert(raw,out/'rtmw3d_l_fp16_red_green.mp4')
    if not balanced:render_2d_control(video,out,rows)
    def summarize(counter):
        return [{'track_id':tid,'done':s['done'],'passed':s['passed_s'] is not None,
                 'passed_s':s['passed_s'],'events':s['events']} for tid,s in counter.people.items()]
    summary={'video':str(video),'source_sha256':hashlib.sha256(video.read_bytes()).hexdigest(),
        'model':'RTMW3D-L','pose_engine':str(TRT_DIR/'rtmw3d_l_fp16.engine'),
        'detector':'YOLOX-s HumanArt TensorRT FP16; person class 0; topK1000; CPU NMS IoU .65, score .30; every frame',
        'gpu':torch.cuda.get_device_name(0),'tensorrt':trt.__version__,
        'precision':'FP16; FP32-sensitive normalization/reduction layers',
        'frames':count,'source_fps':fps,'width':w,'height':h,'person_instances':person_instances,
        'mean_people':person_instances/count,'processing_s':elapsed,'processing_fps':count/elapsed,
        'timings':{k:stat(v) for k,v in times.items()},
        'rules':{'required':['forward','left','right'],'region':None,'face_model':None,
            'head_confirmation':False,'timeout':None,'exclusive_per_frame':True,
            'exclusive_per_gesture':True,'keypoint_debounce_frames':3,'forward_debounce_frames':6,
            'state_machine':'Unmodified yolo_pointing_simple.Completion',
            'primary_direction':'3D forearm in shoulder/hip body frame; sectors forward <=35deg, lateral 60..120deg',
            'control_direction':'Unmodified yolo_pointing_simple.arm_directions applied to RTMW XY'},
        'camera':{'calibrated':False,'fx':1145.04940459,'fy':1143.78109572,
                  'principal_point':'source image centre','root_depth_prior_m':5.14388},
        'people':completion.summary() if balanced else summarize(completion),'people_same_2d_rules':summarize(control),
        'green_track_count':sum(s['passed_s'] is not None for s in completion.people.values()),
        'green_track_count_same_2d_rules':sum(s['passed_s'] is not None for s in control.people.values()),
        'notes':['Fresh TensorRT inference on every frame; no cached video keypoints used.',
            'Processing FPS includes video decode, detector, CPU NMS, ByteTrack, pose crops, 3D decoding, both rule evaluations, drawing, JSONL and mp4v writing; excludes engine initialization, warmup, H264 transcode and second control-video rendering.',
            'XY SimCC peak scores are not calibrated probabilities and do not validate Z accuracy.',
            'Monocular XYZ uses uncalibrated camera/root-depth priors, not measured scene geometry.',
            'No manual ROI. Only body and arm joints affect decisions; no separate face or head model.',
            'Track IDs differ between detectors; match people visually, not by numerical ID across model families.']}
    if balanced:
        summary['rules']={
            'profile':'balanced_v1','required':['forward','left','right'],
            'region':None,'face_model':None,'head_confirmation':False,
            'exclusive_per_frame':True,'state_machine':'balanced_pointing_rules.BalancedPointing',
            'reference':'Stable initial person reference; shoulder-width-constrained monocular depth; not camera calibration',
            'parameters':completion.config,
            'control_direction':'Previous 2D rules evaluated for numerical comparison; no additional control video'}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    thumbs=[cv2.resize(cv2.imdecode(np.frombuffer(p.read_bytes(),np.uint8),1),(360,round(h*360/w))) for p in previews]
    sheet=np.vstack([np.hstack(thumbs[i:i+3]) for i in range(0,len(thumbs),3)])
    save_image(out/'contact_sheet.jpg',sheet)
    print(json.dumps({'video':video.name,'fps':summary['processing_fps'],
        'green_3d':summary['green_track_count'],'green_same_2d_rules':summary['green_track_count_same_2d_rules'],
        'people':[{k:v for k,v in p.items() if k!='events'} for p in summary['people']]},ensure_ascii=False),flush=True)
    return summary


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--videos',nargs='+');ap.add_argument('--out',type=Path)
    ap.add_argument('--rule-mode',choices=('legacy','balanced'),default='legacy')
    args=ap.parse_args()
    if args.out is None:
        args.out=ROOT/('rtmw3d_l_balanced_test_20261025' if args.rule_mode=='balanced' else 'rtmw3d_l_trt_fp16_test_20261024')
    videos=[Path(s) for s in args.videos] if args.videos else [Path(json.loads((ROOT/'rtmw3d_l_test_20261024'/n/'summary.json').read_text(encoding='utf8'))['video']) for n in ('3','4','9')]
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    t=time.perf_counter();detector=DetectorTRT();pose=PoseTRT();init=time.perf_counter()-t
    summaries=[process(v,args.out/v.stem,detector,pose,args.rule_mode) for v in videos]
    (args.out/'summaries.json').write_text(json.dumps({'model_init_s':init,'videos':summaries},ensure_ascii=False,indent=2),encoding='utf8')
