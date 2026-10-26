import unittest
import json
from pathlib import Path
import numpy as np
from yolo_pointing_simple import Completion,arm_directions


class SimpleTests(unittest.TestCase):
    def pose(self):
        p=np.zeros((17,2),float)
        p[[5,6,11,12]]=[[120,100],[80,100],[115,180],[85,180]]
        p[[7,9,8,10]]=[[120,145],[120,190],[80,145],[80,190]]
        return p,np.ones(17)
    def test_down_not_counted(self):
        p,s=self.pose();self.assertEqual(arm_directions(p,s)[0],set())
    def test_person_left_not_screen_left(self):
        p,s=self.pose();p[[7,9]]=[[160,100],[200,100]]
        self.assertEqual(arm_directions(p,s)[0],{'left'})
        # Mirroring the image preserves anatomical point identities and labels.
        p[:,0]=300-p[:,0]
        self.assertEqual(arm_directions(p,s)[0],{'left'})
    def test_person_right(self):
        p,s=self.pose();p[[8,10]]=[[40,100],[0,100]]
        self.assertEqual(arm_directions(p,s)[0],{'right'})
    def test_forward_projection(self):
        p,s=self.pose();p[[5,6]]=[[124,100],[76,100]];p[[7,9]]=[[154,125],[139,106]]
        self.assertEqual(arm_directions(p,s)[0],{'forward'})
        s[9]=.5
        self.assertEqual(arm_directions(p,s)[0],set())
    def test_low_confidence_arm_not_counted(self):
        p,s=self.pose();p[[7,9]]=[[160,100],[200,100]];s[9]=.1
        self.assertEqual(arm_directions(p,s)[0],set())
        s[9]=.5
        self.assertEqual(arm_directions(p,s)[0],set())
    def test_clear_side_suppresses_other_arm_forward_candidate(self):
        p,s=self.pose();p[[5,6]]=[[124,100],[76,100]]
        p[[7,9]]=[[154,125],[139,106]];p[[8,10]]=[[36,100],[-4,100]]
        directions,arms=arm_directions(p,s)
        self.assertEqual(directions,{'right'})
        self.assertEqual(sum(a['selected'] for a in arms),1)
    def gesture(self,c,tid,start,duration,direction):
        for i in range(start,start+duration):c.update(tid,i,i/30,{direction})
        for i in range(start+duration,start+duration+3):r=c.update(tid,i,i/30,set())
        return r,start+duration+3
    def test_all_three_green_and_latched(self):
        c=Completion();i=0
        for d in ['right','forward','left']:r,i=self.gesture(c,1,i,6,d)
        self.assertTrue(r['passed']);self.assertTrue(c.update(1,i,i/30,set())['passed'])
    def test_tracks_do_not_share(self):
        c=Completion();_,i=self.gesture(c,1,0,6,'left')
        r,_=self.gesture(c,1,i,6,'forward');self.assertFalse(r['passed'])
        r,_=self.gesture(c,2,0,6,'right');self.assertFalse(r['passed'])
    def test_no_time_or_region_requirement(self):
        c=Completion();_,i=self.gesture(c,1,0,6,'left')
        _,i=self.gesture(c,1,1000,6,'right')
        r,_=self.gesture(c,1,2000,6,'forward');self.assertTrue(r['passed'])
    def test_short_jitter_not_counted(self):
        c=Completion(3);c.update(1,0,0,{'left'});c.update(1,1,.03,{'left'})
        for i in range(2,5):r=c.update(1,i,i/30,set())
        self.assertEqual(r['done'],[])
    def test_real_left_gesture_does_not_add_ghost_forward(self):
        fixture=json.loads((Path(__file__).parent/'regression_fixtures/yolo_left_with_ghost_forward.json').read_text(encoding='utf-8'))
        directions,arms=arm_directions(fixture['keypoints_xy'],fixture['keypoint_confidence'])
        self.assertEqual(directions,{'left'})
        self.assertEqual(sum(a['selected'] for a in arms),1)
    def test_same_raise_never_counts_left_and_forward(self):
        c=Completion()
        for i in range(18):c.update(1,i,i/30,{'forward'} if i<6 or i>=12 else {'left'})
        for i in range(18,21):r=c.update(1,i,i/30,set())
        self.assertEqual(r['done'],['left'])
        self.assertEqual(len(c.people[1]['events']),1)
    def test_both_labels_defensively_rejected(self):
        c=Completion()
        for i in range(9):c.update(1,i,i/30,{'left','forward'})
        c.close(8,8/30);self.assertEqual(c.snapshot(1)['done'],[])
    def test_opposite_lateral_hold_can_be_a_new_gesture_without_lowering(self):
        c=Completion()
        for i in range(12):c.update(1,i,i/30,{'left'} if i<6 else {'right'})
        c.close(11,11/30);self.assertEqual(c.snapshot(1)['done'],['left','right'])
        self.assertEqual(len(c.people[1]['events']),2)
    def test_brief_opposite_lateral_jitter_does_not_split_gesture(self):
        c=Completion()
        for i in range(8):c.update(1,i,i/30,{'left'} if i<6 else {'right'})
        c.close(7,7/30);self.assertEqual(c.snapshot(1)['done'],['left'])
    def test_forward_needs_six_consecutive_frames(self):
        c=Completion();r,_=self.gesture(c,1,0,5,'forward');self.assertEqual(r['done'],[])
    def test_missing_arm_is_not_arm_lowering(self):
        c=Completion()
        raised=[{'side':'L','raised':True,'selected':True}]
        for i in range(6):c.update(1,i,i/30,{'left'},raised)
        for i in range(6,12):r=c.update(1,i,i/30,set(),[])
        self.assertEqual(r['done'],[])
        self.assertIsNotNone(c.people[1]['event'])
    def test_long_tracking_gap_cannot_join_gesture(self):
        c=Completion()
        for i in range(6):c.update(1,i,i/30,{'left'})
        c.update(1,30,1,{'forward'});c.close(30,1)
        self.assertEqual(c.snapshot(1)['done'],[])


if __name__=='__main__':unittest.main()
