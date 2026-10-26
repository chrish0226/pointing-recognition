"""Behavioral regressions for temporal arm recognition, without GPU imports."""
import unittest
import numpy as np
from balanced_pointing_rules import BalancedPointing,DEFAULTS,reference_sample,make_reference


def pose(direction=None):
    p=np.zeros((17,3))
    p[[5,6,11,12]]=[[.2,0,0],[-.2,0,0],[.15,.5,0],[-.15,.5,0]]
    p[[7,9,8,10]]=[[.2,.25,0],[.2,.5,0],[-.2,.25,0],[-.2,.5,0]]
    if direction=='left':p[[7,9]]=[[.45,.05,0],[.7,.05,0]]
    if direction=='right':p[[8,10]]=[[-.45,.05,0],[-.7,.05,0]]
    if direction=='forward':p[[8,10]]=[[-.2,.25,-.25],[-.14,.20,-.5]]
    return p,p[:,:2]*200+200,np.ones(17)


class BalancedRulesTests(unittest.TestCase):
    def feed(self,c,points,start,count,tid=1):
        for f in range(start,start+count):state,arms=c.update(tid,f,f/30,*points)
        return state,arms

    def test_static_lowered_arms_never_complete(self):
        c=BalancedPointing();state,_=self.feed(c,pose(),0,60)
        self.assertEqual(state['done'],[])

    def test_real_three_gestures_and_latched_green(self):
        c=BalancedPointing();f=0
        for direction in ('left','forward','right'):
            self.feed(c,pose(),f,12);f+=12
            state,_=self.feed(c,pose(direction),f,16);f+=16
        self.assertEqual(set(state['done']),{'left','forward','right'})
        self.assertTrue(state['passed'])
        state,_=self.feed(c,pose(),f,20)
        self.assertTrue(state['passed'])

    def test_one_lateral_hold_does_not_add_forward(self):
        c=BalancedPointing();self.feed(c,pose(),0,8)
        state,_=self.feed(c,pose('left'),8,35)
        self.assertEqual(state['done'],['left'])

    def test_moderate_bent_arm_below_shoulder_is_allowed(self):
        p,q,s=pose('left');p[[7,9]]=[[.35,.33,0],[.62,.27,0]];q=p[:,:2]*200+200
        c=BalancedPointing();self.feed(c,pose(),0,8)
        state,_=self.feed(c,(p,q,s),8,16)
        self.assertEqual(state['done'],['left'])

    def test_single_frame_wrist_spike_does_not_count(self):
        c=BalancedPointing();self.feed(c,pose(),0,8)
        self.feed(c,pose('left'),8,1)
        state,_=self.feed(c,pose(),9,15)
        self.assertEqual(state['done'],[])

    def test_short_confidence_dropout_does_not_erase_real_gesture(self):
        c=BalancedPointing();self.feed(c,pose(),0,8)
        for f in range(8,26):
            p,q,s=pose('forward')
            if f in (12,15):s[10]=.1
            state,_=c.update(1,f,f/30,p,q,s)
        self.assertEqual(state['done'],['forward'])

    def test_track_gap_cannot_join_two_short_spikes(self):
        c=BalancedPointing();self.feed(c,pose(),0,8)
        self.feed(c,pose('left'),8,2)
        state,_=self.feed(c,pose('left'),30,2)
        self.assertEqual(state['done'],[])

    def test_people_accumulate_independently(self):
        c=BalancedPointing()
        for f in range(30):
            for tid,direction in ((1,'left'),(2,'right')):
                c.update(tid,f,f/30,*pose(direction if f>=8 else None))
        self.assertEqual(c.snapshot(1)['done'],['left'])
        self.assertEqual(c.snapshot(2)['done'],['right'])

    def test_missing_face_landmarks_have_no_effect(self):
        c=BalancedPointing();self.feed(c,pose(),0,8)
        p,q,s=pose('left');p[:5]=np.nan;q[:5]=np.nan;s[:5]=0
        state,_=self.feed(c,(p,q,s),8,16)
        self.assertEqual(state['done'],['left'])

    def test_frontal_shoulders_reject_extreme_predicted_depth_yaw(self):
        p,q,s=pose();p[5,2]=-.5;p[6,2]=.5
        sample=reference_sample(p,q,s);ref=make_reference([sample]*12,DEFAULTS)
        self.assertLess(abs(ref['right'][2]),.01)
        self.assertFalse(ref['calibrated'])

    def test_losing_wrist_confidence_cannot_create_a_direction(self):
        c=BalancedPointing();self.feed(c,pose(),0,8)
        p,q,s=pose('left');s[9]=.2
        state,_=self.feed(c,(p,q,s),8,20)
        self.assertEqual(state['done'],[])

    def test_waist_height_hand_manipulation_is_not_forward_pointing(self):
        c=BalancedPointing();self.feed(c,pose(),0,8)
        p,q,s=pose('forward')
        p[[8,10]]=[[-.2,.25,-.2],[-.14,.30,-.4]];q=p[:,:2]*200+200
        state,_=self.feed(c,(p,q,s),8,25)
        self.assertEqual(state['done'],[])

    def test_quick_three_directions_without_lowering_between_them(self):
        c=BalancedPointing();self.feed(c,pose(),0,10)
        self.feed(c,pose('forward'),10,6)
        self.feed(c,pose('right'),16,8)
        state,_=self.feed(c,pose('left'),24,10)
        self.assertEqual(set(state['done']),{'forward','right','left'})

    def test_shoulder_label_flip_during_held_arm_does_not_add_direction(self):
        c=BalancedPointing();self.feed(c,pose(),0,8)
        self.feed(c,pose('left'),8,15)
        p,q,s=pose('left');p[[5,6]]=p[[6,5]];q[[5,6]]=q[[6,5]]
        self.feed(c,(p,q,s),23,5)
        state,_=self.feed(c,pose('left'),28,15)
        self.assertEqual(state['done'],['left'])


if __name__=='__main__':unittest.main()
