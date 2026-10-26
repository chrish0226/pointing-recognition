"""Behavior checks for current anatomical directions and foreshortened arms."""
import math
import unittest
import numpy as np
from balanced_pointing_rules import BalancedPointing
from realtime_rtmw3d import GestureTally
from test_balanced_pointing_rules import pose


def body_pose(direction=None, yaw=0, short_front=False, folded=False):
    p, _, scores = pose(direction)
    p[[13,14,15,16]] = [[.15,.85,0],[-.15,.85,0],[.15,1.2,0],[-.15,1.2,0]]
    if short_front:
        p[[8,10]] = [[-.2,.025,-.28],[-.2,.05,-.65]]
    if folded:
        p[[8,10]] = [[-.2,.15,0],[-.2,.02,-.015]]
    angle=math.radians(yaw)
    rotate=np.array([[math.cos(angle),0,math.sin(angle)],[0,1,0],[-math.sin(angle),0,math.cos(angle)]])
    p=p@rotate.T
    return p,p[:,:2]*200+500,scores


class CurrentBodyRulesTests(unittest.TestCase):
    def setUp(self):
        self.counter=BalancedPointing()
        self.tally=GestureTally()
        self.frame=0

    def feed(self, data, count=1, walking=False):
        for _ in range(count):
            p,q,s=[a.copy() for a in data]
            if walking:
                t=self.frame/30
                q[:,0]+=130*t
                q[15,0]+=20*math.sin(2*math.pi*2*t)
                q[16,0]-=20*math.sin(2*math.pi*2*t)
            state,_=self.counter.update(1,self.frame,self.frame/30,p,q,s)
            self.tally.observe(1,self.counter.people[1]['phase'])
            self.frame+=1
        return self.tally.decorate(state)

    def test_anatomical_directions_across_camera_view_angles(self):
        for yaw in (-135,-90,-45,0,45,90,135,180):
            for direction in ('left','right','forward'):
                with self.subTest(yaw=yaw,direction=direction):
                    self.setUp()
                    self.feed(body_pose(yaw=yaw),20)
                    state=self.feed(body_pose(direction,yaw=yaw),25)
                    self.assertEqual(state['done'],[direction])
                    self.assertEqual(state['counts'][direction],1)

    def test_turning_between_gestures_updates_body_reference(self):
        for direction,yaw in [('left',0),('right',65),('forward',-65)]:
            self.feed(body_pose(yaw=yaw),35)
            state=self.feed(body_pose(direction,yaw=yaw),25)
        self.assertTrue(state['passed'])
        self.assertEqual(state['counts'],{'forward':1,'left':1,'right':1})

    def test_body_turn_with_same_held_arm_is_not_new_gesture(self):
        self.feed(body_pose(),20)
        self.feed(body_pose('forward',short_front=True),20)
        for yaw in np.linspace(0,90,31):
            self.feed(body_pose('forward',yaw=float(yaw),short_front=True))
        state=self.feed(body_pose('forward',yaw=90,short_front=True),25)
        self.assertEqual(state['counts'],{'forward':1,'left':0,'right':0})

    def test_front_arm_short_in_image_uses_spatial_extension(self):
        self.feed(body_pose(),20)
        data=body_pose(short_front=True)
        self.assertLess(np.linalg.norm(data[1][10]-data[1][6])/100,.18)
        state=self.feed(data,20)
        self.assertEqual(state['counts'],{'forward':1,'left':0,'right':0})

    def test_folded_hand_against_shoulder_is_not_short_forward(self):
        self.feed(body_pose(),20)
        state=self.feed(body_pose(folded=True),40)
        self.assertEqual(state['done'],[])

    def test_collapsed_shoulder_geometry_does_not_crash_or_count(self):
        p,_,s=body_pose()
        p[[5,6,11,12]]=[[0,0,0],[0,0,0],[.5,0,0],[.5,0,0]]
        state=self.feed((p,p[:,:2]*200+500,s),25)
        self.assertEqual(state['done'],[])

    def weak_arm(self):
        p,q,s=body_pose()
        p[[8,10]]=[[-.2,.25,0],[-.4,.18,0]]
        return p,p[:,:2]*200+500,s

    def test_weak_arm_during_gait_is_not_a_point(self):
        state=self.feed(self.weak_arm(),110,walking=True)
        self.assertEqual(state['done'],[])

    def test_clear_point_while_walking_remains_allowed(self):
        self.feed(body_pose(),20,walking=True)
        state=self.feed(body_pose('right'),40,walking=True)
        self.assertEqual(state['counts'],{'forward':0,'left':0,'right':1})

    def test_weak_pose_can_confirm_after_body_reference_settles(self):
        self.feed(body_pose(yaw=90),25)
        for yaw in np.linspace(90,0,16):
            self.feed(body_pose(yaw=float(yaw)))
        early=self.feed(self.weak_arm(),10)
        self.assertEqual(early['done'],[])
        late=self.feed(self.weak_arm(),50)
        self.assertEqual(late['counts'],{'forward':0,'left':0,'right':1})


if __name__=='__main__':unittest.main()
