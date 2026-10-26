import unittest
import numpy as np
from rtmw3d_pointing_rules import arm_directions_3d


class Directions3DTests(unittest.TestCase):
    def pose(self):
        p=np.zeros((17,3));p[[5,6,11,12]]=[[.2,0,0],[-.2,0,0],[.15,.5,0],[-.15,.5,0]]
        p[[7,9,8,10]]=[[.2,.25,0],[.2,.5,0],[-.2,.25,0],[-.2,.5,0]]
        q=p[:,:2]*200+200
        return p,q,np.ones(17)
    def test_front_is_toward_camera_when_person_faces_camera(self):
        p,q,s=self.pose();p[[8,10]]=[[-.2,.1,-.2],[-.2,.05,-.5]]
        q[[8,10]]=[[160,220],[180,210]]
        self.assertEqual(arm_directions_3d(p,q,s)[0],{'forward'})
    def test_anatomical_left_and_right(self):
        p,q,s=self.pose();p[[7,9]]=[[.45,0,0],[.7,0,0]];q=p[:,:2]*200+200
        self.assertEqual(arm_directions_3d(p,q,s)[0],{'left'})
        p,q,s=self.pose();p[[8,10]]=[[-.45,0,0],[-.7,0,0]];q=p[:,:2]*200+200
        self.assertEqual(arm_directions_3d(p,q,s)[0],{'right'})
    def test_no_face_landmarks_used_for_decision(self):
        p,q,s=self.pose();p[[7,9]]=[[.45,0,0],[.7,0,0]];q=p[:,:2]*200+200;s[:5]=0
        self.assertEqual(arm_directions_3d(p,q,s)[0],{'left'})
    def test_camera_rotation_preserves_person_relative_direction(self):
        p,q,s=self.pose();p[[7,9]]=[[.45,0,0],[.7,0,0]];q=p[:,:2]*200+200
        for angle in np.linspace(-np.pi,np.pi,13):
            c=np.cos(angle);t=np.sin(angle);rotation=np.array([[c,0,t],[0,1,0],[-t,0,c]])
            self.assertEqual(arm_directions_3d(p@rotation.T,q,s)[0],{'left'})
    def test_down_and_backward_are_not_forward(self):
        p,q,s=self.pose();self.assertEqual(arm_directions_3d(p,q,s)[0],set())
        p[[8,10]]=[[-.2,.1,.2],[-.2,.05,.5]];q[[8,10]]=[[160,220],[170,210]]
        self.assertEqual(arm_directions_3d(p,q,s)[0],set())
    def test_confidence_and_degenerate_body(self):
        p,q,s=self.pose();p[[7,9]]=[[.45,0,0],[.7,0,0]];q=p[:,:2]*200+200;s[9]=.5
        self.assertEqual(arm_directions_3d(p,q,s)[0],set())
        p[6]=p[5]
        self.assertEqual(arm_directions_3d(p,q,s)[0],set())
    def test_lateral_suppresses_other_arm_forward(self):
        p,q,s=self.pose();p[[7,9]]=[[.45,0,0],[.7,0,0]]
        p[[8,10]]=[[-.2,.1,-.2],[-.2,.05,-.5]]
        q=p[:,:2]*200+200;q[10]=[180,210]
        directions,arms=arm_directions_3d(p,q,s)
        self.assertEqual(directions,{'left'})
        self.assertEqual({a['direction'] for a in arms},{'left','forward'})
        self.assertEqual(sum(a['selected'] for a in arms),1)


if __name__=='__main__':unittest.main()
