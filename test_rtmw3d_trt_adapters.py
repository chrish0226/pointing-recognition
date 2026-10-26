import unittest
import numpy as np
from rtmw3d_trt_pointing import DetectorTRT,decode_pose,make_tracker


class AdapterTests(unittest.TestCase):
    def test_installed_tracker_retains_two_ids_across_short_gap(self):
        from ultralytics.engine.results import Boxes
        tracker=make_tracker(30)
        detections=np.array([[10,10,50,100,.9,0],[100,10,150,100,.8,0]],np.float32)
        first=tracker.update(Boxes(detections,(200,200)))
        ids=set(first[:,4]);self.assertEqual(len(ids),2)
        tracker.update(Boxes(np.empty((0,6),np.float32),(200,200)))
        detections[:,[0,2]]+=1
        after=tracker.update(Boxes(detections,(200,200)))
        self.assertEqual(set(after[:,4]),ids)
    def test_nonperson_classes_are_excluded_and_duplicate_people_suppressed(self):
        boxes=np.array([[[10,10,60,100],[100,20,160,100],[12,12,62,102]]],np.float32)
        scores=np.zeros((1,80,3),np.float32)
        scores[0,0,[0,2]]=[.9,.7];scores[0,2,1]=.99
        result=DetectorTRT.postprocess([boxes,scores],1,640,640)
        self.assertEqual(result.shape,(1,5));np.testing.assert_allclose(result[0,:4],boxes[0,0])
    def test_no_confident_person_returns_empty(self):
        boxes=np.zeros((1,3,4),np.float32);scores=np.zeros((1,80,3),np.float32);scores[0,1,1]=.99
        self.assertEqual(DetectorTRT.postprocess([boxes,scores],1,640,640).shape,(0,5))
    def test_simcc_decoding_uses_distinct_xy_and_depth_scales(self):
        raw=[np.zeros((1,133,n),np.float32) for n in (576,768,576)]
        for a,i in zip(raw,(288,384,288)):a[:,:,i]=1
        xy,s,z,xyz,simcc=decode_pose(raw,np.array([140,240]),np.array([120,160]),(720,1280,3))
        np.testing.assert_allclose(xy,np.tile([140,240],(133,1)))
        np.testing.assert_allclose(z,0);np.testing.assert_allclose(xyz,0);np.testing.assert_allclose(s,1)


if __name__=='__main__':unittest.main()
