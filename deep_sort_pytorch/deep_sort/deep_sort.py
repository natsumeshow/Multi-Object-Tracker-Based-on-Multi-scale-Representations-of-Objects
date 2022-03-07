import numpy as np
import torch

from .deep.feature_extractor import Extractor
from .sort.nn_matching import NearestNeighborDistanceMetric
from .sort.detection import Detection
from .sort.tracker import Tracker


__all__ = ['DeepSort']


class DeepSort(object):
    def __init__(self, model_path, max_dist=0.2, min_confidence=0.3, nms_max_overlap=1.0, max_iou_distance=0.7, max_age=70, n_init=3, nn_budget=100, use_cuda=True):
        self.min_confidence = min_confidence
        self.nms_max_overlap = nms_max_overlap

        self.extractor = Extractor(model_path, use_cuda=use_cuda)

        max_cosine_distance = max_dist
        metric = NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(
            metric, max_iou_distance=max_iou_distance, max_age=max_age, n_init=n_init)

    def update(self, bbox_xywh, confidences, classes, ori_img , yolo_features = None):
        self.height, self.width = ori_img.shape[:2]
        # generate detections
        #features = self._get_features(bbox_xywh, ori_img)
        #print("yolo_features" , np.asarray(yolo_features).shape)
        features = self._get_yolo_features(bbox_xywh, ori_img,yolo_features)
        bbox_tlwh = self._xywh_to_tlwh(bbox_xywh)
        detections = [Detection(bbox_tlwh[i], conf, features[i]) for i, conf in enumerate(
            confidences) if conf > self.min_confidence]

        # run on non-maximum supression
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])

        # update tracker
        self.tracker.predict()
        self.tracker.update(detections, classes,ori_img,yolo_features)

        '''
        for i , track in enumerate(self.tracker.tracks):
            # number change 1 -> int
            if not track.is_confirmed() or track.time_since_update > 10:                
                continue
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            print(i , box)
        '''

        # output bbox identities
        outputs = []
        test_unmaching_tracks = []
        
        '''
        for track in self.tracker.tracks:
            # number change 1 -> int
            if not track.is_confirmed() or track.time_since_update > 10:                
                continue
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)

            # new 
            if track.unmaching_tracks_mean != [] : 
                unmaching_tracks_bbox_tlwh = torch.Tensor(box)
                unmaching_tracks_features = self._get_yolo_features([x1,y1,x2,y2], ori_img,yolo_features,unmach_flag=True)
                test_unmaching_tracks = Detection(unmaching_tracks_bbox_tlwh, self.min_confidence , unmaching_tracks_features[0])

        if test_unmaching_tracks : 
            detections.append(test_unmaching_tracks)   
        print("after" , len(detections))     
        #self.tracker.predict()
        #self.tracker.update(detections, classes)
        '''

        for track in self.tracker.tracks:
            # number change 1 -> int
            #if not track.is_confirmed() or track.time_since_update > 3 :        
            if track.time_since_update > 3:           
                continue
            
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)

            # new 
            track_id = track.track_id
            class_id = track.class_id
            outputs.append(np.array([x1, y1, x2, y2, track_id, class_id], dtype=np.int))
            track.clear_unmaching_tracks_mean()

        '''
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            box = track.to_tlwh()
            x1, y1, x2, y2 = self._tlwh_to_xyxy(box)
            track_id = track.track_id
            class_id = track.class_id
            outputs.append(np.array([x1, y1, x2, y2, track_id, class_id], dtype=np.int))
        '''
        if len(outputs) > 0:
            outputs = np.stack(outputs, axis=0)

        return outputs

    """
    TODO:
        Convert bbox from xc_yc_w_h to xtl_ytl_w_h
    Thanks JieChen91@github.com for reporting this bug!
    """
    @staticmethod
    def _xywh_to_tlwh(bbox_xywh):
        if isinstance(bbox_xywh, np.ndarray):
            bbox_tlwh = bbox_xywh.copy()
        elif isinstance(bbox_xywh, torch.Tensor):
            bbox_tlwh = bbox_xywh.clone()
        bbox_tlwh[:, 0] = bbox_xywh[:, 0] - bbox_xywh[:, 2] / 2.
        bbox_tlwh[:, 1] = bbox_xywh[:, 1] - bbox_xywh[:, 3] / 2.
        return bbox_tlwh

    def _xywh_to_xyxy(self, bbox_xywh):
        x, y, w, h = bbox_xywh
        x1 = max(int(x - w / 2), 0)
        x2 = min(int(x + w / 2), self.width - 1)
        y1 = max(int(y - h / 2), 0)
        y2 = min(int(y + h / 2), self.height - 1)
        return x1, y1, x2, y2

    def _tlwh_to_xyxy(self, bbox_tlwh):
        """
        TODO:
            Convert bbox from xtl_ytl_w_h to xc_yc_w_h
        Thanks JieChen91@github.com for reporting this bug!
        """
        x, y, w, h = bbox_tlwh
        x1 = max(int(x), 0)
        x2 = min(int(x+w), self.width - 1)
        y1 = max(int(y), 0)
        y2 = min(int(y+h), self.height - 1)
        return x1, y1, x2, y2

    def increment_ages(self):
        self.tracker.increment_ages()

    def _xyxy_to_tlwh(self, bbox_xyxy):
        x1, y1, x2, y2 = bbox_xyxy

        t = x1
        l = y1
        w = int(x2 - x1)
        h = int(y2 - y1)
        return t, l, w, h

    # new 
    def _tlwh_to_xywh(self,bbox_tlwh): 
        x,y,w,h = bbox_tlwh
        x1 = int(max(int(x) + w//2 , 0))
        y1 = int(max(int(y) + h//2 , 0))
        w = int(max(int(w),0))
        h = int(max(int(h),0))
        return x1,y1,w,h

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        for box in bbox_xywh:
            x1, y1, x2, y2 = self._xywh_to_xyxy(box)
            im = ori_img[y1:y2, x1:x2]
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features

    def _get_yolo_features(self, bbox_xywh, ori_img,yolo_features , unmach_flag = False) :
        #print(type(bbox_xywh) ) tensor 
        im_crops = []
        w,h,c = ori_img.shape
        #print("len(bbox_xywh)" , len(bbox_xywh))
        for box in bbox_xywh:
            if not unmach_flag : 
                x1, y1, x2, y2 = self._xywh_to_xyxy(box) 
            else : 
                x1, y1, x2, y2 = bbox_xywh
            im_crop = []
            for i in range(len(yolo_features)):
                w_stride = w / yolo_features[i].shape[1] 
                h_stride = h / yolo_features[i].shape[2]
                x1 , x2 = int(x1//w_stride) , int(x2//w_stride)
                y1 , y2 = int(y1//h_stride) , int(y2//h_stride)
                #print("yolo_features[i]" , yolo_features[i].shape)
                """
                if x2 == x1 and y2 > y1:
                    x2 +=1 
                elif y2 == y1 and x2 > x1 :
                    y2+=1
                elif x1 == x2 and y1 == y2 :
                    x2 +=1
                    y2 +=1
                assert x2 >= x1 and y2 >= y1
                """
                im = yolo_features[i][...,y1:y2 , x1:x2]
                #print("im" , im.shape)
                avg_pooling = torch.nn.AdaptiveAvgPool2d((1,1))
                test_feature = avg_pooling(im).view(-1)
                im_crop.append(test_feature)

            im_crop = torch.cat(im_crop,axis = 0)
            im_crops.append(torch.unsqueeze(im_crop,dim=0))

        im_crops = torch.cat(im_crops,axis = 0)
        #print("im_crops shape" , (im_crops).shape , len(im_crops))
        #return np.array(im_crops) if im_crops else np.array([])
        return im_crops 

    """
    def _get_yolo_features(self, bbox_xywh, ori_img,yolo_features , unmach_flag = False) :
        #print(type(bbox_xywh) ) tensor 
        im_crops = []
        w,h,c = ori_img.shape

        w_stride = w / yolo_features.shape[2] 
        h_stride = h / yolo_features.shape[3]
        
        for box in bbox_xywh:
            if not unmach_flag : 
                x1, y1, x2, y2 = self._xywh_to_xyxy(box) 
            else : 
                x1, y1, x2, y2 = bbox_xywh
            x1 , x2 = int(x1//w_stride) , int(x2//w_stride)
            y1 , y2 = int(y1//h_stride) , int(y2//h_stride)
            im = yolo_features[0,...,y1:y2 , x1:x2]
            avg_pooling = torch.nn.AdaptiveAvgPool2d((1,1))
            test_features = avg_pooling(im).view(-1)
            #print(test_features.shape)
            im_crops.append(test_features)

        return np.array(im_crops) if im_crops else np.array([])
        """

