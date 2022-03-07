# vim: expandtab:ts=4:sw=4
import numpy as np
import torch
from numpy import dot
from numpy.linalg import norm

torch.set_default_tensor_type('torch.cuda.FloatTensor')

class TrackState:
    """
    Enumeration type for the single target track state. Newly created tracks are
    classified as `tentative` until enough evidence has been collected. Then,
    the track state is changed to `confirmed`. Tracks that are no longer alive
    are classified as `deleted` to mark them for removal from the set of active
    tracks.

    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """
    A single target track with state space `(x, y, a, h)` and associated
    velocities, where `(x, y)` is the center of the bounding box, `a` is the
    aspect ratio and `h` is the height.

    Parameters
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[ndarray]
        Feature vector of the detection this track originates from. If not None,
        this feature is added to the `features` cache.

    Attributes
    ----------
    mean : ndarray
        Mean vector of the initial state distribution.
    covariance : ndarray
        Covariance matrix of the initial state distribution.
    track_id : int
        A unique track identifier.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state : TrackState
        The current track state.
    features : List[ndarray]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.

    """

    def __init__(self, mean, covariance, track_id, class_id, n_init, max_age,
                 feature=None):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.class_id = class_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        self.state = TrackState.Tentative
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age

        # new 
        #self.unmatched_tracks = []
        self.unmaching_tracks_mean = []
        self.unmaching_tracks_covariance = []

    def to_tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
        width, height)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Get current position in bounding box format `(min x, miny, max x,
        max y)`.

        Returns
        -------
        ndarray
            The bounding box.

        """
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def increment_age(self):
        self.age += 1
        self.time_since_update += 1

    def predict(self, kf):
        """Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.

        """
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.increment_age()

    def update(self, kf, detection,track_idx=None, samples = None,ori_img=None, yolo_features=None, unmatch_flag=False):
        """Perform Kalman filter measurement update step and update the feature
        cache.

        Parameters
        ----------
        kf : kalman_filter.KalmanFilter
            The Kalman filter.
        detection : Detection
            The associated detection.

        """

        if detection is not None :
            self.mean, self.covariance = kf.update(
                self.mean, self.covariance, detection.to_xyah())
            #print("updata self.mean" , self.mean)
            self.features.append(detection.feature)
            #self.cls_name.append(detection.class_name)
            self.hits += 1
            self.time_since_update = 0
            if self.state == TrackState.Tentative and self.hits >= self._n_init:
                self.state = TrackState.Confirmed

        # '''
        elif unmatch_flag: 
            self.mean, self.covariance = kf.update(
            self.mean, self.covariance, self.mean , miss_flag=True)

            try : 
                self.height, self.width = ori_img.shape[:2]
                test_mean = self.mean[:4].copy()
                test_mean[2] =  test_mean[2]*self.mean[3]
                test_mean_features = self._get_yolo_features(self._tlwh_to_xyxy(test_mean) ,ori_img,yolo_features)
                test_mean_feature = test_mean_features[0].to('cpu').detach().numpy().copy()
                test_mean_feature = np.nan_to_num(test_mean_feature)
                #x1,y1,x2,y2 = self._tlwh_to_xyxy(test_mean)
                cosin = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
                get_libra_feature = torch.as_tensor(samples.get(track_idx)[-1]).to('cpu').detach().numpy().copy()
                get_libra_feature = np.nan_to_num(test_mean_feature)

                cos_sim = dot(test_mean_feature, get_libra_feature.T)/(norm(test_mean_feature)*norm(get_libra_feature.T))

                if (1 - cos_sim) == 0 :
                    self.unmaching_tracks_mean = self.mean.copy()
                    self.unmaching_tracks_covariance = self.covariance.copy()
            except :
                pass

    def mark_missed(self):
        """Mark this track as missed (no association at the current time step).
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed).
        """
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted

    def clear_unmaching_tracks_mean(self):
        self.unmaching_tracks_mean = []
        self.unmaching_tracks_covariance = []

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


    def _get_yolo_features(self, xyxy, ori_img,yolo_features) :
        #print(type(bbox_xywh) ) tensor 
        im_crops = []
        w,h,c = ori_img.shape
        x1,y1,x2,y2 = xyxy
        im_crop = []
        for i in range(len(yolo_features)):
            w_stride = w / yolo_features[i].shape[1] 
            h_stride = h / yolo_features[i].shape[2]
            x1 , x2 = int(x1//w_stride) , int(x2//w_stride)
            y1 , y2 = int(y1//h_stride) , int(y2//h_stride)
            #print("yolo_features[i]" , yolo_features[i].shape)
            im = yolo_features[i][...,y1:y2 , x1:x2]
            #print("im" , im.shape)
            avg_pooling = torch.nn.AdaptiveAvgPool2d((1,1))
            test_feature = avg_pooling(im).view(-1)
            im_crop.append(test_feature)

        im_crop = torch.cat(im_crop, axis = 0)
        im_crops.append(im_crop)
        return np.array(im_crops) if im_crops else np.array([])
