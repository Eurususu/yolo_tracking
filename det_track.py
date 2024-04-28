import cv2
import numpy as np
from pathlib import Path

from boxmot import DeepOCSORT
from boxmot import BYTETracker
from trtEngine import BaseEngine

class Predictor(BaseEngine):
    def __init__(self, engine_path):
        super(Predictor, self).__init__(engine_path)
        self.n_classes = 3

# tracker_list ['bytetrack', 'botsort', 'strongsort', 'ocsort', 'deepocsort', 'hybridsort']

# tracker = DeepOCSORT(
#     model_weights=Path('osnet_x0_25_msmt17.pt'), # which ReID model to use
#     device='cuda:0',
#     fp16=False,
# )

tracker = BYTETracker()
video_path = 'rtsp://192.168.12.28/live/3/2'
engine_apth = '/home/jia/PycharmProjects/TensorRT-For-YOLO-Series/best_tiny_int8.trt'
pred = Predictor(engine_apth)
pred.detect_video_with_track(video_path, tracker, conf=0.45,  end2end=True)
