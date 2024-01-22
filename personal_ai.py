import mediapipe as mp
from mediapipe.tasks import python
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import numpy as np
import math


import threading
import queue

class PersonalAI:
    def __init__(self, file_name = "IMG_3429.mp4"):
        self.file_name = file_name
        self.model_path = "pose_landmarker_full.task"
        self.image_q = queue.Queue()
        self.opitions = python.vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(model_asset_path=self.model_path),
            running_mode=python.vision.RunningMode.VIDEO)
            # https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/python#video
      
    def find_angle(self, landmarks, p1, p2, p3):
        land = landmarks.pose_landmarks[0]
        # https://manivannan-ai.medium.com/find-the-angle-between-three-points-from-2d-using-python-348c513e2cd
        x1, y1 = (land[p1].x, land[p1].y)
        x2, y2 = (land[p2].x, land[p2].y)
        x3, y3 = (land[p3].x, land[p3].y)

        angle = math.degrees(math.atan2(y3-y2, x3-x2) - 
                                math.atan2(y1-y2, x1-x2))
        return angle

    def draw_landmarks_on_image(self, rgb_image, detection_result):
        pose_landmarks_list = detection_result.pose_landmarks
        annotated_image = np.copy(rgb_image)

        # Loop through the detected poses to visualize.
        for idx in range(len(pose_landmarks_list)):
            pose_landmarks = pose_landmarks_list[idx]

            # Draw the pose landmarks.
            pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style())
        return annotated_image
    
    def process_video(self, draw, display):
        with python.vision.PoseLandmarker.create_from_options(self.opitions) as landmarker:
            cap = cv2.VideoCapture(self.file_name)
            fps = cap.get(cv2.CAP_PROP_FPS)
            calc_ts = 0

            while cap.isOpened():
                ret, frame = cap.read()

                if ret == True:
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                    calc_ts = int(calc_ts + 1000 / fps)
                    detection_result = landmarker.detect_for_video(mp_image, calc_ts)
                    if draw:
                        frame = self.draw_landmarks_on_image(frame, detection_result)

                    if display:
                        cv2.imshow("Frame", frame)
                        if cv2.waitKey(25) & 0xFF == ord('q'):
                            break
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    self.image_q.put((frame, detection_result, calc_ts))

                else:
                    break
            cap.release()
            cv2.destroyAllWindows()

            self.image_q.put((1, 1, "done"))

    def run(self, draw=True, display=False):
        t1 = threading.Thread(target=self.process_video, args=(draw, display))
        t1.start()

if __name__ == "__main__":
    personalAI = PersonalAI()
    personalAI.process_video(False, True)