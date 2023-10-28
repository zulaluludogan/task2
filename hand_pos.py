import time, random
import pygame
import mediapipe as mp
import cv2

try:
    import pygame
    import mediapipe as mp
except:
    print("Pygame and Mediapipe should be installed. pip install pygame")
    raise ImportError

class MP_Controller:
    def __init__(self, mode=1):
        self.hand_result = mp.tasks.vision.HandLandmarkerResult
        self.hand_landmarker = mp.tasks.vision.HandLandmarker
        self._createHandLandmarker()

    def _createHandLandmarker(self):
        # callback function
        def update_result(
            hand_result: mp.tasks.vision.HandLandmarkerResult,
            output_image: mp.Image,
            timestamp_ms: int,
        ):
            self.hand_result = hand_result

        options_hands = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path="hand_landmarker.task"
            ),  # path to model
            running_mode=mp.tasks.vision.RunningMode.LIVE_STREAM,  # running on a live stream
            num_hands=1,  # track both hands
            min_hand_detection_confidence=0.5,  # lower than value to get predictions more often
            min_hand_presence_confidence=0.5,  # lower than value to get predictions more often
            min_tracking_confidence=0.5,  # lower than value to get predictions more often
            result_callback=update_result,
        )

        # initialize landmarker
        self.hand_landmarker = self.hand_landmarker.create_from_options(options_hands)

    def detect_async(self, frame, mode):
        # convert np frame to mp image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        # detect landmarks
        self.hand_landmarker.detect_async(
            image=mp_image, timestamp_ms=int(time.time() * 1000)
        )

    def get_index_tip_coordinates(self):
        # print('Hand landmarker result:\n {}'.format(self.hand_result))
        if self.hand_result.hand_landmarks != []:
            print(
                "HandLandmark.INDEX_FINGER_TIP result:\n {}".format(
                    self.hand_result.hand_landmarks[0][8]
                )
            )  # (HandLandmark.INDEX_FINGER_TIP=8)

            # GET INDEX_FINGER POSITION
            return (
                self.hand_result.hand_landmarks[0][8].x
                ,self.hand_result.hand_landmarks[0][8].y
                # ,self.hand_result.hand_landmarks[0][8].z
            )
        else:
            return (
                self.hand_result.hand_landmarks
            )


    def close(self):
        # close landmarker
        self.hand_landmarker.close()
        self.face_landmarker.close()


