import mediapipe as mp
import cv2 as cv
import numpy as np
import mouse
import tkinter as tk

# settings for mediapipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


# imagew  imageh screnh screenw
# 480.0   640.0  864    1536

class Hand_Detector:
    rightclick = False
    right_click_trigger = False
    smoothing_factor = 0.004
    screen_width = 0
    screen_height = 0
    screen_dim = False
    move_mouse = False
    distance = False
    # circular = np.loadtxt("gesture.txt")
    # start recording gesture
    history_trigger = False
    # fingers history
    finger0_tip_history = np.array([])
    finger1_tip_history = np.array([[0, 0], [0, 0]])
    finger2_tip_history = np.array([[0, 0], [0, 0]])
    finger3_tip_history = np.array([[0, 0], [0, 0]])
    finger4_tip_history = np.array([])
    # image dimensions
    image_width = 0
    image_height = 0
    top_x = image_width
    top_y = image_height
    bot_x = 0
    bot_y = 0

    def __init__(self, path,
                 num_hands=1,
                 detection_confidence=0.3,
                 presence_confidence=0.3,
                 tracking_confidence=0.3,
                 bounding_box=False):
        # path to the model 
        self.path = path
        self.num_hands = num_hands
        self.detection_confidence = detection_confidence
        self.presence_confidence = presence_confidence
        # base options 
        self.options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=path),
            num_hands=num_hands,
            running_mode=VisionRunningMode.VIDEO,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=presence_confidence,
            min_tracking_confidence=tracking_confidence
        )
        # making the detector 
        self.landmarker = HandLandmarker.create_from_options(self.options)

        # get the bounding box   
        self.bounding_box = bounding_box
        root = tk.Tk()
        self.screen_height = root.winfo_screenheight()
        self.screen_width = root.winfo_screenwidth()
        root.destroy()

    def normalize(self, x: float, y: float, type: int) -> tuple[int, int]:
        """
        @param x: x coordinate from the landmark
        @param y: y coordinate from the landmark
        @param type : 1 -> denormalize ,  0-> normalize 
        return:  x , y
        """
        if type == 1:
            return int(x * self.image_width), int(y * self.image_height)
        elif type == 0:
            return int(x / self.image_width), int(y / self.image_height)
        else:
            print("wrong value")

    # bounding box of the hand 
    def get_bounding_box(self, land_mark, img):

        min_x, max_x, min_y, max_y = self.image_width, 0, self.image_height, 0
        for i in range(21):
            x, y = self.normalize(land_mark[i].x, land_mark[i].y, 1)
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)

        # offset by 10 pixels
        return min_x - 10, min_y - 10, max_x + 10, max_y + 10


    def set_ratios(self):

        self.ratioh = 300 / self.screen_height
        self.ratiow = 300 / self.screen_width

    def draw_hand_landmarks(self, results: any, img: np.array) -> np.array:

        image = img.numpy_view()

        for i in range(len(results.handedness)):

            connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

            for mark in results.hand_landmarks:
                for i in range(21):
                    x, y = self.normalize(mark[i].x, mark[i].y, 1)
                    cv.circle(image, (x, y), radius=8, color=(0, 0, 255), thickness=-1)

                for j in connections:
                    x_start, y_start = self.normalize(mark[j.start].x, mark[j.start].y, 1)
                    x_end, y_end = self.normalize(mark[j.end].x, mark[j.end].y, 1)

                    cv.line(image,
                            (x_start, y_start),
                            (x_end, y_end),
                            (255, 0, 0),
                            thickness=4)

                if self.bounding_box:
                    min_x, min_y, max_x, max_y = self.get_bounding_box(mark, img)
                    cv.rectangle(image,
                                 (min_x, max_y),
                                 (max_x, min_y),
                                 (0, 255, 0),
                                 1)
        return image

    def record_fingertip_gesture(self, first, second, third):

        if self.finger1_tip_history.shape[0] <= 1:
            self.finger1_tip_history = np.append(self.finger1_tip_history, [first],
                                                 axis=0)
            self.finger2_tip_history = np.append(self.finger2_tip_history, [second],
                                                 axis=0)
            self.finger3_tip_history = np.append(self.finger3_tip_history, [third],
                                                 axis=0)
        else:

            self.finger1_tip_history = self.finger1_tip_history[1:]
            self.finger2_tip_history = self.finger2_tip_history[1:]
            self.finger3_tip_history = self.finger3_tip_history[1:]

            self.finger1_tip_history = np.append(self.finger1_tip_history, [first],
                                                 axis=0)
            self.finger2_tip_history = np.append(self.finger2_tip_history, [second],
                                                 axis=0)
            self.finger3_tip_history = np.append(self.finger3_tip_history, [third],
                                                 axis=0)

    def get_distance(self, normalized_coordinates):

        difference = np.array([[normalized_coordinates[1][0] - normalized_coordinates[0][0],
                                normalized_coordinates[1][1] - normalized_coordinates[0][1]]])

        return difference * np.array([self.screen_width, self.screen_height])

    def look_for_clicks(self):

        finger1_finger2_dist = np.linalg.norm(self.finger3_tip_history[1] -
                                              self.finger2_tip_history[1])

        if finger1_finger2_dist < 0.009:
            self.rightclick = True
        if finger1_finger2_dist > 0.08:
            self.right_click_trigger = False
            print("now listening to clicks ")

    def move_cursor_with_finger(self):

        x_1, y_1 = self.normalize(self.finger1_tip_history[1][0],
                                  self.finger1_tip_history[1][1], 1)

        x_1 = np.interp(x_1, (min(self.top_x, self.bot_x),
                              max(self.top_x, self.bot_x)),
                        (0, self.screen_width))

        y_1 = np.interp(y_1, (min(self.top_y, self.bot_y),
                              max(self.top_y, self.bot_y)),
                        (0, self.screen_height))

        curr_x, curr_y = mouse.get_position()
        x_1 = curr_x + (x_1 - curr_x) / 3
        y_1 = curr_y + (y_1 - curr_y) / 3

        mouse.move(x_1, y_1, absolute=True)

        if self.right_click_trigger is False and self.rightclick is True:
            self.rightclick = False
            self.right_click_trigger = True
            mouse.click()
            print("pressed left click")

    def _change_zone(self, x):
        if cv.getWindowProperty("zone", cv.WND_PROP_VISIBLE) < 1:

            print("closed track bar")

        else:

            self.top_x = int((cv.getTrackbarPos("top_x", "zone") / 100) * self.image_width)
            self.top_y = int((cv.getTrackbarPos("top_y", "zone") / 100) * self.image_height)
            self.bot_x = int((cv.getTrackbarPos("bot_x", "zone") / 100) * self.image_width)
            self.bot_y = int((cv.getTrackbarPos("bot_y", "zone") / 100) * self.image_height)

    def creat_zone(self):
        cv.namedWindow("zone")
        cv.resizeWindow("zone", 400, 180)
        cv.createTrackbar("top_x", "zone", 0, 100, self._change_zone)
        cv.createTrackbar("top_y", "zone", 0, 100, self._change_zone)
        cv.createTrackbar("bot_x", "zone", 0, 100, self._change_zone)
        cv.createTrackbar("bot_y", "zone", 0, 100, self._change_zone)
