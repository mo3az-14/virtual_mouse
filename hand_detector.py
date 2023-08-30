from typing import Any
import mediapipe as mp
import cv2
import numpy as np
import mouse
import tkinter as tk

# settings for mediapipe
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
HandLandmarkerResult = mp.tasks.vision.HandLandmarkerResult
VisionRunningMode = mp.tasks.vision.RunningMode


class HandDetector:

    def __init__(self, path,
                 num_hands=1,
                 detection_confidence=0.3,
                 presence_confidence=0.3,
                 tracking_confidence=0.3,
                 bounding_box=False):

        # flag if right click is detected
        self.right_click: bool = False

        # trigger that will turn off the right click after being pressed once and wait
        # for the next right click to avoid application freezing
        self.right_click_trigger: bool = False

        # factor for smoothing the mouse movement
        self.smoothing_factor: int = 3

        # pc screen dimensions
        self.screen_width: int = 0
        self.screen_height: int = 0

        # turns on or off moving the mouse
        self.move_mouse: bool = False

        # start recording gesture
        self.history_trigger: bool = False

        # fingers history
        self.finger1_tip_history: np.ndarray = np.array([[0, 0], [0, 0]])
        self.finger2_tip_history: np.ndarray = np.array([[0, 0], [0, 0]])
        self.finger3_tip_history: np.ndarray = np.array([[0, 0], [0, 0]])

        # image dimensions
        self.image_width: int = 0
        self.image_height: int = 0

        # zone dimensions for the finger movements
        self.x1: int = 0
        self.y1: int = 0
        self.x2: int = 0
        self.y2: int = 0

        # path to the model 
        self.path: str = path

        # number of hands to be detected
        self.num_hands: int = num_hands

        # confidence settings for the hand landmarks detector
        self.detection_confidence: float = detection_confidence
        self.presence_confidence: float = presence_confidence

        # base options 
        self.options: mp.tasks.vision.HandLandmarkerOptions = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=path),
            num_hands=num_hands,
            running_mode=VisionRunningMode.VIDEO,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=presence_confidence,
            min_tracking_confidence=tracking_confidence
        )
        # making the detector 
        self.detector = HandLandmarker.create_from_options(self.options)

        # get the bounding box   
        self.bounding_box = bounding_box
        root = tk.Tk()
        self.screen_height: int = root.winfo_screenheight()
        self.screen_width: int = root.winfo_screenwidth()
        root.destroy()

    def normalize(self, x: float, y: float, norm_type: int | bool) -> tuple[int | Any, int | Any]:
        """
        Normalize or de-normalize the x and y coordinates based on the output image dimensions.
        For normalization, x is divided by the width, and y is divided by the height.
        For de-normalization, x is multiplied by the width, and y is multiplied by the height.

        Parameters:
        __________
         x : float
            The x-coordinate from the landmark.
         y : float
            The y-coordinate from the landmark.
        norm_type : int | bool
            1 for de-normalization, or 0 for normalization.

        Returns:
        _______
        Normalized or de-normalized x and y coordinates as integers.
        """

        if norm_type == 1:
            return int(x * self.image_width), int(y * self.image_height)
        elif norm_type == 0:
            return int(x / self.image_width), int(y / self.image_height)
        else:
            print("wrong value")

    # bounding box of the hand
    def get_bounding_box(self, land_marks: mp.tasks.vision.HandLandmarkerResult,
                         offset: int = 10) -> tuple[int | Any, int | Any, int | Any, int | Any]:
        """
        Gets the bounding box around the hands with an option to apply an offset.

        Parameters:
        __________
        land_marks : mp.tasks.vision.HandLandmarkerResult.hand_landmarks
            The result object from the hand detector.
        offset : int
            The offset value to adjust the bounding box size.
        land_marks : int
            The result object from the hand detector.
        Returns:
        ________
        tuple[int | Any, int | Any, int | Any, int | Any]
             A tuple containing the modified bounding box coordinates (min_x, min_y, max_x, max_y).
        """

        min_x, max_x, min_y, max_y = self.image_width, 0, self.image_height, 0
        for i in range(21):
            x, y = self.normalize(land_marks[i].x, land_marks[i].y, 1)
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)

        # offset by 10 pixels:
        return min_x - offset, min_y - offset, max_x + offset, max_y + offset

    def draw_hand_landmarks(self, results: mp.tasks.vision.HandLandmarkerResult,
                            img: mp.Image | np.ndarray) -> np.array:
        """

        Parameters:
        __________
        results: mp.tasks.vision.HandLandmarkerResult
            The result object from the detector
        img: mp.Image | np.ndarray
             the image or frame we are going to draw the landmarks over
        Returns:
        ________
        np.array:
            The image with the landmarks in a numpy.ndarray format
        """

        image = img.numpy_view() if type(img) == mp.Image else img

        for i in range(len(results.handedness)):
            connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

            for mark in results.hand_landmarks:
                for g in range(21):
                    x, y = self.normalize(mark[g].x, mark[g].y, 1)
                    cv2.circle(image, (x, y), radius=8, color=(0, 0, 255), thickness=-1)

                for j in connections:
                    x_start, y_start = self.normalize(mark[j.start].x, mark[j.start].y, 1)
                    x_end, y_end = self.normalize(mark[j.end].x, mark[j.end].y, 1)

                    cv2.line(image,
                             (x_start, y_start),
                             (x_end, y_end),
                             (255, 0, 0),
                             thickness=4)

                if self.bounding_box:
                    min_x, min_y, max_x, max_y = self.get_bounding_box(mark)
                    cv2.rectangle(image,
                                  (min_x, max_y),
                                  (max_x, min_y),
                                  (0, 255, 0),
                                  1)
        return image

    def history(self, first: np.ndarray | list[int],
                second: np.ndarray | list[int],
                third: np.ndarray | list[int]):
        """
        Record the history of movement for the 3 points from the hand landmarker.
        The first point is used as the point that moves the mouse cursor.
        If the second point touches the third point, it will trigger a left click.

        Note:
        _____
            ONLY RECORDS THE LAST 2 FRAMES
        Usage example:
            - The first point can be the tip of the pointer finger (8).
            - The second point can be the tip of the thumb (4).
            - The third point can be the MCP of the pointer finger (5).

        Example:
        ________
        >>>    self.history(
        >>>        [result.hand_landmarks[0][8].x, result.hand_landmarks[0][8].y],
        >>>        [result.hand_landmarks[0][4].x, result.hand_landmarks[0][4].y],
        >>>        [result.hand_landmarks[0][5].x, result.hand_landmarks[0][5].y]
        >>>    )

        Parameter:
        _________

        first : np.ndarray | list[int]
            Array that contains the x, y coordinates of the first point.
        second : np.ndarray | list[int]
            Array that contains the x, y coordinates of the second point.
        third : np.ndarray | list[int]
            Array that contains the x, y coordinates of the third point.
        """

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

    # enable mouse clicks. for now, it's only left click.
    def listen_for_clicks(self, left_click_threshold: float = 0.009, left_click_listener_threshold: float = 0.08):
        """
        Detects left-click actions based on finger distances using provided threshold values.

        This function calculates distances between normalized coordinates using `np.linalg.norm()`.

        A left click is triggered when the distance between the second and third points is <= to the
        `left_click_threshold`. The function will avoid further left clicks until the distance between the
        second and third points is greater >= to the `left_click_listener_threshold`.

        Parameters
        _________
        left_click_threshold : float
            The threshold distance for left-click activation.
        left_click_listener_threshold : float
            The threshold distance for reactivating left-click detection.
        """

        finger1_finger2_dist = np.linalg.norm(self.finger3_tip_history[1] -
                                              self.finger2_tip_history[1])

        if finger1_finger2_dist <= left_click_threshold:
            self.right_click = True
        if finger1_finger2_dist > left_click_listener_threshold:
            self.right_click_trigger = False
            print("now listening to clicks ")

    def move_cursor_with_finger(self):
        """
        this function enables moving the cursor relative the position of the finger and also responsible for the
        execution of the clicks.

        Examples
        --------

        >>>    key = cv2.waitKey(1) & 0xFF
        >>>
        >>>    if key == ord("p"):
        >>>       self.move_mouse = not self.move_mouse

        """
        x_1, y_1 = self.normalize(self.finger1_tip_history[1][0],
                                  self.finger1_tip_history[1][1], 1)

        x_1 = np.interp(x_1, (min(self.x1, self.x2),
                              max(self.x1, self.x2)),
                        (0, self.screen_width))

        y_1 = np.interp(y_1, (min(self.y1, self.y2),
                              max(self.y1, self.y2)),
                        (0, self.screen_height))

        curr_x, curr_y = mouse.get_position()
        x_1 = curr_x + (x_1 - curr_x) / 3
        y_1 = curr_y + (y_1 - curr_y) / 3

        mouse.move(x_1, y_1, absolute=True)

        if self.right_click_trigger is False and self.right_click is True:
            self.right_click = False
            self.right_click_trigger = True
            mouse.click()
            print("pressed left click")

    # x returns the new value of the track bar when it is changed, but doesn't specify which bar has changed,
    # so we don't use it.
    def _change_zone(self, x=None):
        """
        the callback function for the track bars. this function is responsible for updating the movement_zone points and
        the smoothing factor. x is required by cv2.getTrackbarPos, but it won't be used.
        """
        if cv2.getWindowProperty("zone", cv2.WND_PROP_VISIBLE) < 1:
            print("closed track bar")
        else:
            self.x1 = int((cv2.getTrackbarPos("x1", "zone") / 100) * self.image_width)
            self.y1 = int((cv2.getTrackbarPos("y1", "zone") / 100) * self.image_height)
            self.x2 = int((cv2.getTrackbarPos("x2", "zone") / 100) * self.image_width)
            self.y2 = int((cv2.getTrackbarPos("y2", "zone") / 100) * self.image_height)
            self.smoothing_factor = cv2.getTrackbarPos("smoothing", "zone")

    def movement_zone(self):
        """
        creates 5 track bars for making a mini zone that will move the cursor relative to the position of the finger.
        this function is made because the model will lose the tracking of the hand if a part of the hand disappears
        making it impossible to reach the bottom and the sides of the screen. this zone makes it easier to reach the
        desired point.

        - x1: x-axis of the first point
        - y1: y-axis of the first point
        - x2: x-axis of the second point
        - y2: y-axis of the second point
        - smoothing: the movement smoothing factor. This is made because the inaccuracies of the model make it
          impossible for the cursor to stay still.
        """
        cv2.namedWindow("zone")
        cv2.resizeWindow("zone", 400, 300)
        cv2.createTrackbar("x1", "zone", 0, 100, self._change_zone )
        cv2.createTrackbar("y1", "zone", 0, 100, self._change_zone)
        cv2.createTrackbar("x2", "zone", 0, 100, self._change_zone)
        cv2.createTrackbar("y2", "zone", 0, 100, self._change_zone)
        # higher is slower cursor movement
        cv2.createTrackbar("smoothing", "zone", 0, 100, self._change_zone)
