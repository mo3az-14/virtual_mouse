import mediapipe as mp
import cv2 as cv
from hand_detector import Hand_Detector
import time

prev_frame_timer = 0
new_frame_timer = 0
model_path = r"C:\Users\moaaz\Desktop\wsl\hand_landmarker.task"
test = Hand_Detector(model_path, 2, 0.5, 0.5, 0.5)
cap = cv.VideoCapture(0)
test.image_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
test.image_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
test.creat_zone()

try:
    while True:

        _, frame = cap.read()
        frame = cv.flip(frame, 1)
        font = cv.FONT_HERSHEY_SIMPLEX
        try:

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            res = test.landmarker.detect_for_video(mp_image, int(time.time() * 1_000))
            frame = test.draw_hand_landmarks(res, mp_image)

            if test.history_trigger:
                test.record_fingertip_gesture([res.hand_landmarks[0][8].x, res.hand_landmarks[0][8].y],
                                              [res.hand_landmarks[0][4].x, res.hand_landmarks[0][4].y],
                                              [res.hand_landmarks[0][5].x, res.hand_landmarks[0][5].y])

            test.set_ratios()

            if test.distance:
                test.distance_of_gesture()
            if test.move_mouse:
                test.move_cursor_with_finger()
                test.look_for_clicks()

        except Exception as e:
            print(e)

        key = cv.waitKey(1) & 0xFF
        # print(key)
        if key == ord('q'):
            cap.release()
            cv.destroyAllWindows()
            break

        elif key == ord("h"):
            test.history_trigger = not test.history_trigger
        elif key == ord("p"):
            test.move_mouse = not test.move_mouse

        new_frame_timer = time.time()
        fps = (1 / (new_frame_timer - prev_frame_timer))
        fps = str(int(fps))
        prev_frame_timer = new_frame_timer
        cv.rectangle(frame,
                     (test.top_x, test.top_y),
                     (test.bot_x, test.bot_y),
                     (0, 0, 255),
                     3)
        cv.putText(frame, str(fps), (10, 30), font, 1, (0, 0, 0), 2)
        cv.imshow('frame', frame)

except Exception as e:
    print(e)
