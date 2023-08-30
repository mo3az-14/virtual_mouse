import mediapipe as mp
import cv2
from hand_detector import HandDetector
import time

#######################
# EXAMPLE DRIVER CODE #
#######################

if __name__ == "__main__":

    # calculating fps
    prev_frame_timer: float = 0
    new_frame_timer: float = 0

    # the model we are going to use
    # link to the model
    # https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
    model_path = r"hand_landmarker.task"

    test = HandDetector(model_path, 2, 0.5, 0.5, 0.5 , bounding_box= True)
    cap = cv2.VideoCapture(0)

    # setting up we need to dimensions of our camera before we start
    test.image_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    test.image_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # creating the track bars to set up the zone That we can move our finger in
    test.movement_zone()

    # font for the fps text
    font = cv2.FONT_HERSHEY_SIMPLEX
    while True:
        try:

            # reading camera input
            _, frame = cap.read()

            # flipping the output image from the camera on the vertical axis
            frame = cv2.flip(frame, 1)

            # convert the camera feed to mp.Image format in order to be fed to the model
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            # this is where the detection happens
            res = test.detector.detect_for_video(mp_image, int(time.time() * 1_000))

            # drawing the landmarks on the frame
            frame = test.draw_hand_landmarks(res, mp_image)

            #  start recording the finger history of the movement of the finger
            if test.history_trigger:
                test.history([res.hand_landmarks[0][8].x, res.hand_landmarks[0][8].y],
                             [res.hand_landmarks[0][4].x, res.hand_landmarks[0][4].y],
                             [res.hand_landmarks[0][5].x, res.hand_landmarks[0][5].y])

            # start controlling the mouse
            if test.move_mouse:
                test.move_cursor_with_finger()
                test.listen_for_clicks()

        except Exception as e:
            print(e)

        # listen to keyboard key
        key = cv2.waitKey(1) & 0xFF

        # q to close the app
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break

        # h to start recording the movement history
        elif key == ord("h"):
            test.history_trigger = not test.history_trigger

        # p to enable mouse movement
        elif key == ord("p"):
            test.move_mouse = not test.move_mouse

        # calculating and displaying the fps
        new_frame_timer = time.time()
        fps = (1 / (new_frame_timer - prev_frame_timer))
        fps = str(int(fps))
        cv2.rectangle(frame, (test.x1, test.y1), (test.x2, test.y2), (0, 0, 255), 3)
        cv2.putText(frame, str(fps), (10, 30), font, 1, (0, 0, 0), 2)
        cv2.imshow('frame', frame)
        prev_frame_timer = new_frame_timer
