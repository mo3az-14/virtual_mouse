# import cv2 
# cap = cv2.VideoCapture(0)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# cv2.namedWindow("zone")
# def fu (x): print(x)
# cv2.createTrackbar("top_x" , "zone" , 0 , width  , fu)
# cv2.createTrackbar("top_y" , "zone" , 0, height , fu)
# cv2.createTrackbar("bot_x" , "zone" , 0 , width , fu)
# cv2.createTrackbar("bot_y" , "zone" , 0 , height , fu)

# while True : 
#     ret , frame = cap.read()
#     top_x = cv2.getTrackbarPos("top_x" , "zone")
#     top_y = cv2.getTrackbarPos("top_y" , "zone")
#     bot_x = cv2.getTrackbarPos("bot_x" , "zone")
#     bot_y = cv2.getTrackbarPos("bot_y" , "zone")
#     frame=cv2.flip(frame , 1)
#     cv2.rectangle(frame, (top_x , top_y ), (bot_x , bot_y)  ,(0,255,0) , 1)
#     cv2.imshow("rame",frame )
#     key = cv2.waitKey(1)& 0xff
#     if key == ord("q"): 
#         cap.release()
#         cv2.destroyAllWindows
#         break
import torch 
print( torch.cuda.is_available())

