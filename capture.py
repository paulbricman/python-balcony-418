import cv2
import datetime

cam = cv2.VideoCapture(1) # Change with your camera
cv2.namedWindow("test")

img_counter = 0

# Manually end on ESC, add image_counter condition for automatic end
while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    # Every minute
    if datetime.datetime.now().second == 0 and captured == False:
        img_name = "frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1
        captured = True
    # Only once every minute
    if datetime.datetime.now().second != 0:
        captured = False

# Finish
cam.release()
cv2.destroyAllWindows()
