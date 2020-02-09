import sys
import cv2
import os
import numpy as np
import shutil

device_num = 2
dir_path1 = 'dataM'
dir_path2 = 'dataF'

basename = 'img'
ext = 'jpg'
delay = 1
window_name = 'frame'

cap = cv2.VideoCapture(device_num)

# os.makedirs(dir_path)#, exist_ok=True)
base_path1 = os.path.join(dir_path1, basename)
base_path2 = os.path.join(dir_path2, basename)

n = 0

while True:
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(img, (320, 240), 128, color=(255, 255, 255), thickness=2)
    ret, frame = cap.read()
    dst = cv2.add(img, frame)
    cv2.imshow(window_name, dst)
    key = cv2.waitKey(delay) & 0xFF
    if key == ord('\r'):
        frame = frame[112:112 + 256, 192:192 + 256]
        frame = cv2.resize(frame, (128, 128))
        # frame = cv2.copyMakeBorder(frame, 192, 112, 256, 256, cv2.BORDER_CONSTANT)
        cv2.imwrite('{}_{}.{}'.format(base_path1, n, ext), frame)

        shutil.copy('{}_{}.{}'.format(base_path1, n, ext), '{}_{}.{}'.format(base_path2, n, ext))
        n += 1
        cap.release()
        cv2.destroyAllWindows()
        # cv2.destroyWindow(window_name)

        break
