import sys
import cv2
import os
import numpy as np



cmd = 'python capture.py'
os.system(cmd)

cv2.waitKey(5000)

cmd = 'python inference_my_f.py'
os.system(cmd)
cmd = 'python inference_my_m.py'
os.system(cmd)

