import os
import cv2 
# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def blur(img):
    #median blur
    kernel = 9
    blurred_im = cv2.medianBlur(img, kernel)
    return blurred_im

def threshold(img): 
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 200,cv2.THRESH_TOZERO_INV)

    return thresh

def extract_vid_feats(in_path, out_path):
    #read in video
    cap = cv2.VideoCapture(in_path)
    width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if (cap.isOpened()== False): 
        print("Error opening video stream or file")

    #dictionary for storing values
    output = {"frame": [], "x1": [], "y1": [], "x2": [], "y2": []}
    
    #play back  video
    frame_no = 0
    main_writer= cv2.VideoWriter(f'{out_path}_main.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (width,height))
    
    while cap.isOpened():
        ret, img = cap.read()
        # img = cv2.imread("red_hsv.png")
        # ret = True
        
        if ret == True:
            
            blurred = blur(img)
            thresh = threshold(blurred)
            
            cv2.imshow("Output", thresh)
            main_writer.write(thresh)
            key = cv2.waitKey(1)
            
            #press q to quit
            if key == ord('q'):
                break

            #p to pause
            if key == ord('p'):
                cv2.waitKey(-1)

        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    #save dict as pandas and csv file
    return 1

if __name__ == "__main__":
    extract_vid_feats("videos/A001_CD/A001_CD_trial1.mp4", "threshold")
