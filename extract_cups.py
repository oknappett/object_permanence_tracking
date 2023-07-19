import os
import cv2 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

class cup:
    def __init__(self, x, y, w, h, pixels=None):
        self.x = int(x + (w/2))
        self.y = int(y + (h/2))
        self.w = int(w)
        self.h = int(h)
        self.area = self.w*self.h
        self.colour = self.get_colour(pixels)

    def get_colour(self, pixels):
        try:
            avg_color_per_row = np.average(pixels, axis=0)
            avg_color = np.average(avg_color_per_row, axis=0)
            return avg_color
        except:
            return None

class frame:
    def __init__(self, img):
        #image 
        self.img = img
        
        #shape of image
        self.height = img.shape[0]
        self.width = img.shape[1]
        self.channels = img.shape[2]
        
        #the output image, where the drawings go
        self.annotated = self.img.copy()

    def get_colours(self):
        '''
        Thresholds out dark and light pixels to leave only colour
        Quite specific to the original videos, idea is that theres lots of white and black in
        alpacas, sheep and goats. (and the environments)
        '''

        # smooth the image
        copy = cv2.blur(self.img, (75,75))

        hsv = cv2.cvtColor(copy, cv2.COLOR_BGR2HSV)
        
        #greyscale to get bright and dark values
        gray = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)
       
        # threshold all values out between 120 and 255
        _, mask = cv2.threshold(gray, 140, 200,cv2.THRESH_TOZERO)

        # dilate and morph the output of threshold. Attempt at closing a shape which has been sliced in half
        dilate_mask = cv2.dilate(mask,(10,10),iterations = 1)

        closed_mask = cv2.morphologyEx(dilate_mask, cv2.MORPH_CLOSE, (100, 100))

        res = cv2.bitwise_and(copy, copy, mask= closed_mask)

        return res

    def get_contours(self):
        mask = self.get_colours()
        # convert to greyscale
        grey = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        #threshold so all values above 0 are now 255
        _, thresh = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY)

        # get contours of blue objects in image
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        return contours
    
    def find_cups(self):
        contours = self.get_contours()
        cups = []
        if contours is not None:
            cv2.drawContours(self.annotated, contours, -1, (100, 255, 100), 2)
            for contour in contours:

                # get bounding rectangle for this contour
                x_i, y_i, w_i, h_i = cv2.boundingRect(contour)
                
                #calculate area
                area_i = w_i * h_i
                
                # if above a certain value -> it's part of the cup as theres more blue pixels together in 
                if area_i > 10000:
                    print(area_i)
                    #store values for this frame
                    i = cup(x_i, y_i, w_i, h_i)
                    cups.append(i)
        self.cups = cups
        return cups
        
    def draw_boxes(self):
        cups = self.find_cups()
        if cups is not None:
            for cup in cups:
                x = cup.x
                y = cup.y
                width = cup.w
                height = cup.h
                col = cup.colour

                if None not in[x, y, width, height]:
                    # circle for center
                    cv2.circle(self.annotated, (x, y), 5, (0,255,0), -1)
                    # rectangle around box
                    start = (int(x - (width/2)), int(y - int(width/2)))
                    end = (int(x + (width/2)), int(y + int(width/2)))
                    cv2.rectangle(self.annotated, pt1 =  start, pt2 = end, color = col, thickness = 3)

    def get_cup_vals(self, contours):
        xs = []
        ys = []
        ws = []
        hs = []
        if len(contours) > 0:
            
            for cnt in contours:
                x_i, y_i, w_i, h_i = cv2.boundingRect(cnt)
                xs.append(x_i)
                ys.append(y_i)
                ws.append(w_i)
                hs.append(h_i)
            avg_x = np.mean(xs)
            avg_y = np.mean(ys)
            avg_w = np.max(ws)
            avg_h = np.max(hs)
            centre_x = avg_x + (avg_w/2)
            centre_y = avg_y + (avg_h/2)
            return int(centre_x), int(centre_y), int(avg_w), int(avg_h)
        else:
            return None, None, None, None

    def detect(self):
        #abstract function
        pass

class detect_red_blue(frame):

    def __init__(self, img):
        #initialise frame
        super().__init__(img)
    
    def get_blue(self):
        '''
        return all blue pixels
        '''

        # segment out all colourful pixels
        colours = self.get_colours()
        #convert to hsv colour space
        hsv = cv2.cvtColor(colours, cv2.COLOR_RGB2HSV)
    
        # blue pixel bounds
        lower_blue = np.array([0,0,0])
        upper_blue = np.array([100,255,255])

        #mask image, all pixels within blue bounds turn to white else black
        blueImage = cv2.inRange(hsv, lower_blue, upper_blue)
        blue_res = cv2.bitwise_and(colours,colours, mask= blueImage)
        blue_res = cv2.GaussianBlur(blue_res, (15, 15), 2)
        return blue_res

    def get_red(self):
        '''
        Return all red pixels
        '''
        # segment out colours
        colours = self.get_colours()
        # convert to hsv colour space
        hsv = cv2.cvtColor(colours, cv2.COLOR_RGB2HSV)
    
        #pixel value bounds for red -> trial and error, very specific to this case for some reason
        lower_red = np.array([110,0,0])
        upper_red = np.array([200,255,255])
        
        #mask image, all pixels within pink bounds turn to white else black
        redImage = cv2.inRange(hsv, lower_red, upper_red)
        red_res = cv2.bitwise_and(colours,colours, mask= redImage)
        red_res = cv2.GaussianBlur(red_res, (15, 15), 2)
        return red_res

    def get_mask(self):
        red_mask = self.get_red()
        blue_mask = self.get_blue()
        mask = cv2.bitwise_or(red_mask, blue_mask)
        return red_mask, blue_mask, mask

    def get_contours(self):
        red_mask = self.get_red()
        blue_mask = self.get_blue()
        mask = cv2.bitwise_or(red_mask, blue_mask)
        # convert to greyscale
        grey = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        #threshold so all values above 0 are now 255
        _, thresh = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY)

        # get contours of blue objects in image
        contour, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contour

    def get_blue_center(self):
        '''
        Returns: Centre point of all blue pixels and the width and height of bounding box
        '''
        # segment out only blue pixels
        mask = self.get_blue()
        # convert to greyscale
        grey = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        #threshold so all values above 0 are now 255
        _, thresh = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY)

        # get contours of blue objects in image
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        big_contours = [i for i in contours if cv2.contourArea(i) > 1000]
        cv2.drawContours(self.annotated, big_contours, -1, (0, 255, 0), 3)
        return self.get_cup_vals(big_contours)
        
    def get_red_center(self):
        mask = self.get_red()
        grey = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        big_contours = [i for i in contours if cv2.contourArea(i) > 20000]
        cv2.drawContours(self.annotated, big_contours, -1, (0, 255, 0), 3)

        return self.get_cup_vals(big_contours)
        

    def detect(self):
        
        blue_x, blue_y, blue_width, blue_height = self.get_blue_center()
        red_x, red_y, red_width, red_height = self.get_red_center()

        # draw blue cup centre point & bounding area
        if None not in[blue_x, blue_y, blue_width, blue_height]:
            # circle for center
            cv2.putText(self.annotated, "Blue Cup", (blue_x - 50, blue_y- 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            cv2.circle(self.annotated, (blue_x, blue_y), 10, (0, 255, 0), -1)
           
        # draw red cup centre point & bounding area
        if None not in[red_x, red_y, red_width, red_height]:
            # circle for center
            cv2.putText(self.annotated, "Red Cup", (red_x - 50, red_y- 15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            cv2.circle(self.annotated, (red_x, red_y), 10, (0, 255, 0), -1)
           
        return blue_x, blue_y, red_x, red_y

class detect_same(frame):
    
    def __init__(self, img):
        #initialise frame
        super().__init__(img)


class detect_sizes(frame):
    
    def __init__(self, img):
        #initialise frame
        super().__init__(img)

    def detect(self):
        mask = self.get_colours()
        grey = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY)
        self.out = mask
        contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # print(len(contours))
        vals = self.get_cup_vals(contours)
        self.draw_boxes(vals)
    '''
    TODO
    '''
    pass

def menu() -> int:
    choosing = True
    menu = "Enter option:\n1) Colour cups\n2) Different Sizes\n3) Same cup\n Choice: "
    while choosing:
        option = input(menu)
        if int(option) in [1, 2, 3]:
            choosing = False
            return int(option)

def extract_vid_feats(in_path, out_path):
    #read in video
    choice = menu()
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
    red_writer= cv2.VideoWriter(f'{out_path}_red.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (width,height))
    blue_writer= cv2.VideoWriter(f'{out_path}_blue.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (width,height))
    mask_writer= cv2.VideoWriter(f'{out_path}_mask.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (width,height))
    while cap.isOpened():
        ret, img = cap.read()
        # img = cv2.imread("red_hsv.png")
        # ret = True
        
        if ret == True:
            
            frame_no +=1
            if choice == 1:
                im = detect_red_blue(img)
                blue_x, blue_y, red_x, red_y = im.detect()
                output['frame'].append(frame_no)
                output["x1"].append(blue_x)
                output["y1"].append(blue_y)
                output["x2"].append(red_x)
                output["y2"].append(red_y)
                # output["x2"].append(None)
                # output["y2"].append(None)
            if choice == 2:
                im = detect_sizes(img)
            if choice == 3:
                im = detect_same(img)

            
            red, blue, mask = im.get_mask()
            out_im = im.annotated

            scale = 50
            width = int(im.width * scale / 100)
            height = int(im.height * scale / 100)
            dim = (width, height)

            # red = im.get_red()
            resized_annotated = cv2.resize(out_im, dim, interpolation = cv2.INTER_AREA)
            resized_mask = cv2.resize(mask, dim, interpolation = cv2.INTER_AREA)

            cv2.imshow("Output", resized_annotated)
            main_writer.write(out_im)
            blue_writer.write(blue)
            red_writer.write(red)
            mask_writer.write(mask)
            cv2.imshow("Colour", resized_mask)


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
    df = pd.DataFrame.from_dict(output)
    return df

def plot_vals(df):
    fig, axs = plt.subplots(3, sharex=True)
    fig.suptitle("Location and size of landmark")

    frame_no = df['frame']
    area = df['area']
    xs = df['x1']
    ys = df['y']
    
    axs[0].plot(frame_no, area, "tab:orange")
    axs[0].set(ylabel="Area")

    axs[1].plot(frame_no, xs, "tab:blue")
    axs[1].set(ylabel="x")

    axs[2].plot(frame_no, ys, "tab:green")
    axs[2].set(ylabel="y")
    axs[2].set(xlabel="Time")

    plt.show()

    return fig

if __name__ == '__main__':
    
    ### CHANGE THIS STRING FOR EACH VIDEO, NOT AUTOMATED YET ###
    vid = 'videos/colour/G134_E5.mp4'
    ############################################################
    
    file = vid.split("/")[-1][:-4]
    csv_path = f"csv/{file}_poi.csv"
    output_vid_path = f"out_vid/{file}"
    df = extract_vid_feats(vid, output_vid_path)
    df.to_csv(csv_path, index=False)

    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots()

    x1 = df['x1']
    x2 = df['x2']
    frame_no = df['frame']

    plt.plot(frame_no, x1, color='b', label="blue cup")
    plt.plot(frame_no, x2, color='r', label = "red cup")
    plt.ylim(ymin=0, ymax=2000)
    plt.ylabel("X location/pixels")
    plt.xlabel("Frame number")
    plt.title("X location of red and blue cups")
    plt.legend(loc="best")
    plt.savefig(f"plots/{file}_plot.png")
    plt.show()