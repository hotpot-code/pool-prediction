import cv2
import imutils

from BallDetection import BallDetection 

class VideoHandler():
    def __init__(self, name):
        self.name = name

        ## whiteUpper & whiteLower are the boundaries of the respective ball

        if name == "pool_1":
            self.alpha = 1 # Contrast control (1.0-3.0)
            self.beta = 10 # Brightness control (0-100)
            self.whiteLower = (15, 30, 1)
            self.whiteUpper = (30, 105, 255)
            self.frame_number = 1 #start
            self.radiusMin = 3
            self.radiusMax = 11

            #Origin (0,0) #  tbd
            self.xLeft = 35
            self.xRight = 1245
            self.yTop = 38
            self.yBot = 685

        elif name =="pool_3":
            self.alpha = 1 # Contrast control (1.0-3.0)
            self.beta = 10 # Brightness control (0-100)
            self.whiteLower = (20, 0, 150)
            self.whiteUpper = (45, 165, 255)
            self.frame_number = 870 #start
            self.radiusMin = 3
            self.radiusMax = 11

            #Origin (0,0)
            self.xLeft = 18
            self.xRight = 582
            self.yTop = 20
            self.yBot = 320

        elif name == "pool_4":
            self.alpha = 1 # Contrast control (1.0-3.0)
            self.beta = 10 # Brightness control (0-100)
            self.whiteLower = (20, 0, 150)
            self.whiteUpper = (45, 165, 255)
            self.frame_number = 30 #start
            self.radiusMin = 3
            self.radiusMax = 11

            #origin (0,0)
            self.xLeft = 18
            self.xRight = 582
            self.yTop = 20
            self.yBot = 320

        elif name == "pool_5":
            self.alpha = 0.9 # Contrast control (1.0-3.0)
            self.beta = 20 # Brightness control (0-100)
            #self.whiteLower = (0, 0, 0) #8-ball
            #self.whiteUpper = (30, 255, 100) #8-ball
            self.whiteLower = (20, 0, 150)
            self.whiteUpper = (45, 165, 255)
            self.frame_number = 30
            self.radiusMin = 3
            self.radiusMax = 11

            #Origin (0,0)
            self.xLeft = 18
            self.xRight = 582
            self.yTop = 20
            self.yBot = 320

        else: #default: Pool 
            self.name = "pool"
            self.whiteLower = (15, 30, 1)
            self.whiteUpper = (30, 105, 255)
            self.alpha = 1 # Contrast control (1.0-3.0)
            self.beta = 10 # Brightness control (0-100)
            self.frame_number = 1 #start
            self.radiusMin = 3
            self.radiusMax = 11

            #Origin (0,0) # tbd
            self.xTop = 35
            self.xBot = 1245
            self.yLeft = 38
            self.yRight = 685

        #self.whiteBallDetection = BallDetection(whiteLower, whiteUpper, 3, 11)
        self.vs = cv2.VideoCapture("videos/" + name + ".mp4")

        self.vs.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number-1);

    def giveFrames(self):

        # Get frame from video
        frame = self.vs.read()
        return frame[1]

    def cropFrame(self, frame):
        # crop image
        if self.name == "pool_1":
            #frame = frame[60:620, 100:1150]
            frame = frame
        elif self.name == "pool_3":
            frame = frame
        elif self.name == "pool_4":
            frame = frame
        elif self.name == "pool_5":
            frame = frame
        else:
            frame = frame[60:620, 100:1150]

        frame = imutils.resize(frame, width=600)
        return frame

    def giveParameters(self):
        return (self.alpha, self.beta, self.whiteLower, self.whiteUpper, self.radiusMin, self.radiusMax)
