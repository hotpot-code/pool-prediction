import numpy as np
import math
import cv2
import copy

class CAM_Filter():
    def __init__(self, Ts, process_noise = 1.0, sensor_noise = 0.001):
        self.process_noise = process_noise
        self.xhat = 0
        # init für x posteriori (geschätzte Werte)
        self.x_post = np.matrix([
            [0], # Start x Position
            [0], # Start x Geschwindigkeit
            [0], # Start x Beschleunigung
            [0], # Start y Position
            [0], # Start y Geschwindigkeit
            [0] # Start y Beschleunigung
        ])
        #init für P posteriori (geschätze Abweichung) groß machen
        self.P_post = np.diag([1000, 1000, 1000, 1000, 1000, 1000])
        self.Ts = Ts
        self.sensor_noise = sensor_noise

         # Messrauschen
        self.R = np.diag([self.sensor_noise, self.sensor_noise]) ** 2
        
        self.setProcessNoise(self.process_noise)

        # Zustandstransfermatrix (constant acceleration)
        self.Ad = np.matrix([[1, self.Ts, (self.Ts**2) / 2.0, 0, 0, 0],
                        [0, 1, self.Ts, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, self.Ts, (self.Ts**2) / 2.0],
                        [0, 0, 0, 0, 1, self.Ts],
                        [0, 0, 0, 0, 0, 1]])
        
        self.Gd = np.matrix([
            [(self.Ts**3) / 6.0, 0],
            [(self.Ts**2) / 2.0, 0],
            [self.Ts, 0],
            [0, (self.Ts**3) / 6.0],
            [0, (self.Ts**2) / 2.0],
            [0, self.Ts]
        ])
        
        # Umwandlung von (sx, vx, ax, sy, vy, ay) zu (sx, sy, sx, sy)
        self.C = np.matrix([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0]
        ])

    def setProcessNoise(self, process_noise):
        self.process_noise = process_noise
        # Prozessrauschen
        self.Q = np.eye(2) * self.process_noise ** 2

    def dofilter(self, y1, y2):
        #Hier bitte Ihre Filterimplementierung
        
        # x priori
        x_prior = self.Ad * self.x_post
        # P priori
        P_prior = self.Ad * self.P_post * self.Ad.T + self.Gd * self.Q * self.Gd.T
                
        S = self.C * P_prior * self.C.T + self.R
        K = P_prior * self.C.T * (S**-1)
        
        if (y1 is not None):
            # Messwert Array zu Matrix
            y = np.matrix([
                [y1],
                [y2]
            ])
            self.x_post = x_prior + K * (y - self.C * x_prior)
            self.P_post = (np.eye(6) - K * self.C) * P_prior
        else:
            self.x_post = x_prior
            self.P_post = P_prior
        
        # Schätzung der Systemmesswerte
        self.xhat = np.array([self.x_post[0,0], self.x_post[3,0]])

        #self.xhat = y1_xy
        
        return self.xhat

    def getPredictions(self, max_count=500):

        kalman_copy = copy.copy(self)

        prePos = []
        preVar = []

        while len(prePos) < max_count:
            kalman_copy.dofilter(None, None)
            
            prePos.append([int(kalman_copy.xhat[0]), int(kalman_copy.xhat[1])])
            preVar.append([int(kalman_copy.P_post[0, 0]), int(kalman_copy.P_post[3, 3])])
  
            
        return (prePos, preVar)

    def py_ang(self, v1, v2):
        dot = v1[0]*v2[0] + v1[1]*v2[1]      # dot product
        det = v1[0]*v2[1] - v1[1]*v2[0]      # determinant
        angle = math.atan2(det, dot)  # atan2(y, x) or atan2(sin, cos)
        return angle