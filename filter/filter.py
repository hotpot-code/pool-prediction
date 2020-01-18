import numpy as np
import cv2

class MyFilter():
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
        else:
            y =  self.C * x_prior
        
        self.x_post = x_prior + K * (y - self.C * x_prior)
        self.P_post = (np.eye(6) - K * self.C) * P_prior
                        
        # Schätzung der Systemmesswerte
        self.xhat = np.array([self.x_post[0,0], self.x_post[3,0]])

        #self.xhat = y1_xy
        
        return self.xhat

    def getPredictions(self, count=20):
        prePos = []
        preVar = []
        x_post_temp = self.x_post.copy()
        P_post_temp = self.P_post.copy()
        for i in range(count):
            x_post_temp = self.Ad * x_post_temp
            P_post_temp = self.Ad * P_post_temp * self.Ad.T + self.Gd * self.Q * self.Gd.T
            
            xhat_temp = np.array([x_post_temp[0,0], x_post_temp[3,0]])
            
            #predictions.append([[int(xh[0]), int(xh[1])], [int(pt[0][0]), int(pt[2][2])]])
            prePos.append([int(xhat_temp[0]), int(xhat_temp[1])])
            preVar.append([int(P_post_temp[0, 0]), int(P_post_temp[3, 3])])
        return (prePos, preVar)