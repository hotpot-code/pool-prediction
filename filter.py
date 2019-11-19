import numpy as np

class MyFilter():
    def __init__(self, Ts, process_noise = 1.0):
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

    def dofilter(self, y1, y2):
        #Hier bitte Ihre Filterimplementierung
        
        # Messrauschen
        abweichung_sensor = 0.01 ** 2
        R = np.diag([abweichung_sensor, abweichung_sensor])
        
        # Prozessrauschen
        Q = np.eye(2) * self.process_noise ** 2

        # Zustandstransfermatrix (constant acceleration)
        Ad = np.matrix([[1, self.Ts, (self.Ts**2) / 2.0, 0, 0, 0],
                        [0, 1, self.Ts, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, self.Ts, (self.Ts**2) / 2.0],
                        [0, 0, 0, 0, 1, self.Ts],
                        [0, 0, 0, 0, 0, 1]])
        
        Gd = np.matrix([
            [(self.Ts**3) / 6.0, 0],
            [(self.Ts**2) / 2.0, 0],
            [self.Ts, 0],
            [0, (self.Ts**3) / 6.0],
            [0, (self.Ts**2) / 2.0],
            [0, self.Ts]
        ])
        
        # Umwandlung von (sx, vx, ax, sy, vy, ay) zu (sx, sy, sx, sy)
        C = np.matrix([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0]
        ])
        
        # x priori
        x_prior = Ad * self.x_post
        # P priori
        P_prior = Ad * self.P_post * Ad.T + Gd * Q * Gd.T
                
        S = C * P_prior * C.T + R
        K = P_prior * C.T * (S**-1)
        
        if (y1 is not None):
            # Messwert Array zu Matrix
            y = np.matrix([
                [y1],
                [y2]
            ])
        else:
            y =  C * x_prior
        
        self.x_post = x_prior + K * (y - C * x_prior)
        self.P_post = (np.eye(6) - K * C) * P_prior
                        
        # Schätzung der Systemmesswerte
        self.xhat = np.array([self.x_post[0,0], self.x_post[3,0]])

        #self.xhat = y1_xy
        
        return self.xhat

    def getPredictionAfterSec(self, sec):
        # Zustandstransfermatrix (constant acceleration)
        Ad = np.matrix([[1, sec, (sec**2) / 2.0, 0, 0, 0],
                        [0, 1, sec, 0, 0, 0],
                        [0, 0, 1, 0, 0, 0],
                        [0, 0, 0, 1, sec, (sec**2) / 2.0],
                        [0, 0, 0, 0, 1, sec],
                        [0, 0, 0, 0, 0, 1]])
        x_prior = Ad * self.x_post
        return np.array([x_prior[0,0], x_prior[3,0]])