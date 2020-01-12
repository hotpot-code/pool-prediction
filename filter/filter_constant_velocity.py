import numpy as np

class MyFilter():
    def __init__(self, Ts, process_noise = 1.0, sensor_noise = 0.001):
        self.process_noise = process_noise
        self.sensor_noise = sensor_noise
        self.xhat = 0
        # init für x posteriori (geschätzte Werte)
        self.x_post = np.array([
            [0], # Start x Position
            [0], # Start x Geschwindigkeit
            [0], # Start y Position
            [0], # Start y Geschwindigkeit
        ])
        #init für P posteriori (geschätze Abweichung) groß machen
        self.P_post = np.eye(4) * 100000000
        self.Ts = Ts

    def dofilter(self, y1, y2):
        #Hier bitte Ihre Filterimplementierung
        
        # Messrauschen
        abweichung_sensor = self.sensor_noise ** 2
        R = np.eye(2) * abweichung_sensor
        
        # Prozessrauschen
        Q = np.eye(2) * self.process_noise ** 2

        # Zustandstransfermatrix (constant acceleration)
        Ad = np.array([[1, self.Ts, 0, 0],
                        [0, 1, 0, 0, ],
                        [0, 0, 1, self.Ts],
                        [0, 0, 0, 1]])
        
        Gd = np.array([
            [(self.Ts**2) / 2, 0],
            [self.Ts, 0],
            [0, (self.Ts**2) / 2],
            [0, self.Ts]
        ])
        
        # Umwandlung von (sx, vx, ax, sy, vy, ay) zu (sx, sy, sx, sy)
        C = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0]
        ])
        
        # x priori
        x_prior = np.matmul(Ad, self.x_post)
        # P priori
        P_prior = np.matmul(Ad, np.matmul(self.P_post, Ad.T)) + np.matmul(Gd, np.matmul(Q, Gd.T))
                
        S = np.matmul(C, np.matmul(P_prior, C.T)) + R
        K = np.matmul(P_prior, np.matmul(C.T, np.linalg.inv(S)))
        
        if (y1 is not None):
            # Messwert Array zu Matrix
            y = np.array([
                [y1],
                [y2]
            ])
        else:
            y =  np.matmul(C, x_prior)
        
        self.x_post = x_prior + np.matmul(K, (y - np.matmul(C, x_prior)))
        self.P_post = np.dot((np.eye(4) - np.dot(K, C)), P_prior)
        # Schätzung der Systemmesswerte
        self.xhat = np.array([self.x_post[0,0], self.x_post[2,0]])

        #self.xhat = y1_xy

        return self.xhat

    def get_p_post(self):
        return self.P_post

    def getPredictionAfterSec(self, sec):
        # Zustandstransfermatrix (constant acceleration)
        Ad = np.matrix([[1, sec, 0, 0, ],
                        [0, 1, 0, 0,],
                        [0, 0, 1, sec],
                        [0, 0, 0, 1]])
        x_prior = Ad * self.x_post
        return np.array([x_prior[0,0], x_prior[2,0]])