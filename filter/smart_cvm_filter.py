import numpy as np
import math
import cv2
import copy
from .filter_constant_velocity import CVM_Filter

class Smart_CVM_Filter(CVM_Filter):

    def __init__(self, Ts, process_noise = 1.0, sensor_noise = 0.001, name="CVM Filter", dynamic_process_noise=None, smart_prediction=True):
        super().__init__(Ts, process_noise, sensor_noise, name="Smart CVM Filter")
        self.radius = 25
        self.dynamic_process_noise = dynamic_process_noise
        self.smart_prediction = smart_prediction
        self.normal_process_noise = self.process_noise
        self.name = name
    
    def setRadius(self, radius):
        self.radius = radius
        return self

    def setBoundaries(self, xLeft, xRight, yTop, yBot):
        self.xLeft = xLeft
        self.xRight = xRight
        self.yTop = yTop
        self.yBot = yBot
        return self


    def dofilter(self, y1, y2, radius=None):

        if radius is not None:
            self.radius = radius
        
        if self.dynamic_process_noise is not None:
            if self.is_near_bank():
                self.setProcessNoise(self.dynamic_process_noise)
            else:
                self.setProcessNoise(self.normal_process_noise)

        super(Smart_CVM_Filter, self).dofilter(y1,y2)

        if self.smart_prediction:
            def applyReflectionTopBottom():
                self.x_post[3, 0] = self.x_post[3, 0] * -1

            def applyReflectionLeftRight():
                self.x_post[1, 0] = self.x_post[1, 0] * -1

            def resetHits():
                self.can_hit_right_bank = True
                self.can_hit_bottom_bank = True
                self.can_hit_left_bank = True
                self.can_hit_top_bank = True

            resetHits()

            if self.xhat[0] + self.radius > self.xRight and self.can_hit_right_bank:
                resetHits()
                self.can_hit_right_bank = False
                applyReflectionLeftRight()
            if self.xhat[1] + self.radius > self.yBot and self.can_hit_bottom_bank:
                resetHits()
                self.can_hit_bottom_bank = False
                applyReflectionTopBottom()
            if self.xhat[1] - self.radius < self.yTop and self.can_hit_top_bank:
                resetHits()
                self.can_hit_top_bank = False
                applyReflectionTopBottom()
            if self.xhat[0] - self.radius < self.xLeft and self.can_hit_left_bank:
                resetHits()
                self.can_hit_left_bank = False
                applyReflectionLeftRight()

        return self.xhat

    def is_near_bank(self, detection_radius=2.0):
        if self.xhat[1] + self.radius * detection_radius >= self.yBot:
            return True
        if self.xhat[1] - self.radius * detection_radius <= self.yTop:
            return True
        if self.xhat[0] - self.radius * detection_radius <= self.xLeft:
            return True
        if self.xhat[0] + self.radius * detection_radius >= self.xRight:
            return True
        return False