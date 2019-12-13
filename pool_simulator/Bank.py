import cv2
import numpy as np
import math

class Bank():

    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2

    def update(self, delta_time):
        pass

    def render(self, canvas):
        cv2.line(canvas, self.p1, self.p2, (255, 255, 255), 2)
