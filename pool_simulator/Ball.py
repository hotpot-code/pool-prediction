import cv2
import numpy as np
import math

class Ball():

    def __init__(self, position, size = 2, start_velocity = 900, friction = 0.07):
        self.forward = np.array([0, -1])
        self.position = position
        self.size = size
        self.velocity = start_velocity
        self.frition = friction

    def update(self, delta_time):
        if self.velocity > 0:
            self.velocity += -1 * self.frition * delta_time
        else:
            self.velocity = 0
        self.moveWithVelocity(self.velocity, delta_time)

    def render(self, canvas):
        cv2.circle(canvas, self.position, self.size, (255, 255, 255), -1)

    def moveWithVelocity(self, velocity, delta_time):
        directed_velocity = self.forward * velocity
        new_x = self.position[0] + int(directed_velocity[0] * (delta_time / 1000))
        new_y = self.position[1] + int(directed_velocity[1] * (delta_time / 1000))
        self.position = (new_x, new_y)

    def setRotation(self, angle):
        forward = np.array([0, -1])
        new_x = math.cos(angle) * forward[0] - math.sin(angle) * forward[1]
        new_y = math.sin(angle) * forward[0] + math.cos(angle) * forward[1]
        self.forward = np.array([new_x, new_y])
