import cv2
import numpy as np
import math

class Ball():

    def __init__(self, position, size = 2, start_velocity = 900, friction = 0.06, noise=2.0):
        self.forward = np.array([0, -1])
        self.position = position
        self.size = size
        self.velocity = start_velocity
        self.friction = friction
        self.noise = noise

    def update(self, delta_time):
        # http://web.cs.iastate.edu/~jia/papers/billiard-analysis.pdf
        self.acceleration = -(5/7) * self.friction * 9.81
        if self.velocity > 10:
            self.velocity += self.acceleration * delta_time
        else:
            self.velocity = 0
        self.moveWithVelocity(self.velocity, delta_time)

    def render(self, canvas):
        int_position = (int(round(self.position[0])), int(round(self.position[1])))
        cv2.circle(canvas, int_position, self.size, (255, 255, 255), -1)

    def moveWithVelocity(self, velocity, delta_time):
        directed_velocity = self.forward * velocity
        new_x = self.position[0] + directed_velocity[0] * delta_time
        new_y = self.position[1] + directed_velocity[1] * delta_time
        self.position = (new_x, new_y)

    def setRotation(self, angle):
        forward = np.array([0, -1])
        new_x = math.cos(angle) * forward[0] - math.sin(angle) * forward[1]
        new_y = math.sin(angle) * forward[0] + math.cos(angle) * forward[1]
        self.forward = np.array([new_x, new_y])

    def getVelocity(self):
        return self.forward * self.velocity

    def getNoisedPosition(self):
        R = np.diag([self.noise, self.noise]) ** 2
        noised_position = np.random.multivariate_normal(np.array(self.position).flatten(), R)
        return noised_position
