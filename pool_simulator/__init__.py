# import the necessary packages
import numpy as np
import math
import cv2
from .Ball import Ball
from .Bank import Bank

class PoolSimulation():

    def __init__(self, start_position = (960,540), start_angle = math.pi/2 + 0.3, start_velocity = 900, friction = 0.06, seconds = 0.16):
        self.ball = Ball(start_position, 25, start_velocity, friction)
        self.ball.setRotation(start_angle)
        self.bank_left = Bank((100,100), (100, 980))
        self.can_hit_left_bank = True
        self.bank_right = Bank((1820,100), (1820, 980))
        self.can_hit_right_bank = True
        self.bank_top = Bank((100,100), (1820, 100))
        self.can_hit_top_bank = True
        self.bank_bottom = Bank((100,980), (1820, 980))
        self.can_hit_bottom_bank = True
        self.game_objects = [self.ball, self.bank_left, self.bank_right, self.bank_bottom, self.bank_top]
        self.bank_hits = 0
        self.frame_no = 0
        self.seconds = seconds
        self.isBallMoving = True
        self.bankDetectionRadius = 2
        self.isBallNearBank = False
    
    def resetHitFlags(self):
        self.can_hit_left_bank = True
        self.can_hit_right_bank = True
        self.can_hit_top_bank = True
        self.can_hit_bottom_bank = True

    
    def update(self):
        # Get frame from video
        frame = np.zeros((1080, 1920, 3), np.uint8)

        self.ball.update(self.seconds)

        if (self.ball.position[1] + self.ball.size >= self.bank_bottom.p1[1] and self.can_hit_bottom_bank):
            angle = math.atan2(1, 0) - math.atan2(self.ball.forward[1], self.ball.forward[0])
            self.ball.setRotation(angle)
            self.bank_hits += 1
            self.resetHitFlags()
            self.can_hit_bottom_bank = False
        if (self.ball.position[1] - self.ball.size <= self.bank_top.p1[1] and self.can_hit_top_bank):
            angle = math.atan2(-1, 0) - math.atan2(self.ball.forward[1], self.ball.forward[0])
            self.ball.setRotation(angle + math.pi)
            self.bank_hits += 1
            self.resetHitFlags()
            self.can_hit_top_bank = False
        if (self.ball.position[0] - self.ball.size <= self.bank_left.p1[0] and self.can_hit_left_bank):
            angle = math.atan2(0, -1) - math.atan2(self.ball.forward[1], self.ball.forward[0])
            self.ball.setRotation(angle + math.pi / 2)
            self.bank_hits += 1
            self.resetHitFlags()
            self.can_hit_left_bank = False
        if (self.ball.position[0] + self.ball.size >= self.bank_right.p1[0] and self.can_hit_right_bank):
            angle = math.atan2(0, 1) - math.atan2(self.ball.forward[1], self.ball.forward[0])
            self.ball.setRotation(angle - math.pi / 2)
            self.bank_hits += 1
            self.resetHitFlags()
            self.can_hit_right_bank = False

        self.isBallNearBank = False
        if self.ball.position[1] + self.ball.size * self.bankDetectionRadius >= self.bank_bottom.p1[1]:
            self.isBallNearBank = True
        if self.ball.position[1] - self.ball.size * self.bankDetectionRadius <= self.bank_top.p1[1] :
            self.isBallNearBank = True
        if self.ball.position[0] - self.ball.size * self.bankDetectionRadius <= self.bank_left.p1[0]:
            self.isBallNearBank = True
        if self.ball.position[0] + self.ball.size * self.bankDetectionRadius >= self.bank_right.p1[0]:
            self.isBallNearBank = True
        
        for game_object in self.game_objects:
            game_object.render(frame)

        velocity_vector = (int(self.ball.forward[0] * self.ball.velocity), int(self.ball.forward[1] * self.ball.velocity))

        if self.ball.velocity == 0:
            self.isBallMoving = False

        return (frame, self.ball.position, velocity_vector)
