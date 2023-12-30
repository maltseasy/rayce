from __future__ import annotations
import math

class Vec:
    def __init__(self, degrees: float = 0, magnitude: float = 0):
        self.degrees = degrees
        self.magnitude = magnitude

    def components(self):
        x = self.magnitude * math.sin(math.radians(self.degrees))
        y = self.magnitude * math.cos(math.radians(self.degrees))
        return (x, y)
    
    def from_components(self, components: tuple[float, float]):
        self.magnitude = math.sqrt(components[0]**2 + components[1]**2)

        degree_offset = 0
        opposite = 0
        adjacent = 0
        
        if (components[0] * components[1]) > 0: # q1 or q3
            opposite = components[0]
            adjacent = components[1]
            if components[0] > 0: #q1
                degree_offset = 0
            else: #q3
                degree_offset = 180
        else: # q2 or q4
            opposite = components[1]
            adjacent = components[0]
            if components[0] > 0: #q4
                degree_offset = 90
            else: #q2
                degree_offset = 270
    
        if adjacent == 0:
            self.degrees = degree_offset + 90
        else:
            self.degrees = degree_offset + math.degrees(math.atan(abs(opposite/adjacent)))

    def add(self, vec: Vec):
        v1 = self.components()
        v2 = vec.components()
        v3 = Vec(0, 0)

        v3.from_components((v1[0] + v2[0], v1[1] + v2[1]))
        return v3

    def mult(self, scalar: float):
        out = Vec(self.degrees, self.magnitude)
        out.magnitude *= scalar
        return out
    
    def normalize(self):
        out = Vec(self.degrees, self.magnitude)
        out.magnitude = 1
        return out

    def dot(self, vec: Vec):
        v1 = self.components()
        v2 = vec.components()

        return v1[0]*v2[0] + v1[1]*v2[1]

    def proj(self, vec: Vec):
        alpha = abs(self.degrees-vec.degrees)

        if (alpha <= 90):
            return Vec(vec.degrees, self.dot(vec.normalize()))
        else:
            return Vec((vec.degrees+180)%360, self.dot(vec.normalize()))

    def __str__(self):
        return f"Dir: {self.degrees} | Mag: {self.magnitude}"
    
    def __repr__(self):
        return f"Dir: {self.degrees} | Mag: {self.magnitude}"