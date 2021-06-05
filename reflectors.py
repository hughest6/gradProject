import math
import random

pi = math.pi


class Reflector:

    def __init__(self, reflector_type="", location=0):
        self.reflector_type = reflector_type
        self.location = location


class PlateReflector(Reflector):

    def __init__(self, plate_area=1):
        super().__init__("plate")
        self.plate_area = plate_area

    def rcs(self, *args):
        freq = args[0]
        theta = args[1]
        if theta == 0:
            theta = 1E-12
        gamma = 3E8 / freq
        k = 2
        return (4 * pi * (self.plate_area ** 4)) / (gamma ** 2) * (
                (math.sin(k * self.plate_area * math.sin(theta))) / (k * self.plate_area * math.sin(theta)))

    def randomize(self):
        self.plate_area = 1 + (random.random()*5)

    def print_attributes(self):
        print("plate area: " + str(self.plate_area))


class CylinderReflector(Reflector):

    def __init__(self, cyl_area=1, cyl_length=1):
        super().__init__("cylinder")
        self.cyl_area = cyl_area
        self.cyl_length = cyl_length

    def rcs(self, *args):
        freq = args[0]
        gamma = 3E8 / freq
        return (2 * pi * self.cyl_area * (self.cyl_length ** 2)) / (gamma ** 2)

    def randomize(self):
        self.cyl_area = 1 + (random.random()*5)
        self.cyl_length = 1 + (random.random()*5)

    def print_attributes(self):
        print("cylinder area: " + str(self.cyl_area) + " cylinder length: " + str(self.cyl_length))


class TrihedralReflector(Reflector):

    def __init__(self, area=0):
        super().__init__("trihedral")
        self.area = area

    def rcs(self, *args):
        freq = args[0]
        gamma = 3E8 / freq
        return (4 * pi * (self.area**4)) / (3 * (gamma ** 2))

    def randomize(self):
        self.area = 1 + (random.random()*5)

    def print_attributes(self):
        print("trihedral area: " + str(self.area))
