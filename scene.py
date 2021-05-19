import random
from reflectors import *

class Scene:

    def __init__(self):
        self.reflectors = []
        self.reflector_count = 0
        self.noise = 0
        self.clutter = 0

    def add_reflector(self, reflector, location):
        reflector.location = location
        self.reflectors.append(reflector)
        self.reflector_count = self.reflector_count+1

    def remove_reflector(self, reflector):
        self.reflectors.remove(reflector)
        self.reflector_count = self.reflector_count-1

    def print_reflectors(self):
        if not self.reflectors:
            print("empty")
        for reflector in self.reflectors:
            print(reflector.reflector_type + ": " + str(reflector.location))
            reflector.print_attributes()

    def plot_scene(self):
        pass

    def scene_rcs(self, thetas, frequencies):
        freq_angle = []
        for freq in frequencies:

            power_angle = []
            for theta in thetas:
                power = 0
                for ref in self.reflectors:
                    power += ref.rcs(freq, theta)
                power_angle.append(power)
            freq_angle.append(power_angle)
        return freq_angle

    def clear_all_reflectors(self):
        self.reflectors = []
        self.reflector_count = 0

    def add_random_reflectors(self, num_of_reflectors):

        for i in range(num_of_reflectors):
            j = random.randint(1, 3)
            loc = random.randint(20, 80)

            if j == 1:
                ref = PlateReflector()
                ref.randomize()
                self.add_reflector(ref, loc)

            elif j == 2:
                ref = CylinderReflector()
                ref.randomize()
                self.add_reflector(ref, loc)

            else:
                ref = TrihedralReflector()
                ref.randomize()
                self.add_reflector(ref, loc)
