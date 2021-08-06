import random
import numpy as np
from reflectors import *
from rcs_stats import *
from sklearn.preprocessing import normalize


class Scene:

    def __init__(self, frequencies, thetas):
        self.frequencies = frequencies
        self.thetas = thetas
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

    def scene_rcs(self, add_noise=False):
        freq_angle = []
        for freq in self.frequencies:

            power_angle = []
            for theta in self.thetas:
                power = 0
                for ref in self.reflectors:
                    power += ref.rcs(freq, theta)
                power_angle.append(power)
            freq_angle.append(power_angle)
        freq_angle = np.transpose(freq_angle)
        if add_noise:
            freq_angle = self.add_noise(freq_angle, -30)
        freq_angle = normalize(freq_angle, axis=0)
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

    def scene_statistics(self):
        rcs = self.scene_rcs(True)
        rcs_info = basic_stats(rcs)
        return rcs_info

    @staticmethod
    def add_noise(signal, target_snr_db):
        sig_avg_watts = np.mean(signal)
        sig_avg_db = 10 * np.log10(sig_avg_watts)
        # Calculate noise according to [2] then convert to watts
        noise_avg_db = sig_avg_db - target_snr_db
        noise_avg_watts = 10 ** (noise_avg_db / 10)
        # Generate an sample of white noise
        mean_noise = 0
        noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), signal.shape)
        # Noise up the original signal
        return signal + noise_volts

