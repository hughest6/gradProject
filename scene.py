import random
import numpy as np
from reflectors import *
from rcs_stats import *
from sklearn.preprocessing import normalize
#import matplotlib.pyplot as plt
#from matplotlib import cm


class Scene:

    def __init__(self, frequencies, thetas, snr=1):
        self.frequencies = frequencies
        self.thetas = thetas
        self.reflectors = []
        self.reflector_count = 0
        self.noise = 0
        self.clutter = 0
        self.snr = snr

    def set_snr(self, new_snr):
        self.snr = new_snr

    def add_reflector(self, reflector, location=0):
        reflector.location = location
        self.reflectors.append(reflector)
        self.reflector_count = self.reflector_count+1

    def remove_reflector(self, reflector):
        self.reflectors.remove(reflector)
        self.reflector_count = self.reflector_count-1

    def print_reflectors(self):
        if not self.reflectors:
            print("empty")
        print("shape: " + str(len(self.frequencies)) + " x " + str(len(self.thetas)))
        for reflector in self.reflectors:
            print(reflector.reflector_type + ": " + str(reflector.location))
            reflector.print_attributes()

    def scene_rcs(self, add_awgn_noise=True, add_clutter=False, clutter_factor=2):
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
        freq_angle = normalize(freq_angle, axis=0)
        f_list = [item for sublist in freq_angle for item in sublist]

        awgn = 0
        clutter = 0
        if add_awgn_noise:
            awgn = self.add_awgn_noise(freq_angle, self.snr)
        if add_clutter:
            clutter = self.add_weibull_noise(freq_angle, self.snr, a_param=clutter_factor)
        min_val = min(f_list)
        freq_angle = freq_angle + awgn + clutter
        freq_angle = freq_angle + abs(min_val)
        freq_angle = normalize(freq_angle, axis=0)
        return freq_angle

    # def plot_scene(self, add_noise=False):
    #     rcs = self.scene_rcs(add_noise)
    #     x, y = np.meshgrid(np.array(self.frequencies), np.array(self.thetas))
    #     z = np.array(rcs)
    #     z = z.reshape(x.shape)
    #     fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    #     surf = ax.plot_surface(x, y, z, cmap=cm.get_cmap('plasma'),
    #                            linewidth=0, antialiased=False)
    #     plt.show()

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
    def add_awgn_noise(signal, target_snr_db):
        sig_avg_watts = np.mean(signal)
        sig_avg_db = 10 * np.log10(sig_avg_watts)
        # Calculate noise according to [2] then convert to watts
        noise_avg_db = sig_avg_db - target_snr_db
        noise_avg_watts = 10 ** (noise_avg_db / 10)
        # Generate an sample of white noise
        mean_noise = 0
        noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), signal.shape)
        # Noise up the original signal
        return noise_volts

    @staticmethod
    def add_weibull_noise(signal, target_snr_db, a_param=2):
        sig_avg_watts = np.mean(signal)
        sig_avg_db = 10 * np.log10(sig_avg_watts)
        # Calculate noise according to [2] then convert to watts
        noise_avg_db = sig_avg_db - target_snr_db
        noise_avg_watts = 10 ** (noise_avg_db / 10)
        # Generate an sample of weibull noise & find average power
        noise_volts = np.random.weibull(a_param, signal.shape)
        weibull_noise_power = np.mean(noise_volts)
        power_factor = noise_avg_watts / weibull_noise_power
        noise_volts = noise_volts * power_factor
        # Noise up the original signal
        return noise_volts

