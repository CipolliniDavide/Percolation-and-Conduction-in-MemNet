import scipy.signal as signal
import random
import numpy as np
from .mem_parameters import volt_param, sim_param

class ControlSignal():

    def __init__(self, sources, sim_param=sim_param, volt_param=volt_param):
        self.volt_param = volt_param
        self.sim_param = sim_param
        if self.volt_param.VSTART == 'rand':
            self.VSTART = random.uniform(self.volt_param.VMIN, self.volt_param.VMAX)
        else:
            self.VSTART = volt_param.VSTART

        self.t_list = np.arange(0, sim_param.T + 1 / sim_param.sampling_rate, 1 / sim_param.sampling_rate)  # [s]
        self.src = sources
        # Initialize state and Voltage Input
        #self.V_list = [[] for s in range(len(self.src))]  # V_list[i] is the input on src[i]
        #self.V_list[0] = np.asarray([self.VSTART if t <= 2e-3 else 0.1 for t in self.t_list])
        self.V_list = np.zeros(shape=(len(self.src), len(self.t_list)))
        for i in range(len(self.V_list)):
            # self.V_list[i] = np.asarray([self.VSTART * (i+1) if t <= 2e-3 else 0.1 for t in self.t_list]).transpose()
            self.V_list[i] = np.asarray([self.VSTART * (i+1) if t <= 15e-3 else 0.1 for t in self.t_list]).transpose()

        self.index_to_change = self.V_list[0] != .1

        self.current_amplitude = self.VSTART

    def square_ramp(self, StartAmplitude=1, number_of_cycles=5):
        signal_freq = number_of_cycles / self.sim_param.T
        s = StartAmplitude / 2 + StartAmplitude / 2 * signal.square(2 * np.pi * signal_freq * self.t_list)
        count = (s.sum() - StartAmplitude)/StartAmplitude/number_of_cycles * 2
        r = np.repeat(np.arange(1, number_of_cycles + 1), count)
        sig = s * np.hstack((r, [0]))
        sig[sig == 0] = .1
        return np.roll(sig, shift=1)

    def reset(self):
        self.current_amplitude = self.VSTART

    def update(self, action):
        gain = self.current_amplitude * self.volt_param.amplitude_gain_factor
        if action == 1:
            if (self.current_amplitude + gain) < self.volt_param.VMAX:
                self.V_list[0][self.index_to_change] = self.current_amplitude + gain
                # V_list[0][index_to_change] = V_list[0][index_to_change] * amplitude_gain_factor
                action_code = 1
            else:
                action_code = 0
                self.V_list[0][self.index_to_change] = self.current_amplitude - gain

        if action == 0:
            if (self.current_amplitude - gain) > self.volt_param.VMIN:
                self.V_list[0][self.index_to_change] = self.current_amplitude - gain
                action_code = 0
            else:
                action_code = 1
                self.V_list[0][self.index_to_change] = self.current_amplitude + gain

        self.current_amplitude = np.max(self.V_list)

        return action_code, self.V_list

    def plot_V(self, ax=None):
        from matplotlib import pyplot as plt
        if ax is None:
            fig = plt.figure('volt_input', figsize=(10, 10))
            ax = fig.add_subplot(111)
        # ax.set_title('Voltage Input', fontsize=25)
        for v in range(len(self.V_list)):
            float_index = [i for i in range(len(self.V_list[v])) if self.V_list[v][i] == 'f']
            value_index = [i for i in range(len(self.V_list[v])) if self.V_list[v][i] != 'f']
            p = ax.plot([self.t_list[i] for i in value_index], [float(self.V_list[v][i]) for i in value_index],
                        label='Node ' + str(self.src[v]), linewidth=2)
            color = p[0].get_color()
            ax.plot([self.t_list[i] for i in float_index], [0] * len(float_index), 'x', color=color, linewidth=2)
        ax.set_xlabel('Time [s]', fontsize=20)
        ax.set_ylabel('Voltage [V]', fontsize=20)
        ax.tick_params(axis='both', labelsize='x-large')
        # ax.set_yticks(fontsize=15)
        ax.legend(fontsize=15)
        ax.grid()
        return ax


def prepare_signal(A1, signal_freq, t_list):
    sign = A1/2 + A1/2*signal.square(2 * np.pi * signal_freq * t_list)
    return sign
