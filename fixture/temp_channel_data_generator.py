#from dragonphy import *
import math

from dragonphy import Channel
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

##THIS_DIR = Path(__file__).resolve().parent
#sparam_file_list = ["Case4_FM_13SI_20_T_D13_L6.s4p",
#                    "peters_01_0605_B1_thru.s4p",
#                    "peters_01_0605_B12_thru.s4p",
#                    "peters_01_0605_T20_thru.s4p",
#                    "TEC_Whisper42p8in_Meg6_THRU_C8C9.s4p",
#                    "TEC_Whisper42p8in_Nelco6_THRU_C8C9.s4p"]
#file = sparam_file_list[0] # looks almost clean at 2e9


class ChannelUtil:
    @staticmethod
    def get_channel_data(file_path, bit_freq, sample_freq_desired, total_time, debug=False):
        samples_per_bit = math.ceil(sample_freq_desired / bit_freq)
        sample_freq = bit_freq * samples_per_bit

        num_samples = math.ceil(total_time * sample_freq)
        num_bits = int(total_time*bit_freq)+1
        bits = np.random.randint(2, size=(int(num_bits),))

        if debug:
            print('Channel data request:')
            print('bit_freq', bit_freq)
            print('sample_freq', sample_freq)
            print('num_bits', num_bits)


        # Load the S4P into a channel model
        chan = Channel(channel_type='s4p', sampl_rate=sample_freq, resp_depth=200000, s4p=file_path, zs=50, zl=50)

        # Gives you the pulse response for a pulse corresponding to f_sig, BUT also
        # gives back the answer sampled at f_sig. So it always gives a resolution of
        # 1 sample per bit which is great for testing an FFE, but not for spice.
        # So we ask for the pulse response of a shorter pulse and convolve it ourselves
        # first entry in time is -t_delay. Peak of bit is ~5.5e-9 for this channel
        time, pulse = chan.get_pulse_resp(f_sig=sample_freq, resp_depth=10000, t_delay=0)#-5.4e-9)#shift_delay + mm_lock_pos)
        #cursor_index = np.argmax(pulse)
        #cursor_index = int(sum(np.arange(len(pulse))*pulse*pulse) / sum(pulse*pulse))
        cursor_index = np.argmax(np.convolve(pulse, [1]*samples_per_bit))

        print(time[1] - time[0], (time [-1]-time[0]) / len(time))
        print('len time', len(time))

        if debug:
            plt.plot(time, pulse, '*')
            plt.title('Pulse response due to pulse at SAMPLE RATE, not bit rate')
            plt.grid()
            plt.show()

        # I think this does not rely on floating point comparison anywhere
        t = np.arange(num_samples) / sample_freq
        channel_input = np.vstack([bits]*samples_per_bit).T.reshape(num_bits*samples_per_bit)
        channel_input = channel_input[:num_samples]

        # NOTE we probably don't want 'same' here since the proper time to start
        # and end kinda depends on the location of the cursor, but it's fine for now
        channel_output = np.convolve(channel_input, pulse)
        channel_output = channel_output[cursor_index : cursor_index + num_samples]

        if debug:
            plt.plot(t, channel_output, '*')
            plt.plot(t, channel_input)
            #plt.plot(np.diff(channel_output))
            #plt.plot(time2, pulse2, 'x')
            plt.grid()
            plt.show()

        #dt = 0.8e-9*5
        #t = np.arange(len(channel_output))*dt
        data = np.vstack((t, channel_output)).T

        #print(channel_output.shape)
        #print(t.shape)
        #print(data.shape)
        #np.savetxt('test_channel2.csv', data, delimiter=',')

        return data
