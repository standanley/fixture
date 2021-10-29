#from dragonphy import *
import math

from dragonphy import Channel
import numpy as np
from scipy import linalg
import matplotlib.pyplot as plt

#THIS_DIR = Path(__file__).resolve().parent
sparam_file_list = ["Case4_FM_13SI_20_T_D13_L6.s4p",
                    "peters_01_0605_B1_thru.s4p",  
                    "peters_01_0605_B12_thru.s4p",  
                    "peters_01_0605_T20_thru.s4p",
                    "TEC_Whisper42p8in_Meg6_THRU_C8C9.s4p",
                    "TEC_Whisper42p8in_Nelco6_THRU_C8C9.s4p"]
file = sparam_file_list[0] # looks almost clean at 2e9

def temp_get_data():
    # email data was 2e9, 10e9, 2e9 bit freq
    # sample rate was /5 for the last
    sample_rate_desired = 1e12/5
    bit_freq = 2e9
    total_time = 100e-9*5*2.5#10e-9

    samples_per_bit = math.ceil(sample_rate_desired / bit_freq)
    sample_rate = bit_freq * samples_per_bit

    num_samples = math.ceil(total_time * sample_rate)
    num_bits = int(total_time*bit_freq)+1
    bits = np.random.randint(2, size=(int(num_bits),))

    file_name = f'../tests/channels/{file}'
    #t, imp = s4p_to_impulse(str(file_name), 0.1e-12, 20e-9, zs=50, zl=50)

    #Load the S4P into a channel model
    chan = Channel(channel_type='s4p', sampl_rate=sample_rate, resp_depth=200000, s4p=file_name, zs=50, zl=50)


    # Gives you the pulse response for a pulse corresponding to f_sig, BUT also
    # gives back the answer sampled at f_sig. So it always gives a resolution of
    # 1 sample per bit which is great for testing an FFE, but not for spice.
    # So we ask for the pulse response of a shorter pulse and convolve it ourselves
    # first entry in time is -t_delay. Peak of bit is ~5.5e-9 for this channel
    time, pulse = chan.get_pulse_resp(f_sig=sample_rate, resp_depth=10000, t_delay=0)#-5.4e-9)#shift_delay + mm_lock_pos)
    cursor_index = np.argmax(pulse)
    #cursor_index = int(sum(np.arange(len(pulse))*pulse*pulse) / sum(pulse*pulse))
    cursor_index = np.argmax(np.convolve(pulse, [1]*samples_per_bit))

    print(time[1] - time[0], (time [-1]-time[0]) / len(time))
    print('len time', len(time))

    plt.plot(time, pulse, '*')
    plt.title('Pulse response due to pulse at SAMPLE RATE, not bit rate')
    plt.grid()
    plt.show()


    # I think this does not rely on floating point comparison anywhere
    t = np.arange(num_samples) / sample_rate
    channel_input = np.vstack([bits]*samples_per_bit).T.reshape(num_bits*samples_per_bit)
    channel_input = channel_input[:num_samples]

    #chan_mat = linalg.convolution_matrix(pulse, len(channel_input))
    #imp[target_curs_pos] = 1
    #zf_taps = np.reshape(np.linalg.pinv(chan_mat) @ imp, (N,))

    #test = chan_mat @ channel_input
    # NOTE we probably don't want 'same' here since the proper time to start
    # and end kinda depends on the location of the cursor, but it's fine for now
    test = np.convolve(channel_input, pulse)
    test = test[cursor_index : cursor_index + num_samples]
    #test += 0.5
    #test = test[1500:]
    print(bits)
    plt.plot(t, test, '*')
    plt.plot(t, channel_input)
    #plt.plot(np.diff(test))
    #plt.plot(time2, pulse2, 'x')
    plt.grid()
    plt.show()

    # email data was 4e-9, 4e-9, 0.8e-9*5
    dt = 0.8e-9*5
    t = np.arange(len(test))*dt
    data = np.vstack((t, test)).T

    print(test.shape)
    print(t.shape)
    print(data.shape)
    np.savetxt('test_channel2.csv', data, delimiter=',')

    return test


    if debug_DLE:
        plt.stem(zf_taps)
        plt.show()
        plt.plot(pulse)
        plt.show()
        plt.plot(np.convolve(pulse, zf_taps))
        plt.show()
        exit()
    '''
    bits = np.random.randint(2, size=(int(10**num_bits),))*2 - 1
    codes = chan.compute_output(bits, resp_depth=200, f_sig=f_sig, t_delay=shift_delay + mm_lock_pos)
    codes_plus_noise = codes + np.random.randn(len(codes))*1.8e-2
    eq_codes = np.convolve(codes_plus_noise, zf_taps)

    eq_codes = eq_codes[target_curs_pos:int(10**num_bits) + target_curs_pos]
    est_bits = np.where(eq_codes>0, 1, -1)

    est_codes = chan.compute_output(est_bits, resp_depth=200, f_sig=f_sig, t_delay=shift_delay + mm_lock_pos)

    error_sig = codes-est_codes
    b_err = (bits - est_bits)/2
    err_loc = np.where(np.abs(b_err) == 1)[0]
    total_errors = len(err_loc)

    distance_between_errors = err_loc[1:] - err_loc[:-1]
    first_skip = True

    if total_errors < 1:
        print('No Errors Found')
        continue


    dict_of_errors = {}
    dict_of_error_pol = {}
    curr_error     = "p"
    curr_err_loc   = err_loc[0]
    curr_error_polarity = b_err[err_loc[0]]

    for ii, dist_btwn_err in enumerate(distance_between_errors):
        print(dist_btwn_err, err_loc[ii], b_err[err_loc[ii]], b_err[err_loc[ii]+dist_btwn_err])

        if dist_btwn_err > 15:
            print("Storing:", curr_error, curr_err_loc)
            if curr_error not in dict_of_errors:
                dict_of_errors[curr_error] = []
                dict_of_error_pol[curr_error] = []
            dict_of_errors[curr_error] += [curr_err_loc]
            dict_of_error_pol[curr_error] += [curr_error_polarity]

            curr_err_loc = err_loc[ii] + dist_btwn_err
            curr_error = "p"
            curr_error_polarity = b_err[err_loc[ii]+dist_btwn_err]
            print()
        else:
            new_tag = str(dist_btwn_err-1) if dist_btwn_err > 1 else ""
            if b_err[err_loc[ii]+dist_btwn_err] == curr_error_polarity:
                new_tag += 'p'
            else:
                new_tag += 'n'
            curr_error += new_tag

    if distance_between_errors[-1] > 10:
        dict_of_errors['p'] += [err_loc[-1]]

    running_total = 0
    print(f'Error Rate: {total_errors/int(10**num_bits)}')


    for err_typ in dict_of_errors:
        average_bit_patterns = np.zeros((30,))
        for idx, pol in zip(dict_of_errors[err_typ],dict_of_error_pol[err_typ]):
            plt.plot(error_sig[idx-5+num_of_precursors:idx+num_of_precursors+30])
            plt.plot(b_err[idx-5:idx+30])
            average_bit_patterns = average_bit_patterns + bits[idx-25:idx+5]*pol
        plt.title(err_typ)
        plt.show()

        plt.plot(average_bit_patterns/len(dict_of_errors[err_typ]))
        plt.show()

        for idx in dict_of_errors[err_typ]:
            plt.plot(eq_codes[idx-5+num_of_precursors:idx+num_of_precursors+30])
        plt.title(err_typ)
        plt.show()

        num_of_errors = len(list(filter(lambda x: (x== 'p') or (x == 'n'), err_typ)))
        running_total += len(dict_of_errors[err_typ])*num_of_errors
        print(f'% of {err_typ}: {len(dict_of_errors[err_typ])*num_of_errors/total_errors * 100} %')

    print(running_total, total_errors)





    '''


temp_get_data()
