from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt

rfadj_col_names = [f'<None -- rfadj<{i}>>' for i in range(6)]
vdd_col_name = '<None -- vdd>'
in_col_names = ['<input[0] -- indiff>', '<input[1] -- incm>']
out_diff_col_name = '<out0_outdiff>'
rf_decimal_col_name = '<None -- rfadj -- (6,)>'
data = read_csv('../checkpoint_folder/DCTest_linear_rfadj/extracted_data.csv', keep_default_na=False)
data['ones'] = 1


def linear_plots():
    optional_cols = rfadj_col_names + [vdd_col_name]
    modeling_cols = optional_cols.copy() + in_col_names
    for optional_col in optional_cols:
        for in_col in in_col_names:
            name = in_col + '_' + optional_col
            modeling_cols.append(name)
            data[name] = data[in_col]*data[optional_col]
    modeling_cols.append('ones')

    simultaneous_sweep_ids = data['swept_group'] == 'None'
    simultaneous_data = data[simultaneous_sweep_ids]
    vdd_data = data[data['swept_group'] == 'vdd']
    data_to_use = simultaneous_data

    M = data_to_use[modeling_cols]
    b = data_to_use[out_diff_col_name]
    x, res, rank, svs = np.linalg.lstsq(M, b, rcond=None)

    b_est = M@x
    #plt.plot(b, b_est, 'x')
    #plt.title('Original with total effect')
    #plt.show()

    result_dict = {col: val for col, val in zip(modeling_cols, x)}


    vdd_dependence_cols = []
    for vdd_col in [vdd_col_name]:
        vdd_dependence_cols.append(vdd_col)
        for in_col in in_col_names:
            name = in_col + '_' + vdd_col
            vdd_dependence_cols.append(name)

    rfadj_dependence_cols = []
    for rfadj_col in rfadj_col_names:
        rfadj_dependence_cols.append(rfadj_col)
        for in_col in in_col_names:
            name = in_col + '_' + rfadj_col
            rfadj_dependence_cols.append(name)


    vdd_effect = sum(data_to_use[col]*result_dict[col] for col in vdd_dependence_cols)
    rfadj_effect = sum(data_to_use[col]*result_dict[col] for col in rfadj_dependence_cols)


    #plt.plot(b, b_est, 'x')
    #limits = [-0.005, 0.005]
    #plt.plot(limits, limits, '--')
    #plt.title('Simulated vs. Modeled Amp Output')
    #plt.xlabel('Simulated Output Voltage')
    #plt.ylabel('Modeled Output Voltage')
    #plt.grid()
    #plt.show()


    d = data_to_use
    def get_contribution(name):
        return result_dict[name] * d[name]
    # out = (A + B*vdd + C*rfadj)  + (D + E*vdd + F*rfadj)
    # A + C*rfadj = ((out - (D + E*vdd + F*rfadj)) / in - B*vdd

    D = result_dict['ones']
    E_times_vdd = result_dict[vdd_col_name] * d[vdd_col_name]
    F_times_rfadj = sum(d[c]*result_dict[c] for c in rfadj_col_names)
    #B_times_in = result_dict[vdd_col_name]
    out = b

    constant_effects = (get_contribution('ones')
                        + get_contribution(vdd_col_name)
                        + sum(get_contribution(rcol) for rcol in rfadj_col_names))
    cm_effects = (get_contribution(in_col_names[1])
                  + get_contribution(in_col_names[1]+'_'+vdd_col_name)
                  + sum(get_contribution(in_col_names[1]+'_'+rcol) for rcol in rfadj_col_names))

    diff_rfadj_effect = sum(get_contribution(in_col_names[0]+'_'+rcol) for rcol in rfadj_col_names)
    diff_vdd_effect = get_contribution(in_col_names[0]+'_'+vdd_col_name)
    #diff_rfadj_effect = sum(result_dict[in_col_names[0]+'_'+rcol] * d[rcol] for rcol in rfadj_col_names)

    # NOTE: there was a bug in this where I forgot to divide diff_vdd_effect by in;
    # I forgot that get_contribution would include that factor, which I don't want here
    gain_no_vdd = (out - constant_effects - cm_effects) / d[in_col_names[0]] - diff_vdd_effect/d[in_col_names[0]]
    gain_no_rfadj = (out - constant_effects - cm_effects) / d[in_col_names[0]] - diff_rfadj_effect/d[in_col_names[0]]

    gain_lhs_rfadj = (get_contribution(in_col_names[0]) + diff_rfadj_effect) / d[in_col_names[0]]
    gain_lhs_vdd = (get_contribution(in_col_names[0]) + diff_vdd_effect) / d[in_col_names[0]]


    print(d[rf_decimal_col_name])
    print(gain_no_vdd)

    plt.plot(d[rf_decimal_col_name], gain_no_vdd, '*')
    plt.plot(d[rf_decimal_col_name], gain_lhs_rfadj, 'x')
    plt.legend(['Measured, with vdd effect removed', 'Modeled, not including vdd effect'], loc='lower left')
    plt.title('Estimated gain vs. radj with vdd effect removed')
    plt.xlabel('radj decimal value')
    plt.ylabel('estimated gain')
    plt.ylim((0, 3500))
    plt.grid()
    plt.show()



    plt.plot(d[vdd_col_name], gain_no_rfadj, '*')
    plt.plot(d[vdd_col_name], gain_lhs_vdd, 'x')
    plt.legend(['Measured, with radj effect removed', 'Modeled, not including radj effect'], loc='lower left')
    plt.title('Estimated gain vs. vdd with radj effect removed')
    plt.xlabel('vdd')
    plt.ylabel('estimated gain')
    plt.ylim((0, 3500))
    plt.grid()
    plt.show()

    data_vdd = data[data['swept_group']=='vdd']


def nonlinear_plots():
    np.random.seed(4)
    def circuit_model(in_, radj, vdd):
        gain = 1300 + 300*(vdd-3.0) + 500*(vdd-3.0)**2
        amplitude = 2.2 + 0.2*(vdd-3.0)


linear_plots()
#nonlinear_plots()

