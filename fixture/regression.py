# TODO these import are unused but they fix a weird import error issue
from fixture.signals import SignalArray, SignalIn, CenteredSignalIn
from fixture.optional_fit import get_optional_expression_from_influences, HierarchicalExpression


class Regression:
    # statsmodels gets confused if you try to use '1' to mean a column of 
    # ones, so we manually add a column of ones with the following name
    one_literal = 'const_1'

    @staticmethod
    def clean_string(s):
        # discrepency between the way spice and magma do braces
        # also Patsy won't use < in a variable name
        temp = s.replace('<', '_').replace('>', '_')
        final = temp.replace('[', '_').replace(']', '_')
        return final

    @staticmethod
    def vector_parameter_name_output(name, vec_i, signal):
        #return f'{name}_outvec[{vec_i}]'
        return f'{name}_{signal.friendly_name()}'

    @staticmethod
    def vector_parameter_name_input(name, vec_i, signal):
        #return f'{name}_invec[{vec_i}]'
        return f'{name}_{signal.friendly_name()}'

    @staticmethod
    def vector_input_name(name, vec_i):
        return f'{name}_invec[{vec_i}]'


    def __init__(self, template, test, data, mode_prefix):
        '''
        Incoming data should be of the form 
        {pin:[x1, ...], ...}
        '''

        assert self.one_literal in data, 'Should have been set in go()'

        self.consts = {}

        '''
        The format for results is:
        results[lhs][param][opt] = coef
        example:
        {'amp_output':
            {'dcgain':
                 {'const_1': 0.2658061976765506,
                  <vdd>: -0.011425713964578879},
             }
        }
        '''
        results_expr = {}

        for lhs, rhs in test.parameter_algebra_final.items():

            #df_row_mask = ~ regression_data[lhs_clean].isnull()
            ## TODO when I blank out entries in the data spreadsheet they appear as nan, but those rows aren't filtered. Is that bad?
            #df_filtered = regression_data[df_row_mask]

            # TODO how to handle rows with nan? I copied this from old code
            df_row_mask = ~ data[lhs].isnull()
            # TODO when I blank out entries in the data spreadsheet they appear as nan, but those rows aren't filtered. Is that bad?
            df_filtered = data[df_row_mask]


            # TODO just stole some code from here for config_parse
            total_expr = rhs

            verilog_const_names = [f'c[{i}]' for i in range(total_expr.NUM_COEFFICIENTS)]
            verilog = total_expr.verilog(lhs.friendly_name(), verilog_const_names)

            lhs_data = df_filtered[lhs]

            # do the regression!
            print(f'Starting parameter fit for {total_expr}')
            # we don't need to look at coefs_fit because they're already
            # assigned to total_expr.x_opt
            coefs_fit = total_expr.fit_by_group(df_filtered, lhs_data, mode_prefix)
            print(f'Finished parameter fit for {total_expr}')
            for line in verilog:
                print(line)
            for n, v in zip(verilog_const_names, total_expr.x_opt):
                print(f'{n} = {v};')
            print()
            results_expr[lhs] = total_expr.copy()

        # We don't edit data, so there's no reason to re-save it here
        #self.expr_dataframe = data
        self.results_expr = results_expr


    def make_formula(self, lhs_clean, rhs_names):
        return f'I({lhs_clean}) ~ {" + ".join(rhs_names)} -1'

