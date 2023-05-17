import csv
import os
import pandas
import yaml
import numpy as np
from yaml.representer import Representer
from collections import defaultdict

from fixture.sampler import SampleManager

yaml.add_representer(defaultdict, Representer.represent_dict)


class Checkpoint:
    def __init__(self, template, filepath):
        self.template = template
        self.filepath = filepath
        self.data = {}
        for test in self.template.tests:
            test_data = {
                'input_vectors': None,
                'sim_result_folder': None,
                'extracted_data': None,
                'extracted_data_unprocessed': None,
                'regression_results': None
            }
            self.data[test] = test_data

    @classmethod
    def save_csv(cls, dataframe, f):
        data_converted_none = dataframe.replace({None: 'None'})
        data_converted_none.to_csv(f, na_rep='nan')

    @classmethod
    def load_csv(cls, f):
        data = pandas.read_csv(f)
        data = data.replace({'None': None})
        return data

    def convert_df_columns(self, test, df):
        # signals in column names
        str_to_signal = {str(s): s for s in
                         test.signals.flat()
                         + list(test.signals)
                         #+ list(test.input_signals)
                         }
        df.rename(str_to_signal, axis='columns', inplace=True)

        # sample groups in group_sweep column
        tag = SampleManager.GROUP_ID_TAG
        if tag in df.columns:
            str_to_sample_group = {str(sg): sg for sg in
                                   test.sample_groups_test + test.sample_groups_opt}
            def restore_sg(sg_str):
                return str_to_sample_group.get(sg_str, sg_str)
            #assert (not any(sg_str in str_to_signal
            #               for sg_str in str_to_sample_group)), 'Name collision'
            df[tag] = df[tag].map(restore_sg)

    def _get_save_file(self, test, filename):
        # DON'T FORGET TO CLOSE IT
        folder = os.path.join(self.filepath, str(test))
        os.makedirs(folder, exist_ok=True)
        filename = os.path.join(folder, filename)
        if os.path.exists(filename):
            print(f'Overwriting file {filename}')

        f = open(filename, 'w')
        return f

    def _get_load_file(self, test, filename):
        # DON'T FORGET TO CLOSE IT
        folder = os.path.join(self.filepath, str(test))
        filename = os.path.join(folder, filename)
        f = open(filename, 'r')
        return f


    def save_input_vectors(self, test, input_vectors):
        # TODO use a dataframe so we can use self.save_csv
        self.data[test]['input_vectors'] = input_vectors
        f = self._get_save_file(test, 'input_vectors.csv')
        writer = csv.writer(f)
        keys = list(input_vectors.keys())
        writer.writerow([str(key) for key in keys])
        column_major = [input_vectors[key] for key in keys]
        row_major = zip(*column_major)
        writer.writerows(row_major)
        f.close()

    def load_input_vectors(self, test):
        if self.data[test]['input_vectors'] is None:
            f = self._get_load_file(test, 'input_vectors.csv')
            try:
                input_vectors = pandas.read_csv(f)
            except pandas.errors.EmptyDataError:
                input_vectors = pandas.DataFrame({})
            self.convert_df_columns(test, input_vectors)
            f.close()
            self.data[test]['input_vectors'] = input_vectors

        iv = self.data[test]['input_vectors']
        return iv


    def suggest_run_dir(self, test):
        return os.path.join(self.filepath, str(test))

    def save_run_dir(self, test, run_dir):
        self.data[test]['sim_result_folder'] = run_dir

    def save_extracted_data_unprocessed(self, test, data):
        self.data[test]['extracted_data_unprocessed'] = data
        f = self._get_save_file(test, 'extracted_data_unprocessed.csv')
        self.save_csv(data, f)
        f.close()

    def load_extracted_data_unprocessed(self, test):
        if True or self.data[test]['extracted_data_unprocessed'] is None:
            f = self._get_load_file(test, 'extracted_data_unprocessed.csv')
            data = self.load_csv(f)
            self.convert_df_columns(test, data)
            self.data[test]['extracted_data_unprocessed'] = data
            f.close()

        data = self.data[test]['extracted_data_unprocessed']
        return data

    def save_extracted_data(self, test, data):
        self.data[test]['extracted_data'] = data
        f = self._get_save_file(test, 'extracted_data.csv')
        self.save_csv(data, f)
        f.close()

    def load_extracted_data(self, test):
        if True or self.data[test]['extracted_data'] is None:
            f = self._get_load_file(test, 'extracted_data.csv')
            data = self.load_csv(f)
            self.convert_df_columns(test, data)
            self.data[test]['extracted_data'] = data
            f.close()

        data = self.data[test]['extracted_data']
        return data

    def save_regression_results(self, test, rr):
        # TODO doesn't work with multiple modes
        self.data[test]['regression_results'] = rr
        # remove references from rr because yaml doesn't handle them well
        rr_dict = {}
        for lhs, rhs in rr.items():
            coef_names = [f'lhs[{i}]' for i in range(rhs.NUM_COEFFICIENTS)]
            verilog = rhs.verilog(lhs, coef_names)
            # TODO I think lhs might always be an AnalysisResultSignal
            lhs_str = lhs if isinstance(lhs, str) else lhs.friendly_name()
            rr_dict[lhs_str] = {
                'verilog': verilog,
                'coefs': [float(x) for x in rhs.x_opt]
            }

            #rhs_clean = {}
            #for param, expression in rhs.items():
            #    expression_clean = {}
            #    for thing, coef in expression.items():
            #        if not isinstance(thing, str):
            #            thing_clean = str(thing)
            #            expression_clean[thing_clean] = coef
            #        else:
            #            expression_clean[thing] = coef
            #    rhs_clean[param] = expression_clean
            #rr_clean[lhs] = rhs_clean
        print('aboutto save rr')
        f = self._get_save_file(test, 'regression_results.yaml')
        yaml.dump(rr_dict, f)
        f.close()
        print('finished saving rr')

    def load_regression_results(self, test):
        # loads regression results into rr and returns it
        # ALSO edits test.parameter_algebra_final expressions to set x_opt



        if True or self.data[test]['regression_results'] is None:
            f = self._get_load_file(test, 'regression_results.yaml')
            print('about to load rr')
            rr = yaml.safe_load(f)
            print('finished loading rr')
            f.close()
            rr_clean = {}

            for lhs, rhs in test.parameter_algebra_final.items():
                lhs_str = lhs if isinstance(lhs, str) else lhs.friendly_name()
                assert lhs_str in rr, f'Saved regression results missing info for {lhs}'

                # check that saved verilog matches current verilog
                # this makes sure the versions match
                coef_names = [f'lhs[{i}]' for i in range(rhs.NUM_COEFFICIENTS)]
                verilog = rhs.verilog(lhs_str, coef_names)
                for v_old, v_new in zip(rr[lhs_str]['verilog'], verilog):
                    assert v_old == v_new, 'Mismatch in verilog in saved regression results; probably saved results for a different circuit version'

                # actually assign x_opt
                coefs = rr[lhs_str]['coefs']
                assert len(coefs) == rhs.NUM_COEFFICIENTS
                rhs.x_opt = np.array(coefs)

                # grab coefs for rr_clean
                rr_clean[lhs] = rhs #rr[lhs]['coefs']

            ## edit rr in place to replace things with Signal objects
            #for lhs, rhs in rr.items():
            #    for param, expression in rhs.items():
            #        for thing in list(expression.keys()):
            #            try:
            #                thing_obj = test.signals.from_str(thing)
            #                expression[thing_obj] = expression[thing]
            #                del expression[thing]
            #            except KeyError:
            #                pass

            self.data[test]['regression_results'] = rr_clean
        return self.data[test]['regression_results']




