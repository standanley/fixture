import csv
import os
import pandas
import yaml


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
                'regression_results': None
            }
            self.data[test] = test_data

    def convert_df_columns(self, test, df):
        str_to_signal = {str(s): s for s in test.signals.flat()}
        test = df.rename(str_to_signal, axis='columns', inplace=True)
        print(test)

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
        if True or self.data[test]['input_vectors'] is None:
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

    def save_extracted_data(self, test, data):
        self.data[test]['extracted_data'] = data
        f = self._get_save_file(test, 'extracted_data.csv')
        data.to_csv(f)
        f.close()

    def load_extracted_data(self, test):
        if True or self.data[test]['input_vectors'] is None:
            f = self._get_load_file(test, 'extracted_data.csv')
            data = pandas.read_csv(f)
            self.convert_df_columns(test, data)
            self.data[test]['extracted_data'] = data
            f.close()

        data = self.data[test]['extracted_data']
        return data

    def save_regression_results(self, test, rr):
        # TODO doesn't work with multiple modes
        self.data[test]['regression_results'] = rr
        f = self._get_save_file(test, 'regression_results.yaml')
        yaml.dump(rr, f)
        f.close()
        print(rr)

    def load_regression_results(self, test):
        if self.data[test]['regression_results'] is None:
            f = self._get_save_file(test, 'regression_results.yaml')
            rr = yaml.safe_load(f)
            f.close()
            self.data[test]['regression_results'] = rr
        rr = self.data[test]['regression_results']
        return rr




