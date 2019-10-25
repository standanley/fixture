from fixture import TemplateMaster

class ContinuousComparatorTemplate(TemplateMaster):
    __name__ = 'comparator_template'
    required_ports = ['in_', 'out']

    @classmethod
    def run_single_test(self, tester):
        # For now we treat the input the same as the other analog inputs
        # so we don't need any pokes of our own
        #tester.expect(getattr(self, 'out_single'), 0, save_for_later=True)
        tester.expect(self.out, 0, save_for_later=True)

    @classmethod
    def process_single_test(self, tester):
        results = []
        results.append(tester.results_raw[tester.result_counter])
        tester.result_counter += 1
        # for an amp, for now, no post-processing is required
        return results

    @classmethod
    def get_tripping_point(self, data):
        ''' Given input/output data, return tripping point of comparator
        sample data: [[.2, .8, .5, 1.1],
                      [ 0,  1,  0,   1]]
        '''
        # for now the trip point is halfway between the last zero and the first one

        def transpose(x):
            return list(zip(*list(x)))
        
        zeros = [x for x,y in transpose(data) if y == 0]
        ones  = [x for x,y in transpose(data) if y == 1]
        if len(zeros) == 0:
            return min(ones)
        if len(ones) == 0:
            return max(zeros)
        return (max(zeros) + min(ones))/2

