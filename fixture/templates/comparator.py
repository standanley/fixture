from fixture import TemplateMaster

class ContinuousComparatorTemplate(TemplateMaster):
    __name__ = 'comparator_template'
    required_ports = ['in_', 'out']

    @classmethod
    def specify_test_inputs(self):
        return [self.in_]

    @classmethod
    def run_single_test(self, tester, values):
        # For now we treat the input the same as the other analog inputs
        # so we don't need any pokes of our own
        tester.poke(self.in_, values['in_'])
        return tester.read(self.out)

    @classmethod
    def process_single_test(self, read):
        # TODO the slicing should take place in fault, I'm not sure why it's not
        results = {'out': 1 if read.value > 0.6 else 0}
        return results

    @classmethod
    def get_tripping_point(self, data):
        ''' Given input/output data, return tripping point of comparator
        sample data: [[.2, .8, .5, 1.1],
                      [ 0,  1,  0,   1]]
        '''
        # for now the trip point is halfway between the last zero and the first one

        print(data)

        def transpose(x):
            return list(zip(*list(x)))

        data = (data[self.in_], data['out'])
        
        zeros = [x for x,y in transpose(data) if y == 0]
        ones  = [x for x,y in transpose(data) if y == 1]
        if len(zeros) == 0:
            return min(ones)
        if len(ones) == 0:
            return max(zeros)
        return (max(zeros) + min(ones))/2

