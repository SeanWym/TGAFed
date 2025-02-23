import argparse

class Parser:
    
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.set_arguments()
       
    def set_arguments(self):
        self.parser.add_argument('--gpu', type=str, help='gpu ids to use e.g. 0,1,2,...')
        self.parser.add_argument('--work-type', type=str, help='work-types i.e. gen_data or train')
        self.parser.add_argument('--model', type=str, help='model i.e. fedmatch')
        self.parser.add_argument('--task', type=str, help='task i.e. lc-biid-c10, ls-bimb-c10')
        self.parser.add_argument('--label_unlabel', type=int, help='l=1,2,3  (u=10,9,8)')
        self.parser.add_argument('--frac-clients', type=float, help='fraction of clients per round')
        self.parser.add_argument('--seed', type=int, help='seed for experiment')
        self.parser.add_argument('--semi', type=str, help='semi-supervised learning method i.e. uda,fixmatch') #$$0
        self.parser.add_argument('--dirichlet', type=float, help='Dirichlet parameter (float type)')  
        self.parser.add_argument('--aggregate', type=bool, default=True, help='aggregate == True:uniform_average, aggregate == False:uniform_average_muvar')

    def parse(self):
        args, unparsed  = self.parser.parse_known_args()
        if len(unparsed) != 0:
            raise SystemExit('Unknown argument: {}'.format(unparsed))
        return args
