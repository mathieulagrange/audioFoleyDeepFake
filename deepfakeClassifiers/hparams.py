'''
hparams.py

A file sets hyper parameters for feature extraction and training.
You can change parameters using argument.
For example:
 $ python train_test.py --device=1 --batch_size=32.
'''

import argparse

class HParams(object):
    def __init__(self):
        # Training Parameters
        self.batch_size = 32
        self.num_epochs = 100
        self.learning_rate = 0.1
        self.stopping_rate = 1e-5
        self.momentum = 0.6
        self.factor = 0.2
        self.patience = 30
        self.condition = 'improvement'
        self.input_size = 1
        self.hidden_size = 64
        self.output_size = 64 
        self.pool_size = 3
        self.threshold = 0.5
        self.checkpoint_interval = 10

    # Function for parsing arguments and setting hyperparameters
    def parse_argument(self, args=None, print_argument=True):
        parser = argparse.ArgumentParser()
        for var in vars(self):
            value = getattr(self, var)
            argument = '--' + var
            parser.add_argument(argument, type=type(value), default=value)

        args = parser.parse_args(args)
        for var in vars(self):
            setattr(self, var, getattr(args, var))

        if print_argument:
            print('-------------------------')
            print('Hyper Parameter Settings')
            print('-------------------------')
            for var in vars(self):
                value = getattr(self, var)
                print(var + ': ' + str(value))
            print('-------------------------')

# Usage
hparams = HParams()



