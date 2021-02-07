from dataset import get_data_list, get_infer
from train import *
import argparse as arg


def get_arguments():
    parser = arg.ArgumentParser()
    parser.add_argument('-i', '--iteration', type=int, default=3)
    parser.add_argument('--focal', action='store_false')
    return parser.parse_args()


if __name__=='__main__':
    args = get_arguments()
    train_val = dataset.get_total()
    report = cross_validation(train_val, K=2)
    # infer_ds = dataset.get_infer(dataset.get_data_list(False))
    
