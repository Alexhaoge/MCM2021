from torch.utils.data.dataloader import DataLoader
from dataset import get_data_list, get_infer, get_total
from train import *
import argparse as arg


def get_arguments():
    parser = arg.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, default='False', 
        help='Path to load model. Default "False" means no model')
    parser.add_argument('-k', '--kfolds', type=int, default=5, help='Number of folds')
    parser.add_argument('--infer', action='store_true', help='infer mode')
    parser.add_argument('--focal', action='store_false', help='Use focal-loss')
    parser.add_argument('--no-stop', action='store_true', 
        help='no stop for early stopping when training model for inference')
    return parser.parse_args()


if __name__=='__main__':
    args = get_arguments()
    if args.model != 'False':
        import os
        if not os.path.exists(args.model):
            raise FileNotFoundError
    train_val = get_total()
    print('successully load train_val dataset')
    if not args.infer:
        report = cross_validation(train_val, K=args.kfolds, focal=args.focal)
    else:
        print('Entering Inference Mode...')
        data_loader = DataLoader(train_val, batch_size=16, shuffle=True, num_workers=4)
        trainer = Trainer(data_loader, data_loader, verbose=True, focal=args.focal, no_stop=arg.no_stop)
        infer_list = get_data_list(True)
        infer_loader = DataLoader(get_infer(infer_list), batch_size=16, shuffle=False, num_workers=4)
        print('successully load infer dataset')
        if args.model:
            map_location = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            trainer.model._load_from_state_dict(torch.load(args.model, map_location))
            print('successully load model checkpoints')
        plot = trainer.fit()
        print('fit complete')
        infer_res = trainer.infer(infer_loader)
        print('infer complete')
        infer_list['0'] = infer_res[:,0]
        infer_list['1'] = infer_res[:,1]
        infer_list.to_csv('output/infer.csv')
