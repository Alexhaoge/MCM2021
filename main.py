from torch.utils.data.dataloader import DataLoader
from dataset import get_data_list, get_infer, get_total, get_dataset
from train import *
import argparse as arg
import os


def get_arguments():
    parser = arg.ArgumentParser()
    parser.add_argument('-c', '--cuda', type=int, default=0, help='The number of GPU to use')
    parser.add_argument('-t', '--times', type=int, default=40, help='image augmentation rate')
    parser.add_argument('-b', '--batch', type=int, default=16, help='Batch size')
    parser.add_argument('-p', '--patience', type=int,
                        default=16, help='Patience for early stopping')
    parser.add_argument('-l', '--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('-m', '--model', type=str, default='False', 
                        help='Path to load model. Default "False" means no model')
    parser.add_argument('-k', '--kfolds', type=int, default=5, help='Number of folds')
    parser.add_argument('--infer', action='store_true', help='infer mode')
    parser.add_argument('--no-focal', action='store_false', help='No using focal-loss')
    parser.add_argument('--no-stop', action='store_true', 
                        help='no stop for early stopping when training model for inference')
    parser.add_argument('--new-augment', action='store_true', 
                        help='Whether to generate new augmentation')
    return parser.parse_args()


if __name__=='__main__':
    args = get_arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    if args.model != 'False':
        if not os.path.exists(args.model):
            raise FileNotFoundError('model not found')
    train_val = get_total(times=args.times, new=args.new_augment)
    print('successully load train_val dataset')
    if not args.infer:
        report = cross_validation(
            train_val, K=args.kfolds, 
            learning_rate=args.lr, focal=args.no_focal, 
            patience=args.patience, batch=args.batch
        )
    else:
        print('Entering Inference Mode...')
        data_loader = DataLoader(train_val, batch_size=args.batch, shuffle=True, num_workers=4)
        trainer = Trainer(
            data_loader, data_loader, 
            learning_rate=args.lr, verbose=True, 
            focal=args.no_focal, no_stop=args.no_stop,
            patience=args.patience
        )
        infer_list = get_data_list(True)
        infer_loader = DataLoader(get_infer(infer_list), batch_size=16, shuffle=False, num_workers=4)
        print('successully load infer dataset')
        if args.model != 'False':
            map_location = 'cuda:0' if torch.cuda.is_available() else 'cpu'
            state = torch.load(args.model, map_location)
            trainer.model.load_state_dict(state['model_state_dict'])
            trainer.lossf.load_state_dict(state['loss_state_dict'])
            print('successully load model checkpoints')
        else:
            plot = trainer.fit()
            print('fit complete')
        infer_res = trainer.infer(infer_loader)
        real_list = get_data_list()[0]
        real = get_dataset(real_list)
        real_loader = DataLoader(real, args.batch, shuffle=False, num_workers=4)
        real_infer = trainer.infer(real_loader)
        print('infer complete')
        infer_list['0'] = infer_res[:,0]
        infer_list['1'] = infer_res[:,1]
        real_list['1'] = real_infer[:,1]
        real_list = real_list.append(infer_list, ignore_index=True)
        real_list.to_csv('output/real_infer.csv')
