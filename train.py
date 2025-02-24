from typing import Tuple
from model import Inception, FocalLoss
from torch.optim import RMSprop
from torch.optim.lr_scheduler import ExponentialLR
from torch import nn
import torch
from utils import EarlyStopping, plot
from torch.utils import data
from datetime import datetime

class Trainer:
    def __init__(self, 
        train_loader: data.DataLoader,
        val_loader: data.DataLoader,
        epoch: int = 200,
        learning_rate: float = 0.001,
        verbose: bool = False,
        patience: int = 16,
        no_stop: bool = False,
        focal: bool = True,
    ) -> None:
        self.verbose = verbose
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model = Inception(2)
        self.opt = RMSprop(self.model.parameters(), lr=learning_rate)
        self.scheduler = ExponentialLR(optimizer=self.opt, gamma=0.94)
        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
            self.model = self.model.cuda(self.device)
        self.early = EarlyStopping(self.verbose, patience, no_stop)
        self.epochs = epoch
        self.lossf = FocalLoss() if focal else nn.CrossEntropyLoss()

    def fit(self) -> str:
        train_losses, val_losses = [], []
        for epoch in range(1, self.epochs+1):
            train_loss = self.__train()
            val_loss, tp, tn = self.__eval()[0:3]
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            if self.verbose:
                print(f'Epoch: {epoch}/{self.epochs} \
                    - loss: {train_loss:.4f} \
                    - val_loss: {val_loss:.4f} \
                    - true-positive: {tp} \
                    - true-negative: {tn}')
            filename = 'output/checkpoints/'+datetime.now().strftime('%Y-%m-%d-%H_%M_%S')+str(epoch)+'.tar.gz'
            self.early(val_loss, self.model, self.lossf, self.opt, epoch, filename)
            if self.early.isToStop:
                if self.verbose:
                    print("=> Stopped")
                break
        return plot([train_losses, val_losses], ['Training', 'Validation'], 'Epochs', 'Error', 'Error analysis')

    def __train(self) -> float:
        self.model.train()
        epoch_loss = .0
        for id, (inputs, target) in enumerate(self.train_loader):
            inputs, target = inputs.to(self.device), target.to(self.device)
            output = self.model(inputs)
            loss = self.lossf(output, target)
            self.opt.zero_grad()
            loss.backward()
            self.opt.step()
            if id % 2:
                self.scheduler.step()
            epoch_loss += loss.item()
        return epoch_loss/len(self.train_loader)
    
    def __eval(self) -> Tuple[float, int, int, int, int]:
        self.model.eval()
        loss_sum = .0
        tp, tn, fp, fn = 0, 0, 0, 0
        with torch.no_grad():
            for id, (inputs, target) in enumerate(self.val_loader):
                inputs, target = inputs.to(self.device), target.to(self.device)
                output = self.model(inputs)
                loss_sum += self.lossf(output, target).item()
                tp += ((output[:,0]<output[:,1])&(target==1)).sum().item()
                tn += ((output[:,0]>output[:,1])&(target==1)).sum().item()
                fp += ((output[:,0]<output[:,1])&(target==0)).sum().item()
                fn += ((output[:,0]>output[:,1])&(target==0)).sum().item()
        return loss_sum/len(self.val_loader), tp, tn, fp, fn

    def eval(self) -> dict:
        res = self.__eval()
        return {
            'loss_f': self.lossf.__str__(),
            'loss': res[0],
            'tp': res[1],
            'tn': res[2],
            'fp': res[3],
            'fn': res[4]
        }

    def infer(self, input_loader: data.DataLoader):
        self.model.eval()
        outputs = None
        with torch.no_grad():
            for _, (inputs, _)in enumerate(input_loader):
                output = self.model(inputs.to(self.device))
                if outputs is None:
                    outputs = output
                else:
                    outputs = torch.cat([outputs, output])
        return outputs.cpu().numpy()


def cross_validation(
        ds: data.TensorDataset, 
        K: int = 3, batch: int = 16,
        learning_rate: float = 0.001,
        focal: bool = True,
        patience: int = 16
    ):
    print('Total {} images, {} folds, batch size {}, use focal loss: {}'.format(len(ds), K, batch, focal))
    assert isinstance(ds, data.TensorDataset)
    size = len(ds) // K
    size_list = [size] * K
    size_list[0] += len(ds) % K
    folds = data.random_split(ds, size_list, torch.Generator().manual_seed(59))
    import pandas as pd
    result = pd.DataFrame(columns=['loss_f', 'loss', 'tp', 'tn', 'fp', 'fn', 'plot'])
    for i in range(K):
        print('=======> Fold '+str(i)+' <========')
        train = None
        for j in range(K):
            if j != i:
                train = folds[j] if train is None else train+folds[j]
        train_loader = data.DataLoader(train, batch_size=batch, num_workers=4)
        val_loader = data.DataLoader(folds[i], batch_size=batch, num_workers=4)
        trainer = Trainer(train_loader, val_loader, learning_rate=learning_rate, focal=focal, verbose=True, patience=patience)
        print('dataloader and trainer created, start fitting')
        plot_path = trainer.fit()
        print('start evaluate')
        eval_res = trainer.eval()
        eval_res['plot'] = plot_path
        print('Eval Result\n{}'.format(eval_res))
        result = result.append(eval_res, ignore_index=True)
    filename = 'output/'+datetime.now().strftime('%d-%H_%M_%S') + '-report.csv'
    result.to_csv(filename)
    return result

