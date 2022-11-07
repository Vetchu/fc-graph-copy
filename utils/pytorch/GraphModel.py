



#####WORK IN PROGRESS########
###Probably largely false



import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
from torch.optim import optim


class GraphConvolution_subsubmodule(Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution_subsubmodule, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        # H * W
        support = torch.mm(input, self.weight)
        # N(A) * H * W
        output = torch.spmm(adj, support)
        if self.bias is not None:
            # N(A) * H * W + b
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class GCN_submodule(nn.Module):
    '''
    Two layers GCN 
    ...
    Attributes
    ----------
    n_feat : int,
      input features
    n_hid : int
      hidden dim
    n_class : int
      output class
    dropout: float
        froupout rate
    Methods
    -------
    __init__(self, n_feat, n_hid, n_class, dropout)
        
    forward(self, x, adj)
        forward function，x is input feature，adj transformed Adj matrix
    '''

    def __init__(self, n_feat, n_hid, n_class, dropout):
        super(GCN_submodule, self).__init__()
        # first GCN layer，input feature，size:  n_feat，output size: n_hid
        self.gc1 = GraphConvolution_subsubmodule(n_feat, n_hid)
        # seconf GCN layer，input is output from 1st layer，output probility on each class

        self.gc2 = GraphConvolution_subsubmodule(n_hid, n_class)
        # define dropout 
        self.dropout = dropout

    def forward(self, x, adj):
        # frist convo after  relu
        x = F.relu(self.gc1(x, adj))
        # dropout
        x = F.dropout(x, self.dropout, training=self.training)
        # 2nd
        x = self.gc2(x, adj)
        #  log softmax
        return F.log_softmax(x, dim=1)

class Graphmodel(nn.Module):
    def __init__(self, model, config, attributes, device):
        """

        Parameters
        ----------
        model: CNN.Model
        config: dict
            arguments for the model
        attributes: dict
            arguments for DeepModel
        """
        opt = attributes.pop('optimizer')
        opt_func = opt['name']
        opt_param = opt['param'] if 'param' in opt else {}
        loss = attributes.pop('loss')
        loss_func = loss['name']
        loss_param = loss['param'] if 'param' in loss else {}

        # Set attributes for Model:
        # batch_size
        # epochs
        for k, v in attributes.items():
            setattr(self, k, v)
        self.device = device
        self.model = model(**config)
        self.model.to(device=self.device)
        self.loss_func = getattr(nn, loss_func)(**loss_param).to(device=self.device)
        self.optimizer = getattr(optim, opt_func)(self.model.parameters(), **opt_param)

    def evaluate(self, dl):
        """ evaluate the network's performances in terms of loss and accuracy
            load input numpy arrays in a DataLoader and split it into batches

        Parameters
        ----------
        dl :

        Returns
        -------
        loss : float
            running loss value
        acc : float
            running accuracy
        """
        test_loss, test_acc = AverageMeter(), AverageMeter()
        self.model.eval()
        correct, losses, num_samples = 0, 0.0, 0
        with torch.no_grad():
            for data, target in dl:
                data = data.to(device=self.device)
                target = target.to(device=self.device)

                pred = self.model(data)
                correct += (pred.max(1)[1] == target).sum()
                loss = self.loss_func(pred, target)
                prediction = pred.max(1, keepdim=True)[1]
                test_acc.update(prediction.eq(target.view_as(prediction)).sum().item() / data.size(0), data.size(0))
                test_loss.update(loss.item(), data.size(0))
        self.model.train()
        return test_loss.avg, test_acc.avg

    def predict(self, dl):
        """ evaluate the network's performances in terms of loss and accuracy
            load input numpy arrays in a DataLoader and split it into batches

        Parameters
        ----------
        dl :

        Returns
        -------
        loss : float
            running loss value
        acc : float
            running accuracy
        """
        self.model.eval()
        prediction = []
        y_true = []
        with torch.no_grad():
            for data, target in dl:
                data = data.to(device=self.device)
                pred = self.model(data).max(1)[1]
                for item in pred:
                    prediction.append(item.item())
                for item in target:
                    y_true.append(item.item())
        self.model.train()
        return np.array(prediction, dtype='int'), np.array(y_true, dtype='int')

    def train_on_batch(self, data, targets):
        """ train the network on entire data in one pass

        Parameters
        ----------
        data : numpy.ndarray
            image samples
        targets : numpy.array
            labels of the image samples

        """
        data, targets = data.to(self.device), targets.to(self.device)
        pred = self.model(data)
        loss = self.loss_func(pred, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        prediction = pred.max(1, keepdim=True)[1]
        train_acc = prediction.eq(targets.view_as(prediction)).sum().item() / data.size(0)
        train_loss = loss.detach().item()
        return train_loss, train_acc

    def fit(self, train_loader, validation=None, verbose=False):
        """

        Parameters
        ----------
        train_loader
        validation
        verbose

        Returns
        -------

        """
        train_loss, train_acc = AverageMeter(), AverageMeter()
        total_loss_train, total_acc_train, total_loss_test, total_acc_test = [], [], [], []
        for e in range(self.epochs):
            for i, data in enumerate(train_loader):
                loss, acc = self.train_on_batch(data[0], data[1])
                train_acc.update(acc, data[0].size(0))
                train_loss.update(loss, data[0].size(0))
            if validation is not None:
                loss, acc = self.evaluate(validation)
                total_loss_test.append(loss)
                total_acc_test.append(acc)
                if verbose:
                    print(
                        f"Epoch {e + 1}: Train Loss= {train_loss.avg:.2f}, Train Accuracy= {train_acc.avg:.2f},"
                        f" Test Loss= {loss:.2f}, Test Accuracy= {acc:0.2f}")
            elif verbose:
                print(f"Epoch {e + 1}: Loss= {train_loss.avg:.2f}, Accuracy={train_acc.avg:.2f}")
            total_loss_train.append(train_loss.avg)
            total_acc_train.append(train_acc.avg)
        return total_loss_train, total_acc_train, total_loss_test, total_acc_test

    def train_on_batches(self, train_loader, n_batches, verbose=False):
        """

        Parameters
        ----------
        train_loader:  DataLoader
            Custom data loader
        n_batches: int
        verbose: bool

        Returns
        -------

        """
        train_loss, train_acc = AverageMeter(), AverageMeter()
        n_trained_samples = 0
        for i, (data, target) in enumerate(train_loader):
            loss, acc = self.train_on_batch(data, target)
            train_acc.update(acc, data[0].size(0))
            train_loss.update(loss, data[0].size(0))
            n_trained_samples += len(data)
            if i == n_batches - 1:
                break
        if verbose:
            print(f"Loss= {train_loss.avg:.2f}, Accuracy={train_acc.avg:.2f}")
        return train_loss.avg, train_acc.avg, n_trained_samples

    def get_weights(self):
        """ Call get_weights method of CNN or MLP classes

        """
        with torch.no_grad():
            w = []
            for name, param in self.model.named_parameters():
                w.append(param.data.clone().detach().cpu().numpy())
        return w

    def set_weights(self, w):
        """ Call set_weights method of CNN or MLP classes

        Parameters
        ----------
        w : numpy.array
            networks weights with arbitrary dimensions

        """
        with torch.no_grad():
            for i, (name, param) in enumerate(self.model.named_parameters()):
                p = w[i] if isinstance(w[i], np.ndarray) else np.array(w[i], dtype='float32')
                param.data = torch.from_numpy(p).to(device=self.device)

    def set_optimizer_params(self, param):
        self.optimizer.load_state_dict(param)

    def get_optimizer_params(self):
        return self.optimizer.state_dict()


class AverageMeter(object):
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
