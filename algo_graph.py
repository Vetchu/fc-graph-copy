import os
import time
from matplotlib.rcsetup import all_backends
import networkx as nx
from scipy.sparse import csgraph
import numpy as np
import pandas as pd
from collections import defaultdict
import operator
import pandas as pd
import math
from sklearn.cluster import SpectralClustering
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.optim as optim
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class Patients:
    def __init__(self):
        self.raw_data = RawData()
        self.patientsdf = self.load_patients()
        
    def load_patients(self):
        self.raw_data.read_data_from_csv()
        # extract all existing phenotypes
        self.all_phenotypes = self.get_all_phenotypes()
        # create patients dataframe with id and phenotypes as columns
        patientsdf = self.extract_patients()
        return patientsdf

    def get_all_phenotypes(self):
        # extract all existing phenotypes 
        all_phenotypes = []
        for index, row in self.raw_data.mimic_labevents_df.iterrows():
            patient_list = row.to_string().split(";", 1)
            id_patient = patient_list[0].split()[1]
            hp = patient_list[0].split()[3] #works differently in ipynb and py
            if hp not in all_phenotypes:
                all_phenotypes.append(hp)
        return all_phenotypes

    def extract_patients(self):
        df_columns = ['id'] + self.all_phenotypes 
        patientsdf = pd.DataFrame(columns=df_columns)

        first_row = self.raw_data.mimic_labevents_df.iloc[0]
        first_id =  self.raw_data.mimic_labevents_df.iloc[0]['id']
        patient_dict = dict()
        
        for _, row in self.raw_data.mimic_labevents_df.iterrows():   
            id_patient = row['id']
            hp = row['hpo']
            
            if id_patient == first_id:
                patient_dict[hp] = 1
            else:
                # append previous patient to dataframe
                patient_dict['id'] = first_id
                for phenotype in self.all_phenotypes:
                    if phenotype not in patient_dict.keys():
                        patient_dict[phenotype] = 0
                        
                patientsdf = patientsdf.append(patient_dict, ignore_index = True)
                
                # prepare the next patient
                first_id = id_patient
                phenotypes_dict = dict()
                phenotypes_dict[hp] = 1 

        # append last patient
        patient_dict['id'] = first_id
        for phenotype in self.all_phenotypes:
            if phenotype not in patient_dict.keys():
                patient_dict[phenotype] = 0
        patientsdf = patientsdf.append(patient_dict, ignore_index = True)
        return patientsdf

class RawData:
    def __init__(self):
        self.dirpath = os.path.dirname(os.path.abspath(__file__))
        self.diagnose_icd_hpo_file_name = "DIAGNOSE_ICD_hpo.csv"
        self.mimic_diagnose_icd9_file_name = "MIMIC_DIAGNOSE_ICD9.csv"
        self.mimic_export_1_subjects_file_name = "MIMIC_Export_1_Subjects.csv"
        self.mimic_export_2_labevents_hpo_file_name = "MIMIC_Export_2_Labevents_HPO.csv"        
 
    def read_data_from_csv(self):    
        self.diagnose_df = pd.read_csv(os.path.join(self.dirpath, self.diagnose_icd_hpo_file_name))
        self.mimic_diagnose_df = pd.read_csv(os.path.join(self.dirpath, self.mimic_diagnose_icd9_file_name)) # output data (icd9 codes)
        self.mimic_subjects_df = pd.read_csv(os.path.join(self.dirpath, self.mimic_export_1_subjects_file_name)) # patients ids
        self.mimic_labevents_df = pd.read_csv(os.path.join(self.dirpath, self.mimic_export_2_labevents_hpo_file_name), sep=";", names=['id', 'hpo'], header=None) # input features

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
        #seconf GCN layer，input is output from 1st layer，output probility on each class
        
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

class Model():
    def __init__(self):
        self.patients = Patients()
        self.features = self.patients.patientsdf.iloc[:, self.patients.patientsdf.columns != 'id'].values  
        self.features = torch.tensor(self.features.astype('float'), dtype=torch.float32)
        self.G = self.create_graph(self.patients.patientsdf)
        self.adj = self.create_adj()
        self.labels = self.create_labels()
        self.model = GCN_submodule(n_feat=self.features.shape[1],
            n_hid=16,
            n_class=self.labels.max().item() + 1, #changed
            dropout=0.5)
        self.optimizer = optim.Adam(self.model.parameters(),
                       lr=0.01, 
                       weight_decay=5e-4)
        self.idx_train = range(20)
        self.idx_val = range(20, 40)
        self.epochs = 200

    def create_adj(self):
        adj_mat = np.asarray(nx.to_numpy_matrix(self.G))
        adj = adj_mat + adj_mat.T * (adj_mat.T > adj_mat) - adj_mat @(adj_mat.T > adj_mat)
        adj=torch.tensor(adj,dtype=torch.float32)
        return adj

    def create_graph(self, patients):
        G = nx.Graph()
        # create patients nodes
        all_ids = patients['id']
        for id in all_ids:
            G.add_node(id)
        # add edges between nodes depending on the number of phenotypes shared
        for index_1, row_1 in patients.iterrows():
            for index_2, row_2 in patients.iterrows():
                if index_1 < index_2:
                    shared_hp = 0
                    for hp in range(1, len(row_1)):
                        if row_1[hp] == row_2[hp]:
                            shared_hp += 1
                    G.add_edge(row_1['id'], row_2['id'], weight=shared_hp)
        return G

    def create_labels(self):
        sc = SpectralClustering(4, affinity='precomputed', n_init=100)
        adj_mat = np.asarray(nx.to_numpy_matrix(self.G))
        sc.fit(adj_mat)
        y_sc = sc.labels_
        idx_features_labels = np.array(y_sc)
        #labels = self.encode_onehot(idx_features_labels[:])
        labels = idx_features_labels[:]
        labels = torch.LongTensor(labels)
        return labels

    def encode_onehot(self, labels):
        """
        encode as one hot for label
        """
        classes = set(labels)
        classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                        enumerate(classes)}
        labels_onehot = np.array(list(map(classes_dict.get, labels)),
                                dtype=np.int32)
        return labels_onehot

    def accuracy(self, output, labels):
        """
        """
        preds = output.max(1)[1].type_as(labels)
        correct = preds.eq(labels).double()
        correct = correct.sum()
        return correct / len(labels)

    def train(self, epoch):
        t = time.time()
        self.model.train()
        self.optimizer.zero_grad()
        output = self.model(self.features, self.adj)
        loss_train = F.nll_loss(output[self.idx_train], self.labels[self.idx_train])
        acc_train = self.accuracy(output[self.idx_train], self.labels[self.idx_train])
        loss_train.backward()
        self.optimizer.step()
        self.model.eval()
        output = self.model(self.features, self.adj)
        loss_val = F.nll_loss(output[self.idx_val], self.labels[self.idx_val])
        acc_val = self.accuracy(output[self.idx_val], self.labels[self.idx_val])
        
        if epoch % 10 ==0:
            print('Epoch: {:04d}'.format(epoch+1),
                    'loss_train: {:.4f}'.format(loss_train.item()),
                    'acc_train: {:.4f}'.format(acc_train.item()),
                    'loss_val: {:.4f}'.format(loss_val.item()),
                    'acc_val: {:.4f}'.format(acc_val.item()),
                    'time: {:.4f}s'.format(time.time() - t))


def main():
    t_start = time.time()
    model = Model()
    for epoch in range(model.epochs):
        model.train(epoch)
    print("Training Done")
    print("Time: {:.4f}s".format(time.time() - t_start)) 


if __name__ == "__main__":
    main() 