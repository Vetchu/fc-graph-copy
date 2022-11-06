import os
import warnings
import networkx as nx
import numpy as np
import pandas as pd
import torch
from sklearn.cluster import SpectralClustering

from utils.utils import load_data_loader

warnings.simplefilter(action='ignore', category=FutureWarning)

class Patients:
    def __init__(self, path):
        self.raw_data = self.read_data_from_csv(path)
        self.patientsdf = self.load_patients()

    def load_patients(self):
        # extract all existing phenotypes
        self.all_phenotypes = self.get_all_phenotypes()
        # create patients dataframe with id and phenotypes as columns
        patientsdf = self.extract_patients()
        return patientsdf

    def read_data_from_csv(self, path):
        return pd.read_csv(path, sep=";", names=['id', 'hpo'], header=None)

    def get_all_phenotypes(self):
        # extract all existing phenotypes 
        all_phenotypes = []
        for _, row in self.raw_data.iterrows():
            patient_list = row.to_string().split(";", 1)
            hp = patient_list[0].split()[3]  # works differently in ipynb and py
            if hp not in all_phenotypes:
                all_phenotypes.append(hp)
        return all_phenotypes

    def extract_patients(self):
        df_columns = ['id'] + self.all_phenotypes
        patientsdf = pd.DataFrame(columns=df_columns)

        first_id = self.raw_data.iloc[0]['id']
        patient_dict = dict()

        for _, row in self.raw_data.iterrows():
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

                patientsdf = patientsdf.append(patient_dict, ignore_index=True)

                # prepare the next patient
                first_id = id_patient
                phenotypes_dict = dict()
                phenotypes_dict[hp] = 1

                # append last patient
        patient_dict['id'] = first_id
        for phenotype in self.all_phenotypes:
            if phenotype not in patient_dict.keys():
                patient_dict[phenotype] = 0
        patientsdf = patientsdf.append(patient_dict, ignore_index=True)
        return patientsdf


class GraphCSVInput():
    def __init__(self, path):
        self.patients = Patients(path)
        self.features = self.patients.patientsdf.iloc[:, self.patients.patientsdf.columns != 'id'].values
        self.features = torch.tensor(self.features.astype('float'), dtype=torch.float32)
        self.G = self.create_graph(self.patients.patientsdf)
        self.adj = self.create_adj()
        self.labels = self.create_labels()

    def create_adj(self):
        adj_mat = np.asarray(nx.to_numpy_matrix(self.G))
        adj = adj_mat + adj_mat.T * (adj_mat.T > adj_mat) - adj_mat @ (adj_mat.T > adj_mat)
        adj = torch.tensor(adj, dtype=torch.float32)
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
        # labels = self.encode_onehot(idx_features_labels[:])
        labels = idx_features_labels[:]
        labels = torch.LongTensor(labels)
        return labels


class DataReader:
    def __init__(self, file_format):
        self.file_format = file_format

    def read_file(self, path):
        if path is None or path.split("/")[-1] == 'None':
            print(f"'None' cannot be a file name ({path})."
                  f"The program consider this as NO FILE IS REQUIRED!")
            return None
        if os.path.exists(path):
            if self.file_format == 'csv':
                graphcsvinput = GraphCSVInput(path)
                x = graphcsvinput.features
                y = graphcsvinput.labels
                return x, y
            print(f"{self.file_format} is not supported file format")
        else:
            print(f"{path} File Not Found!!!")
            print(f" Program will be terminated!!!")
            exit(0)


class DataLoader(DataReader):
    def __init__(self, train_path, test_path, train_batch_size=None, test_batch_size=32, torch_mode=True):
        path = train_path if train_path is not None else test_path
        super().__init__(file_format=path.strip().split(".")[-1].strip())
        self.train_path = train_path
        self.test_path = test_path

        self.train_loader = None
        self.test_loader = None
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.torch_mode = torch_mode

    @property
    def sample_data(self):
        return self.read_file(self.train_path)

    def lazy_init(self, train_batch_size, test_batch_size, torch_mode=True):
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.torch_mode = torch_mode
        if self.torch_mode:
            self.load_data_loader()
        else:
            self.train_loader = IterLoader(self.train_path, self.train_batch_size)
            self.train_loader_for_test = IterLoader(self.train_path, self.test_batch_size)

    def load_data_loader(self):
        data = self.read_file(self.train_path)
        if data is not None:
            x_train, y_train = data
            self.train_loader = load_data_loader(x_train, y_train, self.train_batch_size)
        data = self.read_file(self.test_path)
        if data is not None:
            x_test, y_test = data
            self.test_loader = load_data_loader(x_test, y_test, self.test_batch_size)


class IterLoader(DataReader):
    def __init__(self, path, batch_size):
        super().__init__(path.strip().split(".")[-1].strip())
        self.path = path
        self.batch_size = batch_size
        self.batches = None
        self.max = 0
        self.n = 0
        x, y = self.read_file(self.path)
        n_samples = len(x)
        self.len = n_samples - (n_samples % self.batch_size)

    def __iter__(self):
        self.load()
        return self

    def __next__(self):
        try:
            x, y = self.batches[self.n]
        except:
            raise StopIteration()
        dl = load_data_loader(x, y, len(x))
        self.n += 1
        return next(iter(dl))

    def get_sample_data(self):
        x, y = self.batches[0]
        dl = load_data_loader(x, y, 1)
        return next(iter(dl))

    def load(self):
        samples, labels = self.read_file(self.path)
        n_samples = len(samples)
        n_batches = n_samples // self.batch_size

        samples_indices = np.arange(n_samples)
        np.random.shuffle(samples_indices)
        batches_idx = np.split(samples_indices[:self.batch_size * n_batches], n_batches)
        self.batches = []
        for batch_ind in batches_idx:
            x_batch = samples[batch_ind]
            y_batch = labels[batch_ind]
            self.batches.append([x_batch, y_batch])
        self.max = len(self.batches)
        self.n = 0
        self.len = self.max * self.batch_size

    def __len__(self):
        return self.len
