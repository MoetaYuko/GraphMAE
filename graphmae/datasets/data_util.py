
from collections import namedtuple, Counter
import numpy as np
import os.path as osp

import torch
import torch.nn.functional as F
import scipy.sparse as sp

import dgl
from dgl.data import (
    load_data,
    DGLDataset,
    TUDataset,
    CoraGraphDataset,
    CiteseerGraphDataset,
    PubmedGraphDataset
)
from dgl.data.utils import download, loadtxt
from dgl.data.citation_graph import _preprocess_features
from dgl.transforms import reorder_graph
from ogb.nodeproppred import DglNodePropPredDataset
from dgl.data.ppi import PPIDataset
from dgl.dataloading import GraphDataLoader


from sklearn.preprocessing import StandardScaler


class SDCNDataset(DGLDataset):
    _url_prefix = 'https://github.com/bdy9527/SDCN/raw/master'

    def __init__(self,
                 name,
                 raw_dir=None,
                 force_reload=False,
                 verbose=True,
                 transform=None):
        assert name.lower() in ['acm', 'dblp']

        super(SDCNDataset, self).__init__(name,
                                          raw_dir=raw_dir,
                                          force_reload=force_reload,
                                          verbose=verbose,
                                          transform=transform)

    def process(self):
        root = self.raw_path
        filenames = osp.join(root, self.name.lower() + '{}.txt')
        objnames = ['', '_label', '_graph']

        features = loadtxt(filenames.format(objnames[0]), ' ')
        features = sp.coo_matrix(features, dtype=float)
        labels = loadtxt(filenames.format(objnames[1]), ' ').flatten()
        graph = loadtxt(filenames.format(objnames[2]), ' ')

        g = dgl.graph((graph[:, 0], graph[:, 1]), num_nodes=len(labels))

        g.ndata['label'] = dgl.backend.tensor(labels)
        g.ndata['feat'] = dgl.backend.tensor(
            _preprocess_features(features),
            dtype=dgl.backend.data_type_dict['float32'])
        self._num_classes = len(np.unique(labels))
        self._labels = labels
        self._g = reorder_graph(
            g, node_permute_algo='rcmk', edge_permute_algo='dst', store_ids=False)

        if self.verbose:
            print('Finished data loading and preprocessing.')
            print('  NumNodes: {}'.format(self._g.number_of_nodes()))
            print('  NumEdges: {}'.format(self._g.number_of_edges()))
            print('  NumFeats: {}'.format(self._g.ndata['feat'].shape[1]))
            print('  NumClasses: {}'.format(self.num_classes))

    def __getitem__(self, idx):
        assert idx == 0, "This dataset has only one graph"
        if self._transform is None:
            return self._g
        else:
            return self._transform(self._g)

    def __len__(self):
        return 1

    def download(self):
        dirs = {'data': ['', '_label'], 'graph': ['_graph']}
        for d, suffices in dirs.items():
            for suffix in suffices:
                filename = f'{self.name.lower()}{suffix}.txt'
                download(
                    f'{self._url_prefix}/{d}/{filename}',
                    osp.join(self.raw_path, filename))

    @property
    def num_classes(self):
        return self._num_classes


class ACMGraphDataset(SDCNDataset):
    def __init__(self,
                 raw_dir=None,
                 force_reload=False,
                 verbose=True,
                 transform=None):
        super().__init__('acm', raw_dir, force_reload, verbose, transform)


class DBLPGraphDataset(SDCNDataset):
    def __init__(self,
                 raw_dir=None,
                 force_reload=False,
                 verbose=True,
                 transform=None):
        super().__init__('dblp', raw_dir, force_reload, verbose, transform)


GRAPH_DICT = {
    "cora": CoraGraphDataset,
    "citeseer": CiteseerGraphDataset,
    "pubmed": PubmedGraphDataset,
    "acm": ACMGraphDataset,
    "dblp": DBLPGraphDataset,
    "ogbn-arxiv": DglNodePropPredDataset
}


def preprocess(graph):
    feat = graph.ndata["feat"]
    graph = dgl.to_bidirected(graph)
    graph.ndata["feat"] = feat

    graph = graph.remove_self_loop().add_self_loop()
    graph.create_formats_()
    return graph


def scale_feats(x):
    scaler = StandardScaler()
    feats = x.numpy()
    scaler.fit(feats)
    feats = torch.from_numpy(scaler.transform(feats)).float()
    return feats


def load_dataset(dataset_name):
    assert dataset_name in GRAPH_DICT, f"Unknow dataset: {dataset_name}."
    if dataset_name.startswith("ogbn"):
        dataset = GRAPH_DICT[dataset_name](dataset_name)
    else:
        dataset = GRAPH_DICT[dataset_name]()

    if dataset_name == "ogbn-arxiv":
        graph, labels = dataset[0]
        num_nodes = graph.num_nodes()

        split_idx = dataset.get_idx_split()
        train_idx, val_idx, test_idx = split_idx["train"], split_idx["valid"], split_idx["test"]
        graph = preprocess(graph)

        if not torch.is_tensor(train_idx):
            train_idx = torch.as_tensor(train_idx)
            val_idx = torch.as_tensor(val_idx)
            test_idx = torch.as_tensor(test_idx)

        feat = graph.ndata["feat"]
        feat = scale_feats(feat)
        graph.ndata["feat"] = feat

        train_mask = torch.full((num_nodes,), False).index_fill_(0, train_idx, True)
        val_mask = torch.full((num_nodes,), False).index_fill_(0, val_idx, True)
        test_mask = torch.full((num_nodes,), False).index_fill_(0, test_idx, True)
        graph.ndata["label"] = labels.view(-1)
        graph.ndata["train_mask"], graph.ndata["val_mask"], graph.ndata["test_mask"] = train_mask, val_mask, test_mask
    else:
        graph = dataset[0]
        graph = graph.remove_self_loop()
        graph = graph.add_self_loop()
    num_features = graph.ndata["feat"].shape[1]
    num_classes = dataset.num_classes
    return graph, (num_features, num_classes)


def load_inductive_dataset(dataset_name):
    if dataset_name == "ppi":
        batch_size = 2
        # define loss function
        # create the dataset
        train_dataset = PPIDataset(mode='train')
        valid_dataset = PPIDataset(mode='valid')
        test_dataset = PPIDataset(mode='test')
        train_dataloader = GraphDataLoader(train_dataset, batch_size=batch_size)
        valid_dataloader = GraphDataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = GraphDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        eval_train_dataloader = GraphDataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        g = train_dataset[0]
        num_classes = train_dataset.num_labels
        num_features = g.ndata['feat'].shape[1]
    else:
        _args = namedtuple("dt", "dataset")
        dt = _args(dataset_name)
        batch_size = 1
        dataset = load_data(dt)
        num_classes = dataset.num_classes

        g = dataset[0]
        num_features = g.ndata["feat"].shape[1]

        train_mask = g.ndata['train_mask']
        feat = g.ndata["feat"]
        feat = scale_feats(feat)
        g.ndata["feat"] = feat

        g = g.remove_self_loop()
        g = g.add_self_loop()

        train_nid = np.nonzero(train_mask.data.numpy())[0].astype(np.int64)
        train_g = dgl.node_subgraph(g, train_nid)
        train_dataloader = [train_g]
        valid_dataloader = [g]
        test_dataloader = valid_dataloader
        eval_train_dataloader = [train_g]

    return train_dataloader, valid_dataloader, test_dataloader, eval_train_dataloader, num_features, num_classes



def load_graph_classification_dataset(dataset_name, deg4feat=False):
    dataset_name = dataset_name.upper()
    dataset = TUDataset(dataset_name)
    graph, _ = dataset[0]

    if "attr" not in graph.ndata:
        if "node_labels" in graph.ndata and not deg4feat:
            print("Use node label as node features")
            feature_dim = 0
            for g, _ in dataset:
                feature_dim = max(feature_dim, g.ndata["node_labels"].max().item())

            feature_dim += 1
            for g, l in dataset:
                node_label = g.ndata["node_labels"].view(-1)
                feat = F.one_hot(node_label, num_classes=feature_dim).float()
                g.ndata["attr"] = feat
        else:
            print("Using degree as node features")
            feature_dim = 0
            degrees = []
            for g, _ in dataset:
                feature_dim = max(feature_dim, g.in_degrees().max().item())
                degrees.extend(g.in_degrees().tolist())
            MAX_DEGREES = 400

            oversize = 0
            for d, n in Counter(degrees).items():
                if d > MAX_DEGREES:
                    oversize += n
            # print(f"N > {MAX_DEGREES}, #NUM: {oversize}, ratio: {oversize/sum(degrees):.8f}")
            feature_dim = min(feature_dim, MAX_DEGREES)

            feature_dim += 1
            for g, l in dataset:
                degrees = g.in_degrees()
                degrees[degrees > MAX_DEGREES] = MAX_DEGREES

                feat = F.one_hot(degrees, num_classes=feature_dim).float()
                g.ndata["attr"] = feat
    else:
        print("******** Use `attr` as node features ********")
        feature_dim = graph.ndata["attr"].shape[1]

    labels = torch.tensor([x[1] for x in dataset])

    num_classes = torch.max(labels).item() + 1
    dataset = [(g.remove_self_loop().add_self_loop(), y) for g, y in dataset]

    print(f"******** # Num Graphs: {len(dataset)}, # Num Feat: {feature_dim}, # Num Classes: {num_classes} ********")

    return dataset, (feature_dim, num_classes)
