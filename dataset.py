import os
import os.path as osp
import pickle
import shutil

import torch
from tqdm import tqdm

from torch_geometric.data import (
    Data,
    InMemoryDataset,
    download_url,
    extract_zip,
)


class MyDataset(InMemoryDataset):
    # url = 'https://www.dropbox.com/s/feo9qle74kg48gy/molecules.zip?dl=1'
    # split_url = ('https://raw.githubusercontent.com/graphdeeplearning/'
    #              'benchmarking-gnns/master/data/molecules/{}.index')

    def __init__(self, root, subset=False, split='train', transform=None,
                 pre_transform=None, pre_filter=None):
        self.subset = subset
        assert split in ['train', 'val', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        path = osp.join(self.processed_dir, f'{split}.pt')
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return [
            'train.pickle', 'val.pickle', 'test.pickle', 'train.index',
            'val.index', 'test.index'
        ]

    @property
    def processed_dir(self):
        name = 'subset' if self.subset else 'full'
        return osp.join(self.root, name, 'processed')

    @property
    def processed_file_names(self):
        return ['train.pt', 'val.pt', 'test.pt']

    def download(self):
        print("Data not found!")
        raise RuntimeError
        # shutil.rmtree(self.raw_dir)
        # path = download_url(self.url, self.root)
        # extract_zip(path, self.root)
        # os.rename(osp.join(self.root, 'molecules'), self.raw_dir)
        # os.unlink(path)
        #
        # for split in ['train', 'val', 'test']:
        #     download_url(self.split_url.format(split), self.raw_dir)

    def process(self):
        for split in ['train', 'val', 'test']:
            print("finding", osp.join(self.raw_dir, f'{split}.pickle'))
            with open(osp.join(self.raw_dir, f'{split}.pickle'), 'rb') as f:
                mols = pickle.load(f)

            indices = range(len(mols))

            if self.subset:
                with open(osp.join(self.raw_dir, f'{split}.index'), 'r') as f:
                    indices = [int(x) for x in f.read()[:-1].split(',')]

            pbar = tqdm(total=len(indices))
            pbar.set_description(f'Processing {split} dataset')

            data_list = []
            for idx in indices:
                mol = mols[idx]

                x = mol['atom_type'].to(torch.float32).view(-1, mol['atom_type'].shape[1]) #  x = mol['atom_type'].to(torch.long).view(-1, 1)
                y = mol['logP_SA_cycle_normalized'].to(torch.float)

                adj = mol['bond_type']
                edge_index = adj.nonzero(as_tuple=False).t().contiguous()
                edge_attr = adj[edge_index[0], edge_index[1]].to(torch.float32)  # torch.long
                # print("x:", x)
                data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                            y=y)

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)
                pbar.update(1)

            pbar.close()

            torch.save(self.collate(data_list),
                       osp.join(self.processed_dir, f'{split}.pt'))
