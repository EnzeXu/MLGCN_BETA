from torch_scatter import scatter
import torch
import pickle
import pandas as pd

import numpy as np
import os.path as osp
import os
from tqdm import tqdm
from torch_geometric.utils import degree
import random
import math
import matplotlib.pyplot as plt
from config import config


def draw_two_dimension(
        y_lists,
        x_list,
        color_list,
        line_style_list,
        legend_list=None,
        legend_fontsize=15,
        fig_title=None,
        fig_x_label="time",
        fig_y_label="val",
        show_flag=True,
        save_flag=False,
        save_path=None,
        save_dpi=300,
        fig_title_size=20,
        fig_grid=False,
        marker_size=0,
        line_width=2,
        x_label_size=15,
        y_label_size=15,
        number_label_size=15,
        fig_size=(8, 6)
) -> None:
    """
    Draw a 2D plot of several lines
    :param y_lists: (list[list]) y value of lines, each list in which is one line. e.g., [[2,3,4,5], [2,1,0,-1], [1,4,9,16]]
    :param x_list: (list) x value shared by all lines. e.g., [1,2,3,4]
    :param color_list: (list) color of each line. e.g., ["red", "blue", "green"]
    :param line_style_list: (list) line style of each line. e.g., ["solid", "dotted", "dashed"]
    :param legend_list: (list) legend of each line, which CAN BE LESS THAN NUMBER of LINES. e.g., ["red line", "blue line", "green line"]
    :param legend_fontsize: (float) legend fontsize. e.g., 15
    :param fig_title: (string) title of the figure. e.g., "Anonymous"
    :param fig_x_label: (string) x label of the figure. e.g., "time"
    :param fig_y_label: (string) y label of the figure. e.g., "val"
    :param show_flag: (boolean) whether you want to show the figure. e.g., True
    :param save_flag: (boolean) whether you want to save the figure. e.g., False
    :param save_path: (string) If you want to save the figure, give the save path. e.g., "./test.png"
    :param save_dpi: (integer) If you want to save the figure, give the save dpi. e.g., 300
    :param fig_title_size: (float) figure title size. e.g., 20
    :param fig_grid: (boolean) whether you want to display the grid. e.g., True
    :param marker_size: (float) marker size. e.g., 0
    :param line_width: (float) line width. e.g., 1
    :param x_label_size: (float) x label size. e.g., 15
    :param y_label_size: (float) y label size. e.g., 15
    :param number_label_size: (float) number label size. e.g., 15
    :param fig_size: (tuple) figure size. e.g., (8, 6)
    :return:
    """
    assert len(y_lists[0]) == len(x_list), "Dimension of y should be same to that of x"
    assert len(y_lists) == len(line_style_list) == len(color_list), "number of lines should be fixed"
    y_count = len(y_lists)
    plt.figure(figsize=fig_size)
    for i in range(y_count):
        plt.plot(x_list, y_lists[i], markersize=marker_size, linewidth=line_width, c=color_list[i], linestyle=line_style_list[i])
    plt.xlabel(fig_x_label, fontsize=x_label_size)
    plt.ylabel(fig_y_label, fontsize=y_label_size)
    plt.tick_params(labelsize=number_label_size)
    if legend_list:
        plt.legend(legend_list, fontsize=legend_fontsize)
    if fig_title:
        plt.title(fig_title, fontsize=fig_title_size)
    if fig_grid:
        plt.grid(True)
    if save_flag:
        plt.savefig(save_path, dpi=save_dpi)
    if show_flag:
        plt.show()
    plt.clf()
    plt.close()


def draw_two_dimension_regression(
        y_lists,
        x_lists,
        color_list,
        line_style_list,
        legend_list=None,
        legend_fontsize=15,
        fig_title=None,
        fig_x_label="time",
        fig_y_label="val",
        show_flag=True,
        save_flag=False,
        save_path=None,
        save_dpi=300,
        fig_title_size=15,
        fig_grid=False,
        marker_size=0,
        line_width=2,
        x_label_size=15,
        y_label_size=15,
        number_label_size=15,
        fig_size=(8, 6)
) -> None:
    """
    Draw a 2D plot of several lines
    :param y_lists: (list[list]) y value
    :param x_lists: (list[list]) x value
    :param color_list: (list) color of each line. e.g., ["red", "blue", "green"]
    :param line_style_list: (list) line style of each line. e.g., ["solid", "dotted", "dashed"]
    :param legend_list: (list) legend of each line, which CAN BE LESS THAN NUMBER of LINES. e.g., ["red line", "blue line", "green line"]
    :param legend_fontsize: (float) legend fontsize. e.g., 15
    :param fig_title: (string) title of the figure. e.g., "Anonymous"
    :param fig_x_label: (string) x label of the figure. e.g., "time"
    :param fig_y_label: (string) y label of the figure. e.g., "val"
    :param show_flag: (boolean) whether you want to show the figure. e.g., True
    :param save_flag: (boolean) whether you want to save the figure. e.g., False
    :param save_path: (string) If you want to save the figure, give the save path. e.g., "./test.png"
    :param save_dpi: (integer) If you want to save the figure, give the save dpi. e.g., 300
    :param fig_title_size: (float) figure title size. e.g., 20
    :param fig_grid: (boolean) whether you want to display the grid. e.g., True
    :param marker_size: (float) marker size. e.g., 0
    :param line_width: (float) line width. e.g., 1
    :param x_label_size: (float) x label size. e.g., 15
    :param y_label_size: (float) y label size. e.g., 15
    :param number_label_size: (float) number label size. e.g., 15
    :param fig_size: (tuple) figure size. e.g., (8, 6)
    :return:
    """
    y_count = len(y_lists)
    for i in range(y_count):
        assert len(y_lists[i]) == len(x_lists[i]), "Dimension of y should be same to that of x"
    assert len(y_lists) == len(x_lists) == len(line_style_list) == len(color_list), "number of lines should be fixed"

    plt.figure(figsize=fig_size)
    for i in range(y_count):
        # plt.plot(x_lists[i], y_lists[i], markersize=marker_size, linewidth=line_width, c=color_list[i], linestyle=line_style_list[i])
        fit = np.polyfit(x_lists[i], y_lists[i], 1)
        line_fn = np.poly1d(fit)
        y_line = line_fn(x_lists[i])
        plt.scatter(x_lists[i], y_lists[i])
        plt.plot(x_lists[i], y_line, markersize=marker_size, linewidth=line_width, c=color_list[i], linestyle=line_style_list[i])
    plt.xlabel(fig_x_label, fontsize=x_label_size)
    plt.ylabel(fig_y_label, fontsize=y_label_size)
    plt.tick_params(labelsize=number_label_size)
    plt.tight_layout()
    if legend_list:
        plt.legend(legend_list, fontsize=legend_fontsize)
    if fig_title:
        plt.title(fig_title, fontsize=fig_title_size)
    if fig_grid:
        plt.grid(True)
    if save_flag:
        plt.savefig(save_path, dpi=save_dpi)
    if show_flag:
        plt.show()
    plt.clf()
    plt.close()


def tensor_min_max_scaler(data, new_min=0.0, new_max=1.0):
    assert isinstance(data, torch.Tensor)
    data_min = torch.min(data, 0).values
    data_max = torch.max(data, 0).values
    assert 0.0 not in (data_max - data_min)
    core = (data - data_min) / (data_max - data_min)
    data_new = core * (new_max - new_min) + new_min
    return data_new


def numpy_min_max_scaler_1d(data, new_min=0.0, new_max=1.0):
    assert isinstance(data, np.ndarray)
    data_min = np.min(data)
    data_max = np.max(data)
    assert data_max - data_min > 0
    core = (data - data_min) / (data_max - data_min)
    data_new = core * (new_max - new_min) + new_min
    return data_new


def load_one_map(file_path):
    with open(file_path, "r") as f:
        data = f.readlines()
    data = [line.split() for line in data]
    data = np.asarray(data, dtype=float)
    assert data.shape[0] == data.shape[1], "load data error in file {}".format(file_path)
    return data


def load_one_charge(file_path):
    with open(file_path, "r") as f:
        data = f.readlines()
    data = [line.split()[1:] for line in data]
    data = np.asarray(data, dtype=float)
    lengths = [len(item) for item in data]
    assert min(lengths) == max(lengths) == 7, "load data error in file {}".format(file_path)
    return data


def load_one_coordinate(file_path):
    with open(file_path, "r") as f:
        data = f.readlines()
    data = [line.split() for line in data]
    data = np.asarray(data, dtype=float)
    # print(data.shape, data)
    return data


def generate_gap_file(folder, save_path, length, file_format, overwrite_flag=False):
    # files = os.listdir(folder)
    if osp.exists(save_path) and not overwrite_flag:
        print("Gap file {} exists. Skip generating ...".format(save_path))
        gaps = np.load(save_path)
        return gaps
    print("{}: {} files".format(folder, length))
    # files.sort()
    gaps = []
    # print(files[:30])
    for i in range(length):
        filename = osp.join(folder, file_format.format(i + 1))
        with open(filename, "r") as f:
            lines = f.readlines()
        if len(lines) < 80 or (float(lines[40]) - float(lines[39]) < 1.0 and float(lines[41]) - float(lines[40]) < 1.0) or float(lines[40]) - float(lines[39]) > 10 or float(lines[41]) - float(lines[40]) > 10:
            print(filename, len(lines))
            one_gap = np.mean(np.asarray(gaps))
        else:
            one_gap = float(lines[40]) - float(lines[39])
        # print(one_gap)
        gaps.append(one_gap)
    gaps = np.asarray(gaps)
    print(len(gaps))
    np.save(save_path, gaps)
    return gaps

def generate_atom_edges(edge_index_0, atom_num):
    item_list, item_count = edge_index_0.unique(return_counts=True)
    dic = {int(item_list[i]): int(item_count[i]) for i in range(item_list.shape[0])}
    atom_edges_array = torch.tensor([[dic[i]] if i in dic else [0] for i in range(atom_num)], dtype=torch.float32)
    # print(atom_edges_array)
    return atom_edges_array

def generate_dataset(input_path, output_path, config):
    assert osp.exists(input_path)
    if not osp.exists(osp.join(output_path, "raw")):
        print("Created new folder: {}".format(osp.join(output_path, "raw")))
        os.makedirs(osp.join(output_path, "raw"))
    gaps = generate_gap_file(osp.join(input_path, "EIGENVALS"), osp.join(input_path, "{}_gaps.npy".format(config.dataset)), config.length, config.format_eigen)
    gaps = numpy_min_max_scaler_1d(gaps)
    datasets = []
    for i in tqdm(range(config.length)):
        bt_path = osp.join(input_path, "BTMATRIXES", config.format_bmat.format(i + 1))
        bt_data = load_one_map(bt_path)
        bt_data = torch.tensor(bt_data)
        d_path = osp.join(input_path, "DMATRIXES", config.format_dmat.format(i + 1))
        d_data = load_one_map(d_path)
        d_data = torch.tensor(d_data)
        c_path = osp.join(input_path, "CONFIGS", config.format_conf.format(i + 1))
        c_data = load_one_coordinate(c_path)
        c_data = torch.tensor(c_data, dtype=torch.float32)
        charge_path = osp.join(input_path, "CHARGES", config.format_charge.format(i + 1))
        charge_data = load_one_charge(charge_path)
        charge_data = charge_data[:, [0, 2, 3, 4, 5, 6]]
        charge_data = torch.tensor(charge_data)

        matrix_data = d_data * bt_data  # 4.0 * (d_data - 1.2) * bt_data

        edge_index = bt_data.nonzero().t().contiguous()
        # edge_attr_from_dis = d_data[edge_index[0], edge_index[1]].to(torch.float32)
        # edge_attr = torch.tensor([[edge_attr_from_dis[i], 0.0, 0.0, 0.0] for i in range(edge_attr_from_dis.shape[0])],
        #                          dtype=torch.float32)
        node_degree = generate_atom_edges(edge_index[0], config.max_natoms)
        assert node_degree.shape[0] == c_data.shape[0], "Need to match max_natoms!"

        atom_num_array = [6] * 54 + [7] * 71 + [15] * 1
        electron_num_array = [4] * 54 + [5] * 72
        atom_type_dict = {
            6: [1, 0, 0],
            7: [0, 1, 0],
            15: [0, 0, 1]
        }
        atom_type = torch.Tensor([atom_type_dict.get(item) for item in atom_num_array])
        atomic_number = torch.Tensor([[item] for item in atom_num_array])
        electron_number = torch.Tensor([[item] for item in electron_num_array])

        x_full = torch.cat([c_data, node_degree, charge_data], 1)  # pos 3 cols + atom edge 1 col = 4 cols of feature + 6 cols of charges
        x_full = tensor_min_max_scaler(x_full)
        # x_full = torch.cat([c_data, node_degree, atom_type, atomic_number, electron_number], 1)  # 9 cols of feature


        # np.set_printoptions(threshold=np.inf)
        # print(matrix_data)
        # print(bt_data[0])
        # print(d_data[0])
        # for ii in range(126):
        #     print("####" if bt_data[1][ii] == 1 else "", bt_data[1][ii], d_data[1][ii])
        # atom_num_array = torch.Tensor([[6]] * 54 + [[7]] * 71 + [[15]] * 1)
        # x_full = torch.cat([c_data, atom_num_array], 1)

        one_data = {
            'num_atom': config.max_natoms,
            'atom_type': x_full,
            'bond_type': matrix_data,
            'logP_SA_cycle_normalized': torch.Tensor([gaps[i]])
        }
        datasets.append(one_data)
    print("Finished! Train: {} Test: {} Val: {}".format(config.train_length, config.test_length, config.val_length))
    datasets_train = datasets[: config.train_length]
    datasets_test = datasets[config.train_length: config.train_length + config.test_length]
    datasets_val = datasets[config.train_length + config.test_length:]
    with open(osp.join(output_path, "raw/train.pickle"), "wb") as f:
        pickle.dump(datasets_train, f)
    with open(osp.join(output_path, "raw/test.pickle"), "wb") as f:
        pickle.dump(datasets_test, f)
    with open(osp.join(output_path, "raw/val.pickle"), "wb") as f:
        pickle.dump(datasets_val, f)
    with open(osp.join(output_path, "raw/all.pickle"), "wb") as f:
        pickle.dump(datasets, f)
    # np.save(osp.join(output_path, "raw/train.pickle"), datasets_train, allow_pickle=True)
    # np.save(osp.join(output_path, "raw/test.pickle"), datasets_test, allow_pickle=True)
    # np.save(osp.join(output_path, "raw/val.pickle"), datasets_val, allow_pickle=True)


def generate_fept_dataset(input_path, output_path, config):
    assert osp.exists(input_path)
    if not osp.exists(output_path):
        print("Created new folder: {}".format(output_path))
        os.makedirs(output_path)
    gaps = generate_gap_file(osp.join(input_path, "EIGENVALS"), osp.join(input_path, "{}_gaps.npy".format(config.dataset)), config.length, config.format_eigen)
    # datasets = []
    for i in tqdm(range(config.length)):
        bt_path = osp.join(input_path, "BTMATRIXES", config.format_bmat.format(i + 1))
        bt_data = load_one_map(bt_path)
        bt_data = torch.tensor(bt_data)
        d_path = osp.join(input_path, "DMATRIXES", config.format_dmat.format(i + 1))
        d_data = load_one_map(d_path)
        # d_data = torch.tensor(d_data)
        c_path = osp.join(input_path, "CONFIGS", config.format_conf.format(i + 1))
        c_data_raw = load_one_coordinate(c_path)
        c_data = torch.tensor(c_data_raw, dtype=torch.float32)
        # matrix_data = d_data * bt_data  # 4.0 * (d_data - 1.2) * bt_data

        edge_index = bt_data.nonzero().t().contiguous()
        edge_attr_from_dis = d_data[edge_index[0], edge_index[1]]

        node_degree = generate_atom_edges(edge_index[0], config.max_natoms)
        assert node_degree.shape[0] == c_data.shape[0], "Need to match max_natoms!"

        atom_num_array = [6] * 54 + [7] * 71 + [15] * 1
        electron_num_array = [4] * 54 + [5] * 72
        atom_type_dict = {
            6: [1, 0, 0],
            7: [0, 1, 0],
            15: [0, 0, 1]
        }
        atom_type = torch.Tensor([atom_type_dict.get(item) for item in atom_num_array])
        atomic_number = torch.Tensor([[item] for item in atom_num_array])
        electron_number = torch.Tensor([[item] for item in electron_num_array])

        # x_full = torch.cat([c_data, node_degree], 1)  # pos 3 cols + atom edge 1 col = 4 cols of feature
        x_full = torch.cat([c_data, node_degree, atom_type, atomic_number, electron_number], 1)  # 9 cols of feature

        node_distance_dict = dict()
        for j in range(126):
            node_distance_dict[j] = []
        for j in range(edge_attr_from_dis.shape[0]):
            node_distance_dict[edge_index.cpu().detach().numpy()[0][j]] += [edge_attr_from_dis[j]]
            node_distance_dict[edge_index.cpu().detach().numpy()[1][j]] += [edge_attr_from_dis[j]]
        # node_distance_avg = [sum(node_distance_dict[j]) / len(node_distance_dict[j]) for j in range(126)]
        node_distance_avg = [np.asarray(node_distance_dict[j]).mean() for j in range(126)]
        # print(node_distance_dict[2])
        # print("{:.12f}".format(node_distance_avg[2]))

        # print([len(node_distance_dict[j]) for j in range(126)])

        one_data = {
            "atom_num": atom_num_array,
            "atom_index": range(126),
            "x": c_data_raw[:, 0],
            "y": c_data_raw[:, 1],
            "z": c_data_raw[:, 2],
            "node_degree": [int(item[0]) for item in node_degree.cpu().detach().numpy()],
            "electron_num": [int(item[0]) for item in electron_number.cpu().detach().numpy()],
            "atom_type_0": [int(item[0]) for item in atom_type.cpu().detach().numpy()],
            "atom_type_1": [int(item[1]) for item in atom_type.cpu().detach().numpy()],
            "atom_type_2": [int(item[2]) for item in atom_type.cpu().detach().numpy()],
            "node_distance_avg": node_distance_avg,
        }

        one_data_format = {
            "atom_num": "{}",
            "atom_index": "{}",
            "x": "{:.15f}",
            "y": "{:.15f}",
            "z": "{:.15f}",
            "node_degree": "{}",
            "electron_num": "{}",
            "atom_type_0": "{}",
            "atom_type_1": "{}",
            "atom_type_2": "{}",
            "node_distance_avg": "{:.15f}",
        }

        one_data_format_rjust = {
            "atom_num": 3,
            "atom_index": 3,
            "x": 18,
            "y": 18,
            "z": 18,
            "node_degree": 3,
            "electron_num": 3,
            "atom_type_0": 3,
            "atom_type_1": 3,
            "atom_type_2": 3,
            "node_distance_avg": 18,
        }

        one_data_y = gaps[i]

        one_data_keys = ["atom_num", "atom_index", "x", "y", "z", "node_degree", "electron_num", "atom_type_0", "atom_type_1", "atom_type_2", "node_distance_avg"]

        string = ""
        string += "{:.15f}".format(one_data_y) + "\n"
        for j in range(126):
            string += "\t".join([one_data_format[one_key].format(one_data[one_key][j]).rjust(one_data_format_rjust[one_key]) for one_key in one_data_keys])
            string += "\n"
        with open(osp.join(output_path, "out_{}".format(i)), "w") as f:
            f.write(string)

        # datasets.append(one_data)
    # print("Finished! Train: {} Test: {} Val: {}".format(config.train_length, config.test_length, config.val_length))
    # datasets_train = datasets[: config.train_length]
    # datasets_test = datasets[config.train_length: config.train_length + config.test_length]
    # datasets_val = datasets[config.train_length + config.test_length:]
    # with open(osp.join(output_path, "raw/train.pickle"), "wb") as f:
    #     pickle.dump(datasets_train, f)
    # with open(osp.join(output_path, "raw/test.pickle"), "wb") as f:
    #     pickle.dump(datasets_test, f)
    # with open(osp.join(output_path, "raw/val.pickle"), "wb") as f:
    #     pickle.dump(datasets_val, f)
    # np.save(osp.join(output_path, "raw/train.pickle"), datasets_train, allow_pickle=True)
    # np.save(osp.join(output_path, "raw/test.pickle"), datasets_test, allow_pickle=True)
    # np.save(osp.join(output_path, "raw/val.pickle"), datasets_val, allow_pickle=True)

def worker_init_fn(worker_id, seed=0):
    random.seed(seed + worker_id)


def generate_deg(dataset):
    max_degree = -1
    for data in dataset:
        # print("data.num_nodes:", data.num_nodes)
        # print("data.edge_index[1]:", data.edge_index[1].shape)
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        max_degree = max(max_degree, int(d.max()))

    # Compute the in-degree histogram tensor
    deg = torch.zeros(max_degree + 1, dtype=torch.long)
    for data in dataset:
        d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
        deg += torch.bincount(d, minlength=deg.numel())
    return deg


def compute_correlation(x, y):
    xBar = np.mean(x)
    yBar = np.mean(y)
    SSR = 0.0
    varX = 0.0
    varY = 0.0
    for i in range(0, len(x)):
        diffXXbar = x[i] - xBar
        difYYbar = y[i] - yBar
        SSR += (diffXXbar * difYYbar)
        varX += diffXXbar ** 2
        varY += difYYbar ** 2
    SST = math.sqrt(varX * varY)
    if SST == 0.0:
        return -1
    return SSR / SST





if __name__ == "__main__":
    # data = pd.read_csv("../data/ZINC/raw/atom_dict.pickle")
    # print(data)
    # with open("../data/ZINC/raw/atom_dict.pickle", "rb") as f:
    #     data_atom = pickle.load(f)
    # print(type(data_atom), data_atom.shape)
    # print(data_atom)

    # with open("../data/ZINC/raw/train.pickle", 'r') as f:
    #     data = [x.split('\t') for x in f.read().split('\n')[1:-1]]
    #
    #     rows, cols = [], []
    #     for n_id, col, _ in data:
    #         col = [int(x) for x in col.split(',')]
    #         rows += [int(n_id)] * len(col)
    #         cols += col
    #     x = SparseTensor(row=torch.tensor(rows), col=torch.tensor(cols))
    #     x = x.to_dense()
    #
    # print(x)
    # data = np.load("../data/ZINC/raw/test.pickle", allow_pickle=True)
    #
    # print(len(data))

    # with open("../data/GCN_N3P/raw/train.pickle", "rb") as f:
    #     data = pickle.load(f)
    # print(len(data))
    # # data = np.load("../data/ZINC/raw/test.index", allow_pickle=True)
    # # with open("../data/ZINC/raw/val.index", 'r') as f:
    # #     data = f.readline()
    # # print(data)
    # # print(len(data))
    # # data = data.split(",")
    # # print(data)
    # # print(len(data))
    # #
    # # data = np.load("../data/ZINC/raw/val.pickle", allow_pickle=True)
    # print(type(data))
    # print(len(data))
    # for i in range(5):
    #     # print(type(data[i]))
    #     # print(data[i].keys())
    # # print(type(data[-1]))
    # # print(data[-1].keys())
    #     if data[i]["num_atom"] != 24:
    #         print(i)
    #         print(data[i])

    # with open("../data/ZINC/raw/val.index", 'w') as f:
    #     f.write("abcdefg")
    # generate_dataset("../../MLGCN/data/GCN_N3P/")
    # load_one_coordinate("../../MLGCN/data/GCN_N3P/CONFIGS/COORD_1")
    # generate_gap_file("../../MLGCN/data/GCN_N3P/EIGENVALS/", "../../MLGCN/data/GCN_N3P/GCN_N3P_gaps.npy")
    # a = [1,2,3,4,5,6,7]
    #
    # print(a[slice([2,4,5])])
    # print(type({"1":222, "2":333}))

    generate_dataset("data/GCN_N3P/", "dataset/GCN_N3P/", config)
    # a = np.asarray([1,2])
    # print(type(a))
    # charge = load_one_charge("data/GCN_N3P/CHARGES/charge_data_1")
    # print(charge)
    # print(charge[:,[0,2,3,4,5,6]])
    # a = torch.tensor([[1,6,3], [5,2,6], [3,4,7]])
    # min_max_scaler(a)
    # print(1.01 in a)
    # b = torch.tensor([1,2,3])
    # print(a - b)

    # generate_fept_dataset("data/GCN_N3P/", "dataset/GCN_N3P_FePt_Format/", config)
    # print("{:3d} + {:.2f}".format(12, 3.222))
    # a = np.asarray([[1,2,3], [4,5,6]])
    # print(a[:,1])
    # with open("test.txt", "w") as f:
    #     f.write("{}\t\t{}\t\t{}".format("78".rjust(3), "99".rjust(3), "1".rjust(3)))
    pass
