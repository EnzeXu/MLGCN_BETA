import torch
import time
import os.path as osp
import argparse
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchsummary import summary

from config import config
from model_gcn import MyNetwork, train, test
from dataset import MyDataset
from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "3"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=config.epoch, help="epoch")
    parser.add_argument("--epoch_step", type=int, default=config.epoch_step, help="epoch_step")
    parser.add_argument("--batch_size", type=int, default=config.batch_size, help="batch_size")
    parser.add_argument('--lr', type=float, default=config.lr, help='learning rate, default=0.01')
    parser.add_argument('--seed', type=int, default=config.seed, help='seed')
    parser.add_argument("--main_path", type=str, default=config.main_path, help="main_path")
    parser.add_argument("--dataset", type=str, default=config.dataset, help="dataset")
    parser.add_argument("--dataset_save_as", type=str, default=config.dataset, help="dataset_save_as")
    parser.add_argument("--max_natoms", type=int, default=config.max_natoms, help="max_natoms")
    parser.add_argument("--length", type=int, default=config.length, help="important: data length")
    parser.add_argument("--root_bmat", type=str, default=config.root_bmat, help="root_bmat")
    parser.add_argument("--root_dmat", type=str, default=config.root_dmat, help="root_dmat")
    parser.add_argument("--root_conf", type=str, default=config.root_conf, help="root_conf")
    parser.add_argument("--format_bmat", type=str, default=config.format_bmat, help="format_bmat")
    parser.add_argument("--format_dmat", type=str, default=config.format_dmat, help="format_dmat")
    parser.add_argument("--format_conf", type=str, default=config.format_conf, help="format_conf")
    parser.add_argument("--format_eigen", type=str, default=config.format_eigen, help="format_eigen")
    parser.add_argument("--loss_fn_id", type=int, default=config.loss_fn_id, help="loss_fn_id")
    parser.add_argument("--gpu", type=int, default=True, help="using gpu or not")
    args = parser.parse_args()
    args.train_length = int(args.length * 0.8)
    args.test_length = int(args.length * 0.1)
    args.val_length = args.length - args.train_length - args.test_length

    args.device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")  # device = "cpu"

    print("[Step 1] Configurations")
    print("using: {}".format(args.device))
    for item in args.__dict__.items():
        if item[0][0] == "_":
            continue
        print("{}: {}".format(item[0], item[1]))

    print("[Step 2] Preparing dataset...")
    dataset_path = osp.join(args.main_path, 'dataset', args.dataset)
    train_dataset = MyDataset(dataset_path, subset=False, split='train')
    test_dataset = MyDataset(dataset_path, subset=False, split='test')
    val_dataset = MyDataset(dataset_path, subset=False, split='val')

    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, worker_init_fn=worker_init_fn, generator=g)

    deg = generate_deg(train_dataset)

    print("[Step 3] Initializing model")
    main_save_path = osp.join(args.main_path, "train", args.dataset_save_as)
    if not os.path.exists(main_save_path):
        os.makedirs(main_save_path)

    model_save_path = osp.join(main_save_path, "model_last.pt")
    figure_save_path_loss_whole = osp.join(main_save_path, "loss_whole.png")
    figure_save_path_loss_last_half = osp.join(main_save_path, "loss_last_half.png")
    figure_save_path_loss_last_quarter = osp.join(main_save_path, "loss_last_quarter.png")

    regression_result_train_true = osp.join(main_save_path, "train_true.npy")
    regression_result_train_pred = osp.join(main_save_path, "train_pred.npy")
    regression_result_val_true = f"{main_save_path}/val_true.npy"
    regression_result_val_pred = f"{main_save_path}/val_pred.npy"
    regression_result_test_true = f"{main_save_path}/test_true.npy"
    regression_result_test_pred = f"{main_save_path}/test_pred.npy"
    figure_regression_train_path = f"{main_save_path}/regression_train.png"
    figure_regression_val_path = f"{main_save_path}/regression_val.png"
    figure_regression_test_path = f"{main_save_path}/regression_test.png"
    print("main_save_path: {}".format(main_save_path))
    print("model_save_path: {}".format(model_save_path))
    print("figure_save_path_loss_whole: {}".format(figure_save_path_loss_whole))
    print("figure_save_path_loss_last_half: {}".format(figure_save_path_loss_last_half))
    print("figure_save_path_loss_last_quarter: {}".format(figure_save_path_loss_last_quarter))
    print("regression_result_train_true: {}".format(regression_result_train_true))
    print("regression_result_train_pred: {}".format(regression_result_train_pred))
    print("regression_result_val_true: {}".format(regression_result_val_true))
    print("regression_result_val_pred: {}".format(regression_result_val_pred))
    print("regression_result_test_true: {}".format(regression_result_test_true))
    print("regression_result_test_pred: {}".format(regression_result_test_pred))
    print("figure_regression_train_path: {}".format(figure_regression_train_path))
    print("figure_regression_val_path: {}".format(figure_regression_val_path))
    print("figure_regression_test_path: {}".format(figure_regression_test_path))

    model = MyNetwork(deg).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, min_lr=0.00001)
    # summary(model, [(126, 4), (2, 324), (324,), (126,)])  # ((8064, 1), (2, 20736), 20736, 8064)
    # summary(model)
    print(model)
    print("[Step 4] Training...")

    start_time = time.time()
    start_time_0 = start_time

    epoch_loss_list = []
    for epoch in range(1, args.epoch + 1):
        model, train_loss = train(model, args, train_loader, optimizer)
        val_loss = test(model, args, val_loader)
        test_loss = test(model, args, test_loader)
        scheduler.step(val_loss)
        epoch_loss_list.append(train_loss)

        if epoch % args.epoch_step == 0:
            now_time = time.time()
            print("Epoch [{0:05d}/{1:05d}] Loss_train:{2:.6f} Loss_val:{3:.6f} Loss_test:{4:.6f} Lr:{5:.6f} (Time:{6:.6f}s Time total:{7:.2f}min Time remain: {8:.2f}min)".format(epoch, args.epoch, train_loss, val_loss, test_loss, optimizer.param_groups[0]["lr"], now_time - start_time, (now_time - start_time_0) / 60.0, (now_time - start_time_0) / 60.0 / epoch * (args.epoch - epoch)))
            start_time = now_time
            torch.save(
                {
                    "args": args,
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "loss": train_loss
                }, model_save_path)

    # Draw loss
    print("[Step 5] Drawing training result...")
    loss_length = len(epoch_loss_list)
    loss_x = range(1, args.epoch + 1)
    # draw loss_whole
    draw_two_dimension(
        y_lists=[epoch_loss_list],
        x_list=loss_x,
        color_list=["blue"],
        legend_list=["loss: last={0:.6f}, min={1:.6f}".format(epoch_loss_list[-1], min(epoch_loss_list))],
        line_style_list=["solid"],
        fig_title="Loss - whole ({} - Loss{})".format(args.dataset, args.loss_fn_id),
        fig_x_label="Epoch",
        fig_y_label="Loss",
        fig_size=(8, 6),
        show_flag=False,
        save_flag=True,
        save_path=figure_save_path_loss_whole
    )

    # draw loss_last_half
    draw_two_dimension(
        y_lists=[epoch_loss_list[-(loss_length // 2):]],
        x_list=loss_x[-(loss_length // 2):],
        color_list=["blue"],
        legend_list=["loss: last={0:.6f}, min={1:.6f}".format(epoch_loss_list[-1], min(epoch_loss_list))],
        line_style_list=["solid"],
        fig_title="Loss - last half ({} - Loss{})".format(args.dataset, args.loss_fn_id),
        fig_x_label="Epoch",
        fig_y_label="Loss",
        fig_size=(8, 6),
        show_flag=False,
        save_flag=True,
        save_path=figure_save_path_loss_last_half
    )

    # draw loss_last_quarter
    draw_two_dimension(
        y_lists=[epoch_loss_list[-(loss_length // 4):]],
        x_list=loss_x[-(loss_length // 4):],
        color_list=["blue"],
        legend_list=["loss: last={0:.6f}, min={1:.6f}".format(epoch_loss_list[-1], min(epoch_loss_list))],
        line_style_list=["solid"],
        fig_title="Loss - last quarter ({} - Loss{})".format(args.dataset, args.loss_fn_id),
        fig_x_label="Epoch",
        fig_y_label="Loss",
        fig_size=(8, 6),
        show_flag=False,
        save_flag=True,
        save_path=figure_save_path_loss_last_quarter
    )

    # Test
    print("[Step 6] Testing...")
    model.eval()
    train_true_list = []
    train_pred_list = []
    val_true_list = []
    val_pred_list = []
    test_true_list = []
    test_pred_list = []
    # train_dataset = MolDataset(train_logp, Y, args)
    # test_dataset = MolDataset(test_logp, Y, args)
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=2,
                                  worker_init_fn=worker_init_fn, generator=g, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=2, worker_init_fn=worker_init_fn,
                                 generator=g, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=2, worker_init_fn=worker_init_fn,
                                 generator=g, shuffle=False)

    total_error = 0
    for data in train_dataloader:
        data = data.to(args.device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        train_true_list += list(data.y.cpu().detach().numpy())
        train_pred_list += list(out.squeeze().cpu().detach().numpy())
        total_error += (out.squeeze() - data.y).abs().sum().item()
    train_loss = total_error / len(train_dataloader.dataset)
    total_error = 0

    for data in val_dataloader:
        data = data.to(args.device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        val_true_list += list(data.y.cpu().detach().numpy())
        val_pred_list += list(out.squeeze().cpu().detach().numpy())
        total_error += (out.squeeze() - data.y).abs().sum().item()
    val_loss = total_error / len(val_dataloader.dataset)

    total_error = 0
    for data in test_dataloader:
        data = data.to(args.device)
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        test_true_list += list(data.y.cpu().detach().numpy())
        test_pred_list += list(out.squeeze().cpu().detach().numpy())
        total_error += (out.squeeze() - data.y).abs().sum().item()
    test_loss = total_error / len(test_dataloader.dataset)


    train_true_list = np.asarray(train_true_list)
    train_pred_list = np.asarray(train_pred_list)
    val_true_list = np.asarray(val_true_list)
    val_pred_list = np.asarray(val_pred_list)
    test_true_list = np.asarray(test_true_list)
    test_pred_list = np.asarray(test_pred_list)

    np.save(regression_result_train_true, train_true_list)
    np.save(regression_result_train_pred, train_pred_list)
    np.save(regression_result_val_true, val_true_list)
    np.save(regression_result_val_pred, val_pred_list)
    np.save(regression_result_test_true, test_true_list)
    np.save(regression_result_test_pred, test_pred_list)

    print("[Step 7] Drawing train/val/test result...")

    r_train = compute_correlation(train_true_list, train_pred_list)
    draw_two_dimension_regression(
        x_lists=[train_true_list],
        y_lists=[train_pred_list],
        color_list=["red"],
        legend_list=["Regression: R={0:.3f}, R^2={1:.3f}".format(r_train, r_train ** 2.0)],
        line_style_list=["solid"],
        fig_title="Regression - {0} - Train - {1} points".format(args.dataset, len(train_true_list)),
        fig_x_label="Truth",
        fig_y_label="Predict",
        fig_size=(8, 6),
        show_flag=False,
        save_flag=True,
        save_path=figure_regression_train_path
    )

    r_val = compute_correlation(val_true_list, val_pred_list)
    draw_two_dimension_regression(
        x_lists=[val_true_list],
        y_lists=[val_pred_list],
        color_list=["red"],
        legend_list=["Regression: R={0:.3f}, R^2={1:.3f}".format(r_val, r_val ** 2.0)],
        line_style_list=["solid"],
        fig_title="Regression - {0} - Val - {1} points".format(args.dataset, len(val_true_list)),
        fig_x_label="Truth",
        fig_y_label="Predict",
        fig_size=(8, 6),
        show_flag=False,
        save_flag=True,
        save_path=figure_regression_val_path
    )

    r_test = compute_correlation(test_true_list, test_pred_list)
    draw_two_dimension_regression(
        x_lists=[test_true_list],
        y_lists=[test_pred_list],
        color_list=["red"],
        legend_list=["Regression: R={0:.3f}, R^2={1:.3f}".format(r_test, r_test ** 2.0)],
        line_style_list=["solid"],
        fig_title="Regression - {0} - Test - {1} points".format(args.dataset, len(test_true_list)),
        fig_x_label="Truth",
        fig_y_label="Predict",
        fig_size=(8, 6),
        show_flag=False,
        save_flag=True,
        save_path=figure_regression_test_path
    )



if __name__ == "__main__":
    run()
    # a = np.load("train/GCN_N3P/test_pred.npy")
    # b = np.load("train/GCN_N3P/test_true.npy")
    # for aa, bb in zip(a, b):
    #     print(aa, bb)
    # with open("dataset/GCN_N3P/raw/train.pickle", "rb") as f:
    #     data = pickle.load(f)
    # for i in range(len(data) - 1)[:10]:
    #
    #     print(i, np.sum(np.abs(data[i]["bond_type"].cpu().detach().numpy() - data[i + 1]["bond_type"].cpu().detach().numpy())))
    #     np.set_printoptions(threshold=np.inf)
    #     print(data[i]["bond_type"].cpu().detach().numpy())

