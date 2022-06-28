class Config:
    main_path = "./"
    dataset = "GCN_N3P"
    max_natoms = 126
    length = 1600
    train_length = int(length * 0.8)
    test_length = int(length * 0.1)
    val_length = length - train_length - test_length
    root_bmat = main_path + 'data/{}/BTMATRIXES/'.format(dataset)
    root_dmat = main_path + 'data/{}/DMATRIXES/'.format(dataset)
    root_conf = main_path + 'data/{}/CONFIGS/'.format(dataset)
    format_bmat = "BTMATRIX_{}"
    format_dmat = "DMATRIX_{}"
    format_conf = "COORD_{}"
    format_eigen = "EIGENVAL_{}"
    format_charge = "charge_data_{}"
    loss_fn_id = 1

    epoch = 100
    epoch_step = 1  # print loss every {epoch_step} epochs
    batch_size = 64
    lr = 0.01
    seed = 0


config = Config