# -*- coding: utf-8 -*-


class Config(object):
    # path
    train_path = "./data/traning"   # where to load training data
    test_path = "./data/test"       # where to read testing data
    dst_path = "./data/results"     # where to output results
    model_path = "./checkpoints"    # where to save (intermediate) models
    load_model = None               # which file to load when testing

    # training parameters
    batch_sz = 6        # batch size
    n_gf = 64           # number of filters at the highest level
    lr = 1e-3           # learning rate
    beta1 = 0.9         # parameter 1 for Adam optimizer
    beta2 = 0.999       # parameter 2 for Adam optimizer
    max_epoch = 40      # max epoch number
    save_freq = 5       # saving frequency


opt = Config()
