import time
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torchvision as tv
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from utils import *
from model import *
from config import *


class ImgDataset(Dataset):
    def __init__(self, root, aligned=False):
        self.aligned = aligned
        self.files_A = sorted(glob(os.path.join(root, "X/*.*")))
        self.files_B = sorted(glob(os.path.join(root, "Y/*.*")))

    def __getitem__(self, index):
        # image_A = Image.open(self.files_A[index % len(self.files_A)])
        image_A = cv2.imread(self.files_A[index % len(self.files_A)])

        if self.aligned:
            image_B = cv2.imread(self.files_B[index % len(self.files_B)])
        else:
            image_B = cv2.imread(self.files_B[random.randint(0, len(self.files_B)-1)])

        item_A = tv.transforms.ToTensor()(image_A)
        item_A = tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(item_A)

        image_B = image_B[:, :, 1] > 0.5
        image_B = image_B.astype(np.float32)
        item_B = tv.transforms.ToTensor()(image_B)

        return {"A": item_A, "B": item_B}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))


def train(**kwargs):

    for k_, v_ in kwargs.items():
        setattr(opt, k_, v_)

    device = torch.device("cuda:0")
    dataset = ImgDataset(opt.train_path, aligned=True)

    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_sz,
        drop_last=True,
        num_workers=2
    )

    model = UNet_Res_Multi(64, 3, 1)
    model.to(device)

    if opt.load_model:
        model.load_state_dict(torch.load(opt.load_model, map_location=device))
    else:
        model.init_weights()

    optimizer = torch.optim.Adam(
        model.parameters(),
        betas=(opt.beta1, opt.beta2),
        lr=opt.lr
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=1,
        T_mult=2
    )
    criterion = nn.BCEWithLogitsLoss().to(device)

    cur_time = time.strftime('%m-%d %H:%M:%S', time.localtime(time.time()))
    print('start', cur_time)

    for epoch in range(opt.max_epoch):
        for i, batch in enumerate(dataloader):
            X = batch['A'].to(device)
            Y = batch['B'].to(device)
            predict_Y = model(X)

            optimizer.zero_grad()
            loss = criterion(predict_Y, Y)
            loss.backward()
            optimizer.step()

        loss_record.append(loss.item())
        scheduler.step()

        if (epoch + 1) % opt.save_freq == 0:
            torch.save(model.state_dict(), '%s/model_%s.pth' % (opt.model_path, epoch+1))

        cur_time = time.strftime('%m-%d %H:%M:%S', time.localtime(time.time()))
        print('epoch', epoch, cur_time)


if __name__ == '__main__':
    train()
