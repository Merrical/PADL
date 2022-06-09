import os
import torch
import numpy as np
from apex import amp
import nibabel as nib
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from utils.metrics import get_soft_dice
from utils.functions import adjust_learning_rate


def train_riga_padl(args, log_folder, checkpoint_folder, visualization_folder, metrics_folder, model, optimizer,
                    loss_func, train_set, test_set):

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.cuda()
    model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    writer = SummaryWriter(log_dir=log_folder)

    for this_epoch in range(args.num_epoch):
        print("---------------------")
        print(this_epoch)
        model.train()
        train_loss = 0.0
        train_soft_dice_cup = 0.0
        train_soft_dice_disc = 0.0

        for step, data in enumerate(train_loader):
            imgs = data['image'].to(dtype=torch.float32).cuda()  # torch.Size([B, 3, 256, 256])
            mask = data['mask']  # type = list; len = 6; item_type = tensor; item_size = torch.Size([B, 2, 256, 256])
            # cup_mask = mask[i][:,1,:,:], size = [B,256,256]
            # disc_mask = mask[i][:,0,:,:], size = [B,256,256]

            # get majority voting mask
            mask_major_vote = (mask[0] + mask[1] + mask[2] + mask[3] + mask[4] + mask[5]) / 6.0
            mask_major_vote = mask_major_vote.to(dtype=torch.float32).cuda()

            global_mu, rater_mus, global_sigma, rater_sigmas, rater_samples, global_samples, rater_residuals = model(imgs, training=True)

            loss_global = 0.0
            loss_rater = 0.0
            for i in range(6):
                rater_mask = mask[i].cuda()
                loss_global += loss_func(global_samples[i], rater_mask)
                loss_rater += loss_func(rater_samples[i], rater_mask)

            loss = loss_global + loss_rater

            optimizer.zero_grad()
            adjust_learning_rate(optimizer, this_epoch, args.learning_rate, args.num_epoch, args.power)
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss = train_loss + loss * imgs.size(0)
            global_mu = torch.sigmoid(global_mu)
            train_soft_dice_cup = train_soft_dice_cup + get_soft_dice(outputs=global_mu[:, 1, :, :].cpu(), masks=mask_major_vote[:, 1, :, :].cpu()) * imgs.size(0)
            train_soft_dice_disc = train_soft_dice_disc + get_soft_dice(outputs=global_mu[:, 0, :, :].cpu(), masks=mask_major_vote[:, 0, :, :].cpu()) * imgs.size(0)

        writer.add_scalar("Loss/train", train_loss/(train_set.__len__()), this_epoch)
        writer.add_scalar("Soft_Dice/train_cup", train_soft_dice_cup/(train_set.__len__()), this_epoch)
        writer.add_scalar("Soft_Dice/train_disc", train_soft_dice_disc/(train_set.__len__()), this_epoch)

    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'amp': amp.state_dict()
    }
    torch.save(checkpoint, checkpoint_folder+'/amp_checkpoint.pt')

    test_riga_padl(args, visualization_folder, metrics_folder, model, test_set)


def test_visualization(numpy_arr, folder, nii_name):
    """
    :param numpy_arr: 3D
    :return:
    """
    new_image = nib.Nifti1Image(numpy_arr, np.eye(4))
    new_image.set_data_dtype(np.float32)
    nib.save(new_image, folder+'/'+nii_name)


def test_riga_padl(args, visualization_folder, metrics_folder, model, test_set):

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.cuda()
    test_loader = DataLoader(test_set, batch_size=95, shuffle=False, num_workers=4, pin_memory=True)

    metrix_file = metrics_folder + "/dice.txt"
    file_handle = open(metrix_file, 'a')
    file_handle.write('testing data size: %d \n' % (test_set.__len__()))
    file_handle.close()

    model.eval()
    test_soft_dice_cup = 0.0
    test_soft_dice_disc = 0.0

    test_soft_dice_disc_raters = [0.0]*6
    test_soft_dice_cup_raters = [0.0]*6

    for step, data in enumerate(test_loader):
        with torch.no_grad():
            imgs = data['image'].to(dtype=torch.float32).cuda()  # torch.Size([B, 3, 256, 256])
            mask = data['mask']  # type = list; len = 6; item_type = tensor; item_size = torch.Size([B, 2, 256, 256])

            mask_major_vote = (mask[0]+mask[1]+mask[2]+mask[3]+mask[4]+mask[5])/6.0
            global_mu, rater_mus, global_sigma, rater_sigmas, rater_samples, global_samples, rater_residuals = model(imgs, training=False)

            rater_mus_sigmoid = torch.sigmoid(rater_mus)
            global_mu_sigmoid = torch.sigmoid(global_mu)

            test_soft_dice_cup = test_soft_dice_cup + get_soft_dice(outputs=global_mu_sigmoid[:, 1, :, :].cpu(), masks=mask_major_vote[:, 1, :, :].cpu()) * imgs.size(0)
            test_soft_dice_disc = test_soft_dice_disc + get_soft_dice(outputs=global_mu_sigmoid[:, 0, :, :].cpu(), masks=mask_major_vote[:, 0, :, :].cpu()) * imgs.size(0)

            test_soft_dice_disc_raters = [test_soft_dice_disc_raters[i] + get_soft_dice(outputs=rater_mus_sigmoid[i][:, 0, :, :].cpu(), masks=mask[i][:, 0, :, :].cpu()) * imgs.size(0) for i in range(6)]
            test_soft_dice_cup_raters = [test_soft_dice_cup_raters[i] + get_soft_dice(outputs=rater_mus_sigmoid[i][:, 1, :, :].cpu(), masks=mask[i][:, 1, :, :].cpu()) * imgs.size(0) for i in range(6)]

    file_handle = open(metrix_file, 'a')
    file_handle.write("Mean Voting: ({}, {})\n".format(round(test_soft_dice_disc / test_set.__len__() * 100, 2),
                                                       round(test_soft_dice_cup / test_set.__len__() * 100, 2)))
    file_handle.write("Average: ({}, {})\n".format(round(np.mean(test_soft_dice_disc_raters) / test_set.__len__() * 100, 2),
                                                   round(np.mean(test_soft_dice_cup_raters) / test_set.__len__() * 100, 2)))

    for i in range(6):
        file_handle.write(
            "rater{}: ({}, {})\n".format(i + 1, round(test_soft_dice_disc_raters[i] / test_set.__len__() * 100, 2),
                                         round(test_soft_dice_cup_raters[i] / test_set.__len__() * 100, 2)))
