""" Full assembly of the parts to form the complete network """

from .unet_parts import *
from torch import nn
import torch
from torch.distributions import Normal, Independent
from .res_net import resnet34, resnet18, resnet50, resnet101, resnet152
import torch.nn.functional as F


class SaveFeatures():
    features = None

    def __init__(self, m): self.hook = m.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output): self.features = output

    def remove(self): self.hook.remove()


class UnetBlock(nn.Module):
    def __init__(self, up_in, x_in, n_out):
        super().__init__()
        # super(UnetBlock, self).__init__()
        up_out = x_out = n_out // 2
        self.x_conv = nn.Conv2d(x_in, x_out, 1)
        self.tr_conv = nn.ConvTranspose2d(up_in, up_out, 2, stride=2)

        self.bn = nn.BatchNorm2d(n_out)

    def forward(self, up_p, x_p):
        up_p = self.tr_conv(up_p)
        x_p = self.x_conv(x_p)
        cat_p = torch.cat([up_p, x_p], dim=1)
        return self.bn(F.relu(cat_p))


class ResUNet(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2, pretrained=False):
        super().__init__()
        # super(ResUnet, self).__init__()
        cut, lr_cut = [8, 6]

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        elif resnet == 'resnet101':
            base_model = resnet101()
        elif resnet == 'resnet152':
            base_model = resnet152
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34, resnet50,'
                            'resnet101 and resnet152')

        layers = list(base_model(pretrained=pretrained).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers

        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4, 5, 6]]
        self.up1 = UnetBlock(512, 256, 256)
        self.up2 = UnetBlock(256, 128, 256)
        self.up3 = UnetBlock(256, 64, 256)
        self.up4 = UnetBlock(256, 64, 256)

        self.up5 = nn.ConvTranspose2d(256, self.num_classes, 2, stride=2)

    def forward(self, x):
        x = F.relu(self.rn(x))              # x = [b_size, 2048, 8, 8]

        '''~~~ 0: Decoder ~~~'''
        x = self.up1(x, self.sfs[3].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        fea = x
        output = self.up5(x)
        '''~~~ 0: ENDs ~~~'''

        return output

    def close(self):
        for sf in self.sfs: sf.remove()


class UNet_dist(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2, pretrained=False):
        super().__init__()
        cut, lr_cut = [8, 6]

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        elif resnet == 'resnet101':
            base_model = resnet101()
        elif resnet == 'resnet152':
            base_model = resnet152
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34, resnet50,'
                            'resnet101 and resnet152')

        layers = list(base_model(pretrained=pretrained).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers

        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4, 5, 6]]
        self.up1 = UnetBlock(512, 256, 256)
        self.up2 = UnetBlock(256, 128, 256)
        self.up3 = UnetBlock(256, 64, 256)
        self.up4 = UnetBlock(256, 64, 256)

        self.up5 = nn.ConvTranspose2d(256, self.num_classes*14, 2, stride=2)

    def forward(self, x, training=True):
        rater_num = 6
        x = F.relu(self.rn(x))  # x = [b_size, 2048, 8, 8]
        x = self.up1(x, self.sfs[3].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        output = self.up5(x)

        global_mu = output[:, 0:2, :, :]
        global_sigma = output[:, 2:4, :, :]
        global_sigma = torch.abs(global_sigma)

        mu_residual_list = []
        sigma_rater_list = []
        for i in range(rater_num):
            start_index = 3 + 4 * i
            end_index = start_index + 2
            mu_residual_temp = output[:, start_index:end_index, :, :]
            sigma_rater_temp = output[:, end_index: end_index+2, :, :]
            sigma_rater_temp = torch.abs(sigma_rater_temp)
            mu_residual_list.append(mu_residual_temp)
            sigma_rater_list.append(sigma_rater_temp)

        mu_rater_list = []
        for i in range(rater_num):
            mu_rater_temp = global_mu + mu_residual_list[i]
            mu_rater_list.append(mu_rater_temp)

        global_dist = Independent(Normal(loc=global_mu, scale=global_sigma), 1)
        dist_rater_list = []
        for i in range(rater_num):
            dist_rater_temp = Independent(Normal(loc=mu_rater_list[i], scale=sigma_rater_list[i]), 1)
            dist_rater_list.append(dist_rater_temp)

        if training:
            global_samples = global_dist.rsample([6])
            rater_samples_list = []
            for i in range(rater_num):
                sample_temp = dist_rater_list[i].rsample()
                sample_temp = torch.sigmoid(sample_temp)
                rater_samples_list.append(sample_temp)
        else:
            global_samples = global_dist.sample([50])
            rater_samples_list = []

        global_samples = torch.sigmoid(global_samples)
        return global_mu, global_sigma, mu_rater_list, sigma_rater_list, global_samples, rater_samples_list

    def close(self):
        for sf in self.sfs: sf.remove()


class PADL(nn.Module):

    def __init__(self, resnet='resnet34', num_classes=2, rater_num=6, pretrained=False):
        super().__init__()
        cut, lr_cut = [8, 6]
        self.rater_num = rater_num

        if resnet == 'resnet34':
            base_model = resnet34
        elif resnet == 'resnet18':
            base_model = resnet18
        elif resnet == 'resnet50':
            base_model = resnet50
        elif resnet == 'resnet101':
            base_model = resnet101()
        elif resnet == 'resnet152':
            base_model = resnet152
        else:
            raise Exception('The Resnet Model only accept resnet18, resnet34, resnet50,'
                            'resnet101 and resnet152')

        layers = list(base_model(pretrained=pretrained).children())[:cut]
        base_layers = nn.Sequential(*layers)
        self.rn = base_layers

        self.num_classes = num_classes
        self.sfs = [SaveFeatures(base_layers[i]) for i in [2, 4, 5, 6]]
        self.up1 = UnetBlock(512, 256, 256)
        self.up2 = UnetBlock(256, 128, 256)
        self.up3 = UnetBlock(256, 64, 256)
        self.up4 = UnetBlock(256, 64, 256)

        self.up5 = nn.ConvTranspose2d(256, 32, 2, stride=2)
        self.bnout = nn.BatchNorm2d(32)

        self.global_mu_head = nn.Conv2d(32, self.num_classes, 1)
        self.global_sigma_head_reduction = nn.Sequential(
                    nn.Conv2d(32, self.num_classes, 1),
                    nn.BatchNorm2d(self.num_classes),
                    nn.ReLU(),
                )
        self.global_sigma_head_output = nn.Conv2d(self.num_classes * 2, self.num_classes, 1)

        self.rater_residual_heads_reduction = list()
        self.rater_residual_heads_output = list()
        self.rater_sigma_heads_reduction = list()
        self.rater_sigma_heads_output = list()
        for i in range(self.rater_num):
            self.rater_residual_heads_reduction.append(
                nn.Sequential(
                    nn.Conv2d(32, self.num_classes, 1),
                    nn.BatchNorm2d(self.num_classes),
                    nn.ReLU(),
                )
            )
            self.rater_residual_heads_output.append(
                nn.Conv2d(self.num_classes * 2, self.num_classes, 1)
            )

            self.rater_sigma_heads_reduction.append(
                nn.Sequential(
                    nn.Conv2d(32, self.num_classes, 1),
                    nn.BatchNorm2d(self.num_classes),
                    nn.ReLU(),
                )
            )
            self.rater_sigma_heads_output.append(
                nn.Conv2d(self.num_classes * 2, self.num_classes, 1)
            )

        self.rater_residual_heads_reduction = nn.ModuleList(self.rater_residual_heads_reduction)
        self.rater_residual_heads_output = nn.ModuleList(self.rater_residual_heads_output)
        self.rater_sigma_heads_reduction = nn.ModuleList(self.rater_sigma_heads_reduction)
        self.rater_sigma_heads_output = nn.ModuleList(self.rater_sigma_heads_output)

    def forward(self, x, training=True):
        x = F.relu(self.rn(x))  # x = [b_size, 2048, 8, 8]
        x = self.up1(x, self.sfs[3].features)
        x = self.up2(x, self.sfs[2].features)
        x = self.up3(x, self.sfs[1].features)
        x = self.up4(x, self.sfs[0].features)
        x = self.up5(x)

        head_input = F.relu(self.bnout(x))

        global_mu = self.global_mu_head(head_input)
        global_mu_sigmoid = torch.sigmoid(global_mu)
        global_entropy_map = -global_mu_sigmoid * torch.log2(global_mu_sigmoid+1e-6) - \
                             (1 - global_mu_sigmoid) * torch.log2(1 - global_mu_sigmoid+1e-6)
        global_entropy_map = global_entropy_map.detach()
        global_sigma_reduction = self.global_sigma_head_reduction(head_input)
        global_sigma_input = (1 + global_entropy_map) * global_sigma_reduction
        global_sigma_input = torch.cat([global_sigma_input, global_mu], dim=1)
        global_sigma = self.global_sigma_head_output(global_sigma_input)
        global_sigma = torch.abs(global_sigma)

        rater_residual_reduction_list = [self.rater_residual_heads_reduction[i](head_input) for i in range(self.rater_num)]
        rater_residual_input = [(1 + global_entropy_map) * rater_residual_reduction_list[i] for i in range(self.rater_num)]
        rater_residual_input = [torch.cat([rater_residual_input[i], global_mu], dim=1) for i in range(self.rater_num)]
        rater_residual = [self.rater_residual_heads_output[i](rater_residual_input[i]) for i in range(self.rater_num)]

        rater_mu = [rater_residual[i] + global_mu for i in range(self.rater_num)]

        rater_sigma_reduction_list = [self.rater_sigma_heads_reduction[i](head_input) for i in range(self.rater_num)]
        rater_sigma_input = [(1 + global_entropy_map) * rater_sigma_reduction_list[i] for i in range(self.rater_num)]
        rater_sigma_input = [torch.cat([rater_sigma_input[i], rater_mu[i]], dim=1) for i in range(self.rater_num)]
        rater_sigma = [self.rater_sigma_heads_output[i](rater_sigma_input[i]) for i in range(self.rater_num)]

        rater_sigmas = torch.stack(rater_sigma, dim=0)
        rater_sigmas = torch.abs(rater_sigmas)
        rater_mus = torch.stack(rater_mu, dim=0)
        rater_residuals = torch.stack(rater_residual, dim=0)
        rater_dists = list()
        for i in range(self.rater_num):
            rater_dists.append(Independent(Normal(loc=rater_mus[i], scale=rater_sigmas[i], validate_args=False), 1))
        global_dist = Independent(Normal(loc=global_mu, scale=global_sigma, validate_args=False), 1)

        if training:
            rater_samples = [dist.rsample() for dist in rater_dists]
            rater_samples = torch.stack(rater_samples, dim=0)
            global_samples = global_dist.rsample([self.rater_num])
        else:
            rater_samples = [dist.sample() for dist in rater_dists]
            rater_samples = torch.stack(rater_samples, dim=0)
            global_samples = global_dist.sample([self.rater_num])

        return global_mu, rater_mus, global_sigma, rater_sigmas, rater_samples, global_samples, rater_residuals

    def close(self):
        for sf in self.sfs: sf.remove()

