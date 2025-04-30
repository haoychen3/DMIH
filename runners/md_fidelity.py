import cv2
from glob import glob
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr
from pytorch_msssim import ssim

import numpy as np
from DISTS_pytorch import DISTS
from torchvision import models,transforms
from PIL import Image
import torch.nn.functional as F


def cal_md_fidelity(ref_sample_pth='', test_sample_pth=''):

    ref_images = glob(ref_sample_pth+"/*.png")
    test_images = glob(test_sample_pth+"/*.png")

    assert len(ref_images) == len(test_images)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LPIPSvgg().to(device)

    ssim_avg = []
    psnr_list = []
    lpips_list = []
    D = DISTS()
    dists_list = []
    for ref, test in zip(sorted(ref_images), sorted(test_images)):
        ref_img = cv2.imread(str(ref), cv2.IMREAD_COLOR)
        test_img = cv2.imread(str(test), cv2.IMREAD_COLOR)

        psnr_z = psnr(ref_img, test_img)
        psnr_list.append(psnr_z)

        ref_img_ = torch.tensor(ref_img).unsqueeze(0).unsqueeze(0).float()
        test_img_ = torch.tensor(test_img).unsqueeze(0).unsqueeze(0).float()

        ssim_z = ssim(test_img_, ref_img_, data_range=255, size_average=False)
        ssim_avg.append(ssim_z)

        X = []
        Y = []
        test_img_d = np.moveaxis(test_img, -1, 0)
        X.append(test_img_d)
        ref_img_d = np.moveaxis(ref_img, -1, 0)
        Y.append(ref_img_d)
        X = np.array(X)
        Y = np.array(Y)
        X = torch.tensor(X, dtype=torch.float)
        Y = torch.tensor(Y, dtype=torch.float)
        # calculate DISTS between X, Y (a batch of RGB images, data range: 0~1)
        # X: (N,C,H,W)
        # Y: (N,C,H,W)
        dists_value = D(X, Y)
        dists_list.append(dists_value.mean())

        ref_ = prepare_image(Image.open(ref).convert("RGB")).to(device)
        dist_ = prepare_image(Image.open(test).convert("RGB")).to(device)

        score = model(ref_, dist_, as_loss=False)
        lpips_list.append(score.item())

    return sum(psnr_list) / len(psnr_list), sum(ssim_avg) / len(ssim_avg), sum(lpips_list) / len(lpips_list), sum(dists_list) / len(dists_list)


def prepare_image(image, resize = False, repeatNum = 1):
    if resize and min(image.size)>256:
        image = transforms.functional.resize(image,256)
    image = transforms.ToTensor()(image)
    return image.unsqueeze(0).repeat(repeatNum,1,1,1)


class LPIPSvgg(torch.nn.Module):
    def __init__(self, channels=3):
        # Refer to https://github.com/richzhang/PerceptualSimilarity

        assert channels == 3
        super(LPIPSvgg, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=True).features
        self.stage1 = torch.nn.Sequential()
        self.stage2 = torch.nn.Sequential()
        self.stage3 = torch.nn.Sequential()
        self.stage4 = torch.nn.Sequential()
        self.stage5 = torch.nn.Sequential()
        for x in range(0, 4):
            self.stage1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.stage2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.stage3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.stage4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.stage5.add_module(str(x), vgg_pretrained_features[x])

        for param in self.parameters():
            param.requires_grad = False

        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1))

        self.chns = [64, 128, 256, 512, 512]
        self.weights = torch.load('runners/LPIPSvgg.pt')
        self.weights = list(self.weights.items())

    def forward_once(self, x):
        h = (x - self.mean) / self.std
        h = self.stage1(h)
        h_relu1_2 = h
        h = self.stage2(h)
        h_relu2_2 = h
        h = self.stage3(h)
        h_relu3_3 = h
        h = self.stage4(h)
        h_relu4_3 = h
        h = self.stage5(h)
        h_relu5_3 = h
        outs = [h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3]
        for k in range(len(outs)):
            outs[k] = F.normalize(outs[k])
        return outs

    def forward(self, x, y, as_loss=True):
        assert x.shape == y.shape
        if as_loss:
            feats0 = self.forward_once(x)
            feats1 = self.forward_once(y)
        else:
            with torch.no_grad():
                feats0 = self.forward_once(x)
                feats1 = self.forward_once(y)
        score = 0
        for k in range(len(self.chns)):
            score = score + (self.weights[k][1] * (feats0[k] - feats1[k]) ** 2).mean([2, 3]).sum(1)
        if as_loss:
            return score.mean()
        else:
            return score



