import numpy as np
import torch
import torch.nn as nn
from .base_model import BaseModel
from . import networks
from .patchnce import PatchNCELoss
import util.util as util
from captum.attr import IntegratedGradients
import cv2
import warnings
from enum import Enum


class attributionModel(nn.Module):
    def __init__(self, base_model):
        super(attributionModel, self).__init__()
        self.base_model = base_model  # 기존 모델
    def forward(self, x):
        x = self.base_model(x)  
        return torch.sum(x, dim=(2, 3)) #For gradcam
    
class VisualizeSign(Enum):
    positive = 1
    absolute_value = 2
    negative = 3
    all = 4
    

class CUTModel(BaseModel):
    """ This class implements CUT and FastCUT model, described in the paper
    Contrastive Learning for Unpaired Image-to-Image Translation
    Taesung Park, Alexei A. Efros, Richard Zhang, Jun-Yan Zhu
    ECCV, 2020

    The code borrows heavily from the PyTorch implementation of CycleGAN
    https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for CUT model
        """
        parser.add_argument('--CUT_mode', type=str, default="CUT", choices='(CUT, cut, FastCUT, fastcut)')

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss：GAN(G(X))')
        parser.add_argument('--lambda_NCE', type=float, default=1.0, help='weight for NCE loss: NCE(G(X), X)')
        parser.add_argument('--nce_idt', type=util.str2bool, nargs='?', const=True, default=False, help='use NCE loss for identity mapping: NCE(G(Y), Y))')
        parser.add_argument('--nce_layers', type=str, default='0,4,8,12,16', help='compute NCE loss on which layers')
        parser.add_argument('--nce_includes_all_negatives_from_minibatch',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help='(used for single image translation) If True, include the negatives from the other samples of the minibatch when computing the contrastive loss. Please see models/patchnce.py for more details.')
        parser.add_argument('--netF', type=str, default='mlp_sample', choices=['sample', 'reshape', 'mlp_sample'], help='how to downsample the feature map')
        parser.add_argument('--netF_nc', type=int, default=256)
        parser.add_argument('--nce_T', type=float, default=0.07, help='temperature for NCE loss')
        parser.add_argument('--num_patches', type=int, default=256, help='number of patches per layer')
        parser.add_argument('--flip_equivariance',
                            type=util.str2bool, nargs='?', const=True, default=False,
                            help="Enforce flip-equivariance as additional regularization. It's used by FastCUT, but not CUT")

        parser.set_defaults(pool_size=0)  # no image pooling

        opt, _ = parser.parse_known_args()

        # Set default parameters for CUT and FastCUT
        if opt.CUT_mode.lower() == "cut":
            parser.set_defaults(nce_idt=True, lambda_NCE=1.0)
        elif opt.CUT_mode.lower() == "fastcut":
            parser.set_defaults(
                nce_idt=False, lambda_NCE=10.0, flip_equivariance=True,
                n_epochs=150, n_epochs_decay=50
            )
        else:
            raise ValueError(opt.CUT_mode)

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G', 'NCE']
        if self.opt.isX:
            self.visual_names = ['real_A', 'fake_B', 'real_B','real_A_sig','fake_B_sig','real_B_sig', 'fake_B_cam_IntegratedGradients','real_B_cam_IntegratedGradients']
        else:
            self.visual_names = ['real_A', 'fake_B', 'real_B']
        self.nce_layers = [int(i) for i in self.opt.nce_layers.split(',')]

        if opt.nce_idt and self.isTrain:
            self.loss_names += ['NCE_Y']
            self.visual_names += ['idt_B']

        if self.isTrain:
            self.model_names = ['G', 'F', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)
        self.netF = networks.define_F(opt.input_nc, opt.netF, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

        if self.opt.isX:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            
        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD, opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)

            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionNCE = []

            for nce_layer in self.nce_layers:
                self.criterionNCE.append(PatchNCELoss(opt).to(self.device))

            self.criterionIdt = torch.nn.L1Loss().to(self.device)
            self.criterionContent = torch.nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def data_dependent_initialize(self, data):
        """
        The feature network netF is defined in terms of the shape of the intermediate, extracted
        features of the encoder portion of netG. Because of this, the weights of netF are
        initialized at the first feedforward pass with some input images.
        Please also see PatchSampleF.create_mlp(), which is called at the first forward() call.
        """
        bs_per_gpu = data["A"].size(0) // max(len(self.opt.gpu_ids), 1)
        self.set_input(data)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]
        self.forward()                     # compute fake images: G(A)
        if self.opt.isTrain:
            self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate graidents for G
            if self.opt.lambda_NCE > 0.0:
                self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.loss_D = self.compute_D_loss()
        self.loss_D.backward()
        self.optimizer_D.step()

        # update G
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
        if self.opt.netF == 'mlp_sample':
            self.optimizer_F.step()

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.
        Parameters:
            input (dict): include the data itself and its metadata information.
        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.real = torch.cat((self.real_A, self.real_B), dim=0) if self.opt.nce_idt and self.opt.isTrain else self.real_A
        if self.opt.flip_equivariance:
            self.flipped_for_equivariance = self.opt.isTrain and (np.random.random() < 0.5)
            if self.flipped_for_equivariance:
                self.real = torch.flip(self.real, [3])

        self.fake = self.netG(self.real)
        self.fake_B = self.fake[:self.real_A.size(0)]
        if self.opt.nce_idt:
            self.idt_B = self.fake[self.real_A.size(0):]

        if self.opt.isX:
            # Explainability: 1. Saliency mask
            threshold_A = 90    #PA90
            threshold_B = 170   #HE170
            real_A_mean = torch.mean(self.real_A,dim=1,keepdim=True)
            fake_B_mean = torch.mean(self.fake_B,dim=1,keepdim=True)
            real_A_normal = (real_A_mean - (threshold_A/127.5-1))*100
            fake_B_normal = (fake_B_mean - (threshold_B/127.5-1))*100
            self.real_A_sig = 1 - torch.sigmoid(real_A_normal)
            self.fake_B_sig = 1 - torch.sigmoid(fake_B_normal)
            real_B_mean = torch.mean(self.real_B,dim=1,keepdim=True)
            real_B_normal = (real_B_mean - (threshold_B/127.5-1))*100
            self.real_B_sig = 1 - torch.sigmoid(real_B_normal)

            # Explainability: 2. Integrated Gradients
            attribution_model = attributionModel(self.netD).eval()
            
            ig = IntegratedGradients(attribution_model)
            attribution = ig.attribute(self.fake_B, internal_batch_size=1, n_steps=50)
            norm_attr = self.normalize_attr(attribution[0].cpu().permute(1,2,0).detach().numpy(), sign="all")
            norm_attr_jet = cv2.applyColorMap(((norm_attr+1)*127.5).astype(np.uint8), cv2.COLORMAP_JET)
            self.fake_B_cam_IntegratedGradients = torch.tensor(norm_attr_jet.astype(np.float32)/127.5-1).permute(2, 0, 1).unsqueeze(0)  #(512, 512,3) -> tensor(1,3,512,512)

            attribution = ig.attribute(self.real_B, internal_batch_size=1, n_steps=50)
            norm_attr = self.normalize_attr(attribution[0].cpu().permute(1,2,0).detach().numpy(), sign="all")
            norm_attr_jet = cv2.applyColorMap(((norm_attr+1)*127.5).astype(np.uint8), cv2.COLORMAP_JET)
            self.real_B_cam_IntegratedGradients = torch.tensor(norm_attr_jet.astype(np.float32)/127.5-1).permute(2, 0, 1).unsqueeze(0)  #(512, 512,3) -> tensor(1,3,512,512)

    def compute_D_loss(self):
        """Calculate GAN loss for the discriminator"""
        fake = self.fake_B.detach()
        # Fake; stop backprop to the generator by detaching fake_B
        pred_fake = self.netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False).mean()
        # Real
        self.pred_real = self.netD(self.real_B)
        loss_D_real = self.criterionGAN(self.pred_real, True)
        self.loss_D_real = loss_D_real.mean()

        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate GAN and NCE loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fake the discriminator
        if self.opt.lambda_GAN > 0.0:
            pred_fake = self.netD(fake)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        if self.opt.lambda_NCE > 0.0:
            self.loss_NCE = self.calculate_NCE_loss(self.real_A, self.fake_B)
        else:
            self.loss_NCE, self.loss_NCE_bd = 0.0, 0.0

        if self.opt.nce_idt and self.opt.lambda_NCE > 0.0:
            self.loss_NCE_Y = self.calculate_NCE_loss(self.real_B, self.idt_B)
            loss_NCE_both = (self.loss_NCE + self.loss_NCE_Y) * 0.5
        else:
            loss_NCE_both = self.loss_NCE

        if self.opt.isX:
            # Saliency loss
            self.loss_content = self.criterionContent(self.real_A_sig, self.fake_B_sig)
            # combined loss and calculate gradients
            self.loss_G = self.loss_G_GAN + loss_NCE_both + self.loss_content
        else:
            self.loss_G = self.loss_G_GAN + loss_NCE_both 
        return self.loss_G

    def calculate_NCE_loss(self, src, tgt):
        n_layers = len(self.nce_layers)
        feat_q = self.netG(tgt, self.nce_layers, encode_only=True)

        if self.opt.flip_equivariance and self.flipped_for_equivariance:
            feat_q = [torch.flip(fq, [3]) for fq in feat_q]

        feat_k = self.netG(src, self.nce_layers, encode_only=True)
        feat_k_pool, sample_ids = self.netF(feat_k, self.opt.num_patches, None)
        feat_q_pool, _ = self.netF(feat_q, self.opt.num_patches, sample_ids)

        total_nce_loss = 0.0
        for f_q, f_k, crit, nce_layer in zip(feat_q_pool, feat_k_pool, self.criterionNCE, self.nce_layers):
            loss = crit(f_q, f_k) * self.opt.lambda_NCE
            total_nce_loss += loss.mean()

        return total_nce_loss / n_layers

    # CAPTUM code for normalization
    def normalize_attr(self, attr, sign, outlier_perc = 2, reduction_axis = 2):
        
    
        def _normalize_scale(attr, scale_factor):
            #if scale_factor == 0: scale_factor = 1e-5
            #assert scale_factor != 0, "Cannot normalize by scale factor = 0"
            if abs(scale_factor) < 1e-5:
                warnings.warn(
                    "Attempting to normalize by value approximately 0, visualized results"
                    "may be misleading. This likely means that attribution values are all"
                    "close to 0."
                )
            attr_norm = attr / scale_factor
            return np.clip(attr_norm, -1, 1)

        def _cumulative_sum_threshold(values, percentile):
            # given values should be non-negative
            assert percentile >= 0 and percentile <= 100, (
                "Percentile for thresholding must be " "between 0 and 100 inclusive."
            )
            sorted_vals = np.sort(values.flatten())
            cum_sums = np.cumsum(sorted_vals)
            threshold_id = np.where(cum_sums >= cum_sums[-1] * 0.01 * percentile)[0][0]
            return sorted_vals[threshold_id]
        
        attr_combined = attr
        if reduction_axis is not None:
            attr_combined = np.sum(attr, axis=reduction_axis)

        # Choose appropriate signed values and rescale, removing given outlier percentage.
        if VisualizeSign[sign] == VisualizeSign.all:
            threshold = _cumulative_sum_threshold(np.abs(attr_combined), 100 - outlier_perc)
        elif VisualizeSign[sign] == VisualizeSign.positive:
            attr_combined = (attr_combined > 0) * attr_combined
            threshold = _cumulative_sum_threshold(attr_combined, 100 - outlier_perc)
        elif VisualizeSign[sign] == VisualizeSign.negative:
            attr_combined = (attr_combined < 0) * attr_combined
            threshold = -1 * _cumulative_sum_threshold(
                np.abs(attr_combined), 100 - outlier_perc
            )
        elif VisualizeSign[sign] == VisualizeSign.absolute_value:
            attr_combined = np.abs(attr_combined)
            threshold = _cumulative_sum_threshold(attr_combined, 100 - outlier_perc)
        else:
            raise AssertionError("Visualize Sign type is not valid.")
        return _normalize_scale(attr_combined, threshold)