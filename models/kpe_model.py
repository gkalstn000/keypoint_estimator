import torch
import models.networks as networks
import util.util as util
import numpy as np
torch.autograd.set_detect_anomaly(True)
class KPEModel(torch.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.FloatTensor = torch.cuda.FloatTensor if self.use_gpu() \
            else torch.FloatTensor
        self.ByteTensor = torch.cuda.ByteTensor if self.use_gpu() \
            else torch.ByteTensor

        self.netG, self.netD = self.initialize_networks(opt)

        # set loss functions
        if opt.isTrain:
            self.criterionMSE = torch.nn.MSELoss() # for keypoint loss
            self.criterionBCE = torch.nn.BCELoss() # for occlusion loss
            self.criterionGAN = networks.GANLoss(
                opt.gan_mode, tensor=self.FloatTensor, opt=self.opt)
            self.criterionFeat = torch.nn.L1Loss()

        self.max_point_tensor = torch.Tensor([opt.max_height-1, opt.max_width]).cuda()

        self.heatmap = {}

    def forward(self, data, mode):
        source, target, occlusion_label = self.preprocess_input(data)

        if mode == 'generator' :
            g_losses, g_map, fake_keypoint = self.compute_generator_loss(source, target, occlusion_label)
            return g_losses, g_map, fake_keypoint
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(
                source, target, occlusion_label)
            return d_loss
        elif mode == 'inference' :
            with torch.no_grad() :
                fake_keypoint, occlusion_pred = self.generate_fake(source)
                self.transform_keypoints_to_heatmap(source, target, fake_keypoint, occlusion_pred)
                return fake_keypoint, occlusion_pred
        else:
            raise ValueError("|mode| is invalid")
    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        if opt.isTrain:
            D_params = list(self.netD.parameters())
        beta1, beta2 = opt.beta1, opt.beta2
        if opt.no_TTUR:
            G_lr, D_lr = opt.lr, opt.lr
        else:
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = torch.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netD = networks.define_D(opt) if opt.isTrain else None

        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)
        return netG, netD

    def preprocess_input(self, data):
        # preprocess the input, such as moving the tensors to GPUs and
        # |data|: dictionary of the input data
        if self.use_gpu() :
            data['source_keypoint'] = data['source_keypoint'].float().cuda()
            data['target_keypoint'] = data['target_keypoint'].float().cuda()
            data['occlusion_label'] = data['occlusion_label'].float().cuda()

        return data['source_keypoint'], data['target_keypoint'], data['occlusion_label']

    def compute_generator_loss(self, source, target, occlusion_label):
        G_losses = {}
        G_map = {}
        fake_keypoint, occlusion_pred = self.generate_fake(source)
        # Calculate losses
        G_losses['MSE_Loss'] = self.criterionMSE(fake_keypoint[~target.isnan()], target[~target.isnan()]) * self.opt.lambda_mse
        G_losses['BCE_loss'] = self.criterionBCE(occlusion_pred.squeeze(), occlusion_label.float()) * self.opt.lambda_bce
        # ========= Keypoint to map =========
        if self.opt.use_D:
            self.transform_keypoints_to_heatmap(source, target, fake_keypoint, occlusion_label)
            pred_fake, pred_real = self.discriminate(self.heatmap['fake_color_map'], self.heatmap['tgt_color_map'])

            G_losses['GAN'] = self.criterionGAN(pred_fake, True,
                                                for_discriminator=False)
            if not self.opt.no_ganFeat_loss:
                num_D = len(pred_fake)
                GAN_Feat_loss = self.FloatTensor(1).fill_(0)
                for i in range(num_D):  # for each discriminator
                    # last output is the final prediction, so we exclude it
                    num_intermediate_outputs = len(pred_fake[i]) - 1
                    for j in range(num_intermediate_outputs):  # for each layer output
                        unweighted_loss = self.criterionFeat(
                            pred_fake[i][j], pred_real[i][j].detach())
                        GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
                G_losses['GAN_Feat'] = GAN_Feat_loss


        return G_losses, self.heatmap, fake_keypoint
    def compute_discriminator_loss(self, source, target, occlusion_label):
        D_losses = {}
        with torch.no_grad():
            fake_keypoint, occlusion_pred = self.generate_fake(source)
            fake_keypoint = fake_keypoint.detach()
            fake_keypoint.requires_grad_()
        self.transform_keypoints_to_heatmap(source, target, fake_keypoint, occlusion_label)
        pred_fake, pred_real = self.discriminate(self.heatmap['fake_color_map'], self.heatmap['tgt_color_map'])

        D_losses['D_Fake'] = self.criterionGAN(pred_fake, False,
                                               for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real, True,
                                               for_discriminator=True)
        return D_losses
    # TODO: Overhead 줄여야함.
    def transform_keypoints_to_heatmap(self, source_keypoint, target_keypoint, fake_keypoint, fake_occlusion_label) :
        fake_keypoint_ = fake_keypoint.detach().clone()
        occlusion_pred_ = fake_occlusion_label.detach().clone()
        occlusion_index = (occlusion_pred_ >= 0.5).squeeze()
        fake_keypoint_[occlusion_index] = float('nan')
        src_color_map, src_gray_map = util.draw_pose_from_cords(source_keypoint*self.max_point_tensor, (self.opt.max_height, self.opt.max_width))
        tgt_color_map, tgt_gray_map = util.draw_pose_from_cords(target_keypoint*self.max_point_tensor, (self.opt.max_height, self.opt.max_width))
        fake_color_map, fake_gray_map = util.draw_pose_from_cords(fake_keypoint_*self.max_point_tensor, (self.opt.max_height, self.opt.max_width))
        self.heatmap['src_color_map'] = src_color_map
        self.heatmap['src_gray_map'] = src_gray_map
        self.heatmap['tgt_color_map'] = tgt_color_map
        self.heatmap['tgt_gray_map'] = tgt_gray_map
        self.heatmap['fake_color_map'] = fake_color_map
        self.heatmap['fake_gray_map'] = fake_gray_map



    def generate_fake(self, source):
        fake_keypoint, occlusion_pred = self.netG(source)
        return fake_keypoint, occlusion_pred
    def discriminate(self, fake_image, real_image):
        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = np.concatenate([fake_image, real_image], axis=0)
        fake_and_real = torch.Tensor(fake_and_real.transpose(0, 3, 1, 2)).float().cuda() # [B, H, W, C] -> [B, C, H, W]
        discriminator_out = self.netD(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real
    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real
    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0