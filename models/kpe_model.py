import torch
import models.networks as networks
import util.util as util
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

        self.netG = self.initialize_networks(opt)

        # set loss functions
        if opt.isTrain:
            self.criterionMSE = torch.nn.MSELoss() # for keypoint loss
            self.criterionBCE = torch.nn.BCELoss() # for occlusion loss

        self.max_point_tensor = torch.Tensor([opt.max_height-1, opt.max_width]).cuda()

        self.heatmap = {}

    def forward(self, data, mode):
        source, target, occlusion_label = self.preprocess_input(data)

        if mode == 'generator' :
            g_losses, g_map, fake_keypoint = self.compute_generator_loss(source, target, occlusion_label)
            return g_losses, g_map, fake_keypoint
        elif mode == 'inference' :
            with torch.no_grad() :
                fake_keypoint, occlusion_pred = self.generate_fake(source)
                self.transform_keypoints_to_heatmap(source, target, fake_keypoint, occlusion_pred)
                return fake_keypoint, occlusion_pred

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())

        beta1, beta2 = opt.beta1, opt.beta2
        G_lr, D_lr = opt.lr, opt.lr

        optimizer_G = torch.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))

        return optimizer_G

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)

        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
        return netG

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
        if self.opt.display:
            self.transform_keypoints_to_heatmap(source, target, fake_keypoint, occlusion_pred)
        # pred_fake, pred_real = self.discriminate(fake_gray_map, real_gray_map)

        return G_losses, self.heatmap, fake_keypoint

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

    def compute_discriminator_loss(self, input_semantics, real_image):
        pass

    def generate_fake(self, source):
        fake_keypoint, occlusion_pred = self.netG(source)
        return fake_keypoint, occlusion_pred
    def discriminate(self, fake_image, real_image):
        pass
    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        pass
    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0