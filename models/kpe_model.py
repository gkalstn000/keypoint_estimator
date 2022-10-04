import torch
import models.networks as networks
import util.util as util

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

    def forward(self, data, mode):
        # Entry point for all calls involving forward pass
        # of deep networks. We used this approach since DataParallel module
        # can't parallelize custom functions, we branch to different
        # routines based on |mode|.
        pass

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
        # transforming the label map to one-hot encoding
        # |data|: dictionary of the input data
        pass

    def compute_generator_loss(self, input_semantics, real_image):
        pass

    def compute_discriminator_loss(self, input_semantics, real_image):
        pass

    def generate_fake(self, input_semantics, real_image, compute_kld_loss=False):
        pass

    def discriminate(self, input_semantics, fake_image, real_image):
        pass
    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        pass
    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0