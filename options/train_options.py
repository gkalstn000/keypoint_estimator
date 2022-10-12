from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        # for displays
        parser.add_argument('--display', action='store_true',
                            help='Whether of showing training results on screen')
        parser.add_argument('--display_freq', type=int, default=8000, help='frequency of showing training results on screen')
        parser.add_argument('--print_freq', type=int, default=8000, help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=8000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=8000, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        parser.add_argument('--debug', action='store_true', help='only do one epoch and displays at each iteration')
        parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')
        # for training
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--niter', type=int, default=500, help='# of iter at starting learning rate. This is NOT the total #epochs. Totla #epochs is niter + niter_decay')
        parser.add_argument('--niter_decay', type=int, default=500, help='# of iter to linearly decay learning rate to zero')
        parser.add_argument('--optimizer', type=str, default='adam')
        parser.add_argument('--beta1', type=float, default=0.0, help='momentum term of adam')
        parser.add_argument('--beta2', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
        parser.add_argument('--D_steps_per_G', type=int, default=1, help='number of discriminator iterations per generator iterations.')

        parser.add_argument('--lambda_mse', type=float, default=10, help='keypoint MSE loss weight')
        parser.add_argument('--lambda_bce', type=float, default=1, help='occlusion BCE loss weight')
        self.isTrain = True
        return parser
