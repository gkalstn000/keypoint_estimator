import torch.nn as nn
from models.networks.base_network import BaseNetwork
from models.networks.transformer_encoder import EncoderLayer, gelu
from models.networks.transformer_embedding import Embedding
def fc_layer(size_in, size_out, keep_prob=0.8):
    linear = nn.Linear(size_in, size_out)

    layer = nn.Sequential(
        linear,
        nn.BatchNorm1d(18), # 이거 설정해보기
        nn.ReLU(),
        nn.Dropout(p=1 - keep_prob),

    )

    return layer

class TransformerGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        # embedding options
        parser.add_argument('--grid_num', type=int, default=100, help='number of grid')
        parser.add_argument('--embedding_dim', type=int, default=3, help='grid embedding dimension')
        # encoder options
        parser.add_argument('--output_dim', type=int, default=2, help='output dimension')
        parser.add_argument('--n_layers', type=int, default=3, help='number of bidirectional lstm layers')
        parser.add_argument('--d_k', type=int, default=2, help='number of bidirectional lstm layers')
        parser.add_argument('--d_v', type=int, default=2, help='number of bidirectional lstm layers')
        parser.add_argument('--n_heads', type=int, default=3, help='number of bidirectional lstm layers')
        # transformer options
        parser.add_argument('--d_ff', type=int, default=128, help='number of bidirectional lstm layers')

        return parser
    def __init__(self, opt):
        super().__init__()
        d_model = opt.embedding_dim * 2 + 1
        self.opt = opt
        self.final_linear = nn.Linear(d_model, d_model)
        self.gelu = gelu
        # Embedding Network
        self.embedding = Embedding(grid_num=opt.grid_num, embedding_dim=opt.embedding_dim)
        # Encoder Network
        self.layers = nn.ModuleList([EncoderLayer(d_model = d_model, d_k = opt.d_k, d_v = opt.d_v, n_heads = opt.n_heads, d_ff = opt.d_ff) for _ in range(opt.n_layers)])
        # keypoint
        self.Keypoint_Regressor = nn.ModuleList()
        input_ = d_model
        for output in [self.d_ff, 32]:
            self.Keypoint_Regressor.append(fc_layer(input_, output))
            input_ = output
        layer = nn.Sequential(
            nn.Linear(input_, opt.output_dim),
        )
        self.Keypoint_Regressor.append(layer)
        # occlusion
        self.Occlusion_Classifier = nn.ModuleList()
        input_ = d_model # embedding*2+1
        for output in [self.d_ff, 32]:
            self.Occlusion_Classifier.append(fc_layer(input_, output))
            input_ = output
        layer = nn.Sequential(
            nn.Linear(input_, 1),
            nn.Sigmoid()
        )
        self.Occlusion_Classifier.append(layer)
    def forward(self, x):
        if len(x.size()) != 3 :
            x = x.unsqueeze(0)

        output = self.embedding(x) # [bach_size, seq_len, d_model]
        for layer in self.layers:
            output = layer(output)     # output: [batch_size, max_len, d_model]
        output = self.gelu(self.final_linear(output)) # [batch_size, max_pred, d_model]
        # keypoint
        keypoint_logits = output
        for linear in self.MLM_Regressor :
            keypoint_logits = linear(keypoint_logits)
        # occlusion
        occlusion_logits = output
        for linear in self.Classifier :
            occlusion_logits = linear(occlusion_logits)

        return keypoint_logits, occlusion_logits
