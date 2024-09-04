from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
import math
from .util import initialize_weights
from .util import NystromAttention
from .util import BilinearFusion
from .util import SNN_Block
from .util import MultiheadAttention




import torch
import torch.nn as nn
import torch.nn.functional as F
# =================================================================
class GatedBimodal(nn.Module):
    def __init__(self, dim):
        super(GatedBimodal, self).__init__()
        self.dim = dim
        self.linear_h = nn.Linear(2 * dim, 2 * dim)
        self.linear_z = nn.Linear(2 * dim, dim)
        self.activation = torch.tanh
        self.gate_activation = torch.sigmoid

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), dim=1)
        h = self.activation(self.linear_h(x))
        z = self.gate_activation(self.linear_z(x))
        return z * h[:, :self.dim] + (1 - z) * h[:, self.dim:], z

class MLPGenreClassifier(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size):
        super(MLPGenreClassifier, self).__init__()
        self.layernorm1 = nn.LayerNorm(input_dim)
        self.linear1 = nn.Linear(input_dim, hidden_size)
        self.layernorm2 = nn.LayerNorm(hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.layernorm3 = nn.LayerNorm(hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_dim)
        self.output_act = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.layernorm1(x)
        x = F.relu(self.linear1(x))
        x = self.layernorm2(x)
        x = F.relu(self.linear2(x))
        x = self.layernorm3(x)
        x = self.linear3(x)
        return self.output_act(x)

class GatedClassifier(nn.Module):
    def __init__(self, visual_dim, textual_dim, output_dim, hidden_size):
        super(GatedClassifier, self).__init__()
        self.visual_mlp = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, hidden_size, bias=False)
        )
        self.textual_mlp = nn.Sequential(
            nn.LayerNorm(textual_dim),
            nn.Linear(textual_dim, hidden_size, bias=False)
        )
        self.gbu = GatedBimodal(hidden_size)
        self.logistic_mlp = MLPGenreClassifier(hidden_size, output_dim, hidden_size)

    def forward(self, x_v, x_t):
        visual_h = self.visual_mlp(x_v)
        textual_h = self.textual_mlp(x_t)
        h, z = self.gbu(visual_h, textual_h)
        y_hat = self.logistic_mlp(h)
        return y_hat, z

class LinearSumClassifier(nn.Module):
    def __init__(self, visual_dim, textual_dim, output_dim, hidden_size):
        super(LinearSumClassifier, self).__init__()
        self.visual_layer = nn.Sequential(
            nn.LayerNorm(visual_dim),
            nn.Linear(visual_dim, hidden_size, bias=False)
        )
        self.textual_layer = nn.Sequential(
            nn.LayerNorm(textual_dim),
            nn.Linear(textual_dim, hidden_size, bias=False)
        )
        self.logistic_mlp = MLPGenreClassifier(hidden_size, output_dim, hidden_size)

    def forward(self, x_v, x_t):
        h = self.visual_layer(x_v) + self.textual_layer(x_t)
        return self.logistic_mlp(h)

# class ConcatenateClassifier(nn.Module):
#     def __init__(self, input_dim, output_dim, hidden_size):
#         super(ConcatenateClassifier, self).__init__()
#         self.linear1 = nn.Linear(input_dim, hidden_size * 2, bias=False)
#         self.layernorm1 = nn.LayerNorm(hidden_size * 2)
#         self.linear2 = nn.Linear(hidden_size * 2, hidden_size, bias=False)
#         self.layernorm2 = nn.LayerNorm(hidden_size)
#         self.linear3 = nn.Linear(hidden_size, output_dim, bias=False)
#         self.logistic = nn.LogSoftmax(dim=1)
#
#     def forward(self, x):
#         x = F.relu(self.layernorm1(self.linear1(x)))
#         x = F.relu(self.layernorm2(self.linear2(x)))
#         x = self.linear3(x)
#         return self.logistic(x)

class MoEClassifier(nn.Module):
    def __init__(self, visual_dim, textual_dim, output_dim, hidden_size):
        super(MoEClassifier, self).__init__()
        self.visual_mlp = MLPGenreClassifier(visual_dim, output_dim, hidden_size)
        self.textual_mlp = MLPGenreClassifier(textual_dim, output_dim, hidden_size)
        self.manager_mlp = nn.Sequential(
            nn.LayerNorm(visual_dim + textual_dim),
            nn.Linear(visual_dim + textual_dim, 1, bias=False)
        )

    def forward(self, x_v, x_t):
        y_v = self.visual_mlp(x_v)
        y_t = self.textual_mlp(x_t)
        manager = self.manager_mlp(torch.cat([x_v, x_t], dim=1))
        g = F.softmax(manager, dim=1)
        y = torch.stack([y_v, y_t])
        return (g.T * y).mean(dim=0) * 1.999 + 1e-5
# ==================================================================================

def dissimilarity_loss(A, B):
    assert A.shape == B.shape, "A and B must have the same shape"
    A_flat = A.view(A.size(0), -1)  # 展平成 B x (N*Dim)
    B_flat = B.view(B.size(0), -1)  # 展平成 B x (N*Dim)
    euclidean_distance = torch.norm(A_flat - B_flat, p=2, dim=-1)
    loss = (1 / (euclidean_distance + 1e-8)).mean()
    return loss

# 定义 FFNExpert 类
class FFNExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FFNExpert, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.bn2 = nn.LayerNorm(input_dim)

    def forward(self, x):
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))
        return x
# 定义 MoE 类
class MoE(nn.Module):
    def __init__(self, input_dim=512, num_experts=4, k=2):
        super(MoE, self).__init__()
        self.k = k
        self.gate = nn.Linear(input_dim, num_experts)
        self.experts = nn.ModuleList(
            [FFNExpert(input_dim, input_dim) for _ in range(num_experts)])

    def forward(self, x):
        B, N, input_dim = x.shape
        x_reshaped = x.view(B * N, input_dim)
        gate_scores = self.gate(x_reshaped)
        topk_scores, topk_indices = gate_scores.view(B, N, -1).topk(self.k, dim=2)
        bottomk_scores, bottomk_indices = gate_scores.view(B, N, -1).topk(self.k, dim=2, largest=False)

        expert_outputs_top = torch.zeros(B, N, self.k, input_dim, device=x.device)
        expert_outputs_bottom = torch.zeros(B, N, self.k, input_dim, device=x.device)

        for b in range(B):
            for n in range(N):
                for i, idx in enumerate(topk_indices[b, n]):
                    expert_outputs_top[b, n, i] = self.experts[idx](x[b, n].unsqueeze(0))
                for i, idx in enumerate(bottomk_indices[b, n]):
                    expert_outputs_bottom[b, n, i] = self.experts[idx](x[b, n].unsqueeze(0))

        weights_top = torch.softmax(topk_scores, dim=2).unsqueeze(-1)
        weights_bottom = torch.softmax(bottomk_scores, dim=2).unsqueeze(-1)

        output_top = (weights_top * expert_outputs_top).sum(dim=2)
        output_bottom = (weights_bottom * expert_outputs_bottom).sum(dim=2)

        orthogonality_loss = dissimilarity_loss(output_top, output_bottom)

        output = output_top + x

        return output, output_top, output_bottom, orthogonality_loss


# =========================================================================================


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512, num_experts=4, k=2):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,
            # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,
            # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1,
        )
        # self.moe = MoE(input_dim=dim, num_experts=num_experts, k=k)

    def forward(self, x):
        x = x + self.attn(self.norm(x))
        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class Transformer_P(nn.Module):
    def __init__(self, feature_dim=512, num_experts=4, k=2):
        super(Transformer_P, self).__init__()
        # Encoder
        self.pos_layer = PPEG(dim=feature_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        nn.init.normal_(self.cls_token, std=1e-6)
        self.layer1 = TransLayer(dim=feature_dim)
        self.layer2 = TransLayer(dim=feature_dim)

        self.norm = nn.LayerNorm(feature_dim)
        # Decoder

    def forward(self, features):
        # ---->pad
        H = features.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([features, features[:, :add_length, :]], dim=1)  # [B, N, 512]
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)
        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]
        # ---->MoE layer
        #   h = self.moe(h)  # [B, N, 512]
        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]
        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]
        # ---->cls_token
        # h = self.norm(h)

        return h[:, 0], h[:, 1:]


class Transformer_G(nn.Module):
    def __init__(self, feature_dim=512, num_experts=4, k=2):
        super(Transformer_G, self).__init__()
        # Encoder
        # self.pos_layer = PPEG(dim=feature_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))
        nn.init.normal_(self.cls_token, std=1e-6)
        self.layer1 = TransLayer(dim=feature_dim)
        self.layer2 = TransLayer(dim=feature_dim)
        self.moe = MoE(input_dim=feature_dim, num_experts=num_experts, k=k)
        self.norm = nn.LayerNorm(feature_dim)
        # Decoder

    def forward(self, features):
        # ---->pad
        H = features.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([features, features[:, :add_length, :]], dim=1)  # [B, N, 512]
        # ---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)
        # ---->Translayer x1
        # h = self.moe(h)

        h = self.layer1(h)  # [B, N, 512]
        # ---->MoE layer
        h ,_1,__1,Nloss1= self.moe(h)  # [B, N, 512]
        # ---->PPEG
        # h = self.pos_layer(h, _H, _W)  # [B, N, 512]
        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]
        # ---->cls_token
        # ---->MoE layer
        h ,_2,__2,Nloss2= self.moe(h)  # [B, N, 512]

        return h[:, 0], h[:, 1:],Nloss1+Nloss2


class token_selection(nn.Module):
    def __init__(self):
        super(token_selection, self).__init__()
        self.MLP_f = nn.Linear(256, 128)
        self.MLP_s= nn.Linear(256, 256)
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)

    def forward(self, start_patch_token, cls_token,Temperature):
        half_token_patch = self.MLP_f(start_patch_token)
        half_token_patch = self.relu(self.dropout(half_token_patch))
        half_token_cls = self.MLP_f(cls_token)
        half_token_cls = half_token_cls.unsqueeze(1)
        half_token_cls = half_token_cls.repeat(1, start_patch_token.size(1), 1)  # Corrected this line
        patch_token = torch.cat([half_token_cls, half_token_patch], dim=2)
        patch_token = self.MLP_s(patch_token)
        patch_token = self.relu(self.dropout(patch_token))
        _patch_token = self.softmax(patch_token)
        topk_values, topk_indices = torch.topk(_patch_token, math.ceil(start_patch_token.size(1)*Temperature), dim=1)
        final_token = torch.gather(start_patch_token, 1, topk_indices.squeeze(1))  # Squeeze the last dimension here

        return final_token



from hypll.manifolds.poincare_ball import Curvature, PoincareBall
from hypll.tensors import TangentTensor
from torch import nn
from hypll import nn as hnn
# from fusion import GatedClassifier,LinearSumClassifier,ConcatenateClassifier,MoEClassifier


manifold = PoincareBall(c=Curvature(requires_grad=True))

class CMTA(nn.Module):
    def __init__(self, omic_sizes=[100, 200, 300, 400, 500, 600], n_classes=4, fusion="concat", model_size="small",alpha=0.5,beta=0.5,tokenS="both",GT=0.5,PT=0.5,HRate=1e-8):
        super(CMTA, self).__init__()
        self.omic_sizes = omic_sizes
        self.n_classes = n_classes
        self.fusion = fusion
        self.alpha=alpha
        self.beta=beta
        self.tokenS=tokenS
        self.GT=GT
        self.PT=PT
        self.HRate=HRate
        ###
        self.size_dict = {
            "pathomics": {"small": [1024, 256, 256], "large": [1024, 512, 256]},
            "genomics": {"small": [1024, 256], "large": [1024, 1024, 1024, 256]},
        }
        # Pathomics Embedding Network
        hidden = self.size_dict["pathomics"][model_size]
        fc = []
        for idx in range(len(hidden) - 1):
            fc.append(nn.Linear(hidden[idx], hidden[idx + 1]))
            fc.append(nn.ReLU())
            fc.append(nn.Dropout(0.25))
        self.pathomics_fc = nn.Sequential(*fc)
        # Genomic Embedding Network
        hidden = self.size_dict["genomics"][model_size]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i + 1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.genomics_fc = nn.ModuleList(sig_networks)

        # Pathomics Transformer
        # Encoder
        self.pathomics_encoder = Transformer_P(feature_dim=hidden[-1])
        # Decoder
        self.pathomics_decoder = Transformer_P(feature_dim=hidden[-1])

        # P->G Attention
        self.P_in_G_Att = MultiheadAttention(embed_dim=256, num_heads=1)
        # G->P Attention
        self.G_in_P_Att = MultiheadAttention(embed_dim=256, num_heads=1)

        # Pathomics Transformer Decoder
        # Encoder
        self.genomics_encoder = Transformer_G(feature_dim=hidden[-1])
        # Decoder
        self.genomics_decoder = Transformer_G(feature_dim=hidden[-1])

        self.hyperbolic_fc1 = hnn.HLinear(in_features=hidden[-1] * 2, out_features=hidden[-1], manifold=manifold)
        self.hyperbolic_fc2 = hnn.HLinear(in_features=hidden[-1], out_features=hidden[-1], manifold=manifold)
        self.hyperbolic_relu = hnn.HReLU(manifold=manifold)
        self.token_selection = token_selection()


        # Classification Layer
        if self.fusion == "Aconcat" or self.fusion == "concat":
            self.mm = nn.Sequential(
                *[nn.Linear(hidden[-1] * 2, hidden[-1]), nn.ReLU(), nn.Linear(hidden[-1], hidden[-1]), nn.ReLU()]
            )
        elif self.fusion == "fineCoarse":
            self.mm = nn.Sequential(
                *[nn.Linear(hidden[-1] * 2, hidden[-1]), nn.ReLU(), nn.Linear(hidden[-1], hidden[-1]), nn.ReLU()]
            )
        elif self.fusion == "bilinear":
            self.mm = BilinearFusion(dim1=hidden[-1], dim2=hidden[-1], scale_dim1=8, scale_dim2=8, mmhid=hidden[-1])

        elif self.fusion == "hyperbolic":
            self.hyperbolic_mm = nn.Sequential(
                self.hyperbolic_fc1,
                self.hyperbolic_relu,
                self.hyperbolic_fc2,
                self.hyperbolic_relu,
                self.hyperbolic_fc2
            )
            self.mm = nn.Sequential(
                *[nn.Linear(hidden[-1] * 2, hidden[-1]), nn.ReLU(), nn.Linear(hidden[-1], hidden[-1]), nn.ReLU()]
            )
        else:
            pass

        self.classifier = nn.Linear(hidden[-1], self.n_classes)

        self.apply(initialize_weights)

    def forward(self, **kwargs):
        # meta genomics and pathomics features
        x_path = kwargs["x_path"]
        x_omic = [kwargs["x_omic%d" % i] for i in range(1, 7)]


        # Enbedding
        # genomics embedding
        genomics_features = [self.genomics_fc[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)]
        genomics_features = torch.stack(genomics_features).unsqueeze(0)  # [1, 6, 1024]
        # pathomics embedding
        pathomics_features = self.pathomics_fc(x_path).unsqueeze(0)
        # x_path:torch.Size([1, 2048, 1024]) 4096 patch

        # print("genomics_features.shape: ",genomics_features.shape)
        # print("pathomics_features.shape:",pathomics_features.shape)
        # encoder
        # pathomics encoder
        cls_token_pathomics_encoder, patch_token_pathomics_encoder = self.pathomics_encoder(
            pathomics_features)  # cls token + patch tokens
        # genomics encoder
        cls_token_genomics_encoder, patch_token_genomics_encoder ,Nloss1= self.genomics_encoder(
            genomics_features)  # cls token + patch tokens

        # print("cls_token_pathomics_encoder.shape: ",cls_token_pathomics_encoder.shape)
        # print("cls_token_genomics_encoder.shape: ",cls_token_genomics_encoder.shape)
        # print("patch_token_pathomics_encoder.shape: ",patch_token_pathomics_encoder.shape)
        # print("patch_token_genomics_encoder.shape: ",patch_token_genomics_encoder.shape)
        # cross-omics attention

        #=============== token selection;

        if self.tokenS=="both":
            patch_token_pathomics_encoder=self.token_selection(patch_token_pathomics_encoder, cls_token_pathomics_encoder,0.5)
            patch_token_genomics_encoder=self.token_selection(patch_token_genomics_encoder, cls_token_genomics_encoder,0.5)
        elif self.tokenS=="P":
            patch_token_pathomics_encoder=self.token_selection(patch_token_pathomics_encoder, cls_token_pathomics_encoder,0.5)
        elif self.tokenS=="G":
            patch_token_genomics_encoder=self.token_selection(patch_token_genomics_encoder, cls_token_genomics_encoder,0.5)
        elif self.tokenS=="N":
            pass

        # =============== token selection;
        # print("===========")
        # print("patch_token_pathomics_encoder.shape: ", patch_token_pathomics_encoder.shape)
        # print("patch_token_genomics_encoder.shape: ", patch_token_genomics_encoder.shape)
        # print("===========")
        pathomics_in_genomics, Att = self.P_in_G_Att(
            patch_token_pathomics_encoder.transpose(1, 0),
            patch_token_genomics_encoder.transpose(1, 0),
            patch_token_genomics_encoder.transpose(1, 0),
        )  # ([14642, 1, 256])
        genomics_in_pathomics, Att = self.G_in_P_Att(
            patch_token_genomics_encoder.transpose(1, 0),
            patch_token_pathomics_encoder.transpose(1, 0),
            patch_token_pathomics_encoder.transpose(1, 0),
        )  # ([7, 1, 256])
        # decoder
        # print(" pathomics_in_genomics: " ,pathomics_in_genomics.shape)
        # print(" genomics_in_pathomics: " ,genomics_in_pathomics.shape)


        # pathomics decoder
        cls_token_pathomics_decoder, tokens_p_d = self.pathomics_decoder(
            pathomics_in_genomics.transpose(1, 0))  # cls token + patch tokens
        # genomics decoder
        cls_token_genomics_decoder,tokens_g_d,Nloss2 = self.genomics_decoder(
            genomics_in_pathomics.transpose(1, 0))  # cls token + patch tokens

        if self.tokenS=="both":
            patch_token_pathomics_decoder=self.token_selection(tokens_p_d, cls_token_pathomics_decoder,self.PT)
            patch_token_genomics_decoder=self.token_selection(tokens_g_d, cls_token_genomics_decoder,self.GT)
        elif self.tokenS=="P":
            patch_token_pathomics_decoder = self.token_selection(tokens_p_d, cls_token_pathomics_decoder, self.PT)
        elif self.tokenS=="G":
            patch_token_genomics_decoder=self.token_selection(tokens_g_d, cls_token_genomics_decoder,self.GT)
        elif self.tokenS=="N":
            pass
        p = patch_token_pathomics_decoder.reshape(patch_token_pathomics_decoder.shape[0], -1)
        g = patch_token_genomics_decoder.reshape(patch_token_genomics_decoder.shape[0], -1)

        lmf = LMF(input_dims=(p.shape[0], g.shape[0]), output_dim=256, rank=4).to(p.device)
        output = lmf(p, g)
        logits = self.classifier(output)
        # print(output.shape)  # should print torch.Size([1, 256])

        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        return hazards, S, cls_token_pathomics_encoder, cls_token_pathomics_decoder, cls_token_genomics_encoder, cls_token_genomics_decoder,Nloss2+Nloss1




import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_



class LMF(nn.Module):
    '''
    Low-rank Multimodal Fusion
    '''

    def __init__(self, input_dims, output_dim, rank, use_softmax=False):
        '''
        Args:
            input_dims - a length-2 tuple, contains (audio_dim, video_dim)
            output_dim - int, specifying the size of output
            rank - int, specifying the size of rank in LMF
        Output:
            (return value in forward) a tensor of shape (batch_size, output_dim)
        '''
        super(LMF, self).__init__()

        # dimensions are specified in the order of audio and video
        self.audio_in = input_dims[0]
        self.video_in = input_dims[1]


        self.output_dim = output_dim
        self.rank = rank


        # define the pre-fusion subnetworks

        # define the post_fusion layers
        # self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        self.audio_factor = Parameter(torch.Tensor(self.rank, self.audio_in + 1, self.output_dim))
        self.video_factor = Parameter(torch.Tensor(self.rank, self.video_in + 1, self.output_dim))
        self.fusion_weights = Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = Parameter(torch.Tensor(1, self.output_dim))

        # init the factors
        xavier_normal_(self.audio_factor)
        xavier_normal_(self.video_factor)
        xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def forward(self, audio_x, video_x):
        '''
        Args:
            audio_x: tensor of shape (batch_size, audio_in)
            video_x: tensor of shape (batch_size, video_in)
        '''
        audio_h = audio_x
        video_h = video_x
        batch_size = audio_h.shape[0]

        _audio_h = torch.cat((
            Variable(torch.ones(batch_size, 1).type(torch.FloatTensor), requires_grad=False).to(audio_h.device),
            audio_h
        ), dim=1)
        _video_h = torch.cat((Variable(torch.ones(batch_size, 1).type( torch.FloatTensor), requires_grad=False).to(audio_h.device), video_h), dim=1)


        fusion_audio = torch.matmul(_audio_h, self.audio_factor)
        fusion_video = torch.matmul(_video_h, self.video_factor)
        fusion_zy = (fusion_audio * fusion_video)

        output = (torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias)
        output = output.view(-1, self.output_dim)

        return output

