import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes = 1000):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

def resnet152(**kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model

class myResnet(nn.Module):
    def __init__(self, resnet, if_fine_tune, device):
        super(myResnet, self).__init__()
        self.resnet = resnet
        self.if_fine_tune = if_fine_tune
        self.device = device

    def forward(self, x, att_size = 7):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)

        x = self.resnet.avgpool(x)
        x = x.view(x.size(0), -1)


        return x


class BiMPM(nn.Module):

    def __init__(self, n_word_dim, n_perspectives, n_hidden_units, dropout = 0):
        super(BiMPM, self).__init__()
        self.n_word_dim = n_word_dim
        self.n_perspectives = n_perspectives
        self.n_hidden_units = n_hidden_units
        self.dropout = dropout

        # each word represented with d-dimensional vector with two components
        # TODO character embedding
        self.d = n_word_dim
        # l is the number of perspectives
        self.l = n_perspectives


        self.m_full_forward_W = nn.Parameter(torch.rand(self.l, self.n_hidden_units))       # W^1 in paper
        self.m_full_backward_W = nn.Parameter(torch.rand(self.l, self.n_hidden_units))      # W^2 in paper
        self.m_maxpool_forward_W = nn.Parameter(torch.rand(self.l, self.n_hidden_units))    # W^3 in paper
        self.m_maxpool_backward_W = nn.Parameter(torch.rand(self.l, self.n_hidden_units))   # W^4 in paper
        self.m_attn_forward_W = nn.Parameter(torch.rand(self.l, self.n_hidden_units))       # W^5 in paper
        self.m_attn_backward_W = nn.Parameter(torch.rand(self.l, self.n_hidden_units))      # W^6 in paper
        self.m_maxattn_forward_W = nn.Parameter(torch.rand(self.l, self.n_hidden_units))    # W^7 in paper
        self.m_maxattn_backward_W = nn.Parameter(torch.rand(self.l, self.n_hidden_units))   # W^8 in paper

        self.aggregation_lstm = nn.LSTM(
            input_size=8*self.l,
            hidden_size=n_hidden_units,
            num_layers=1,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

    def matching_strategy_full(self, v1, v2, W):
        """
        :param v1: batch x seq_len x n_hidden
        :param v2: batch x n_hidden (FULL) or batch x seq_len x n_hidden (ATTENTIVE)
        :param W:  l x n_hidden
        :return: batch x seq_len x l
        """
        l = W.size(0)
        batch_size = v1.size(0)
        seq_len = v1.size(1)

        v1 = v1.unsqueeze(2).expand(-1, -1, l, -1)          # batch x seq_len x l x n_hidden
        W_expanded = W.expand(batch_size, seq_len, -1, -1)  # batch x seq_len x l x n_hidden
        Wv1 = W_expanded.mul(v1)                            # batch x seq_len x l x n_hidden

        if len(v2.size()) == 2:
            v2 = v2.unsqueeze(1).unsqueeze(1).expand(-1, seq_len, l, -1)  # batch x seq_len x l x n_hidden
        elif len(v2.size()) == 3:
            v2 = v2.unsqueeze(2).expand(-1, -1, l, -1)  # batch x seq_len x l x n_hidden
        else:
            raise ValueError(f'Invalid v2 tensor size {v2.size()}')
        Wv2 = W_expanded.mul(v2)

        cos_sim = F.cosine_similarity(Wv1, Wv2, dim=3)
        return cos_sim

    def matching_strategy_pairwise(self, v1, v2, W):
        """
        Used as a subroutine for (2) Maxpooling-Matching
        :param v1: batch x seq_len_1 x n_hidden
        :param v2: batch x seq_len_2 x n_hidden
        :param W: l x n_hidden
        :return: batch x seq_len_1 x seq_len_2 x l
        """
        l = W.size(0)
        batch_size = v1.size(0)

        v1_expanded = v1.unsqueeze(1).expand(-1, l, -1, -1)                 # batch x l x seq_len_1 x n_hidden
        W1_expanded = W.unsqueeze(1).expand(batch_size, -1, v1.size(1), -1) # batch x l x seq_len_1 x n_hidden
        Wv1 = W1_expanded.mul(v1_expanded)                                  # batch x l x seq_len_1 x n_hidden

        v2_expanded = v2.unsqueeze(1).expand(-1, l, -1, -1)                 # batch x l x seq_len_2 x n_hidden
        W2_expanded = W.unsqueeze(1).expand(batch_size, -1, v2.size(1), -1) # batch x l x seq_len_2 x n_hidden
        Wv2 = W2_expanded.mul(v2_expanded)                                  # batch x l x seq_len_2 x n_hidden

        dot = torch.matmul(Wv1, Wv2.transpose(3,2))
        v1_norm = v1_expanded.norm(p=2, dim=3, keepdim=True)
        v2_norm = v2_expanded.norm(p=2, dim=3, keepdim=True)
        norm_product = torch.matmul(v1_norm, v2_norm.transpose(3,2))

        cosine_matrix = dot / norm_product
        cosine_matrix = cosine_matrix.permute(0, 2, 3, 1)

        return cosine_matrix

    def matching_strategy_attention(self, v1, v2):
        """
        Used as a subroutine for (3) Attentive-Matching
        :param v1: batch x seq_len_1 x n_hidden
        :param v2: batch x seq_len_2 x n_hidden
        :return: batch x seq_len_1 x seq_len_2
        """
        dot = torch.bmm(v1, v2.transpose(2, 1))
        v1_norm = v1.norm(p=2, dim=2, keepdim=True)
        v2_norm = v2.norm(p=2, dim=2, keepdim=True)
        norm_product = torch.bmm(v1_norm, v2_norm.transpose(2, 1))

        return dot / norm_product

    def forward(self, context_lstm, aspect_lstm):
        s1_context_forward, s1_context_backward = torch.split(context_lstm, self.n_hidden_units, dim=2)
        s2_context_forward, s2_context_backward = torch.split(aspect_lstm, self.n_hidden_units, dim=2)

        # Matching Layer

        # (1) Full matching
        m_full_s1_f = self.matching_strategy_full(s1_context_forward, s2_context_forward[:, -1, :], self.m_full_forward_W)
        m_full_s1_b = self.matching_strategy_full(s1_context_backward, s2_context_backward[:, 0, :], self.m_full_backward_W)
        m_full_s2_f = self.matching_strategy_full(s2_context_forward, s1_context_forward[:, -1, :], self.m_full_forward_W)
        m_full_s2_b = self.matching_strategy_full(s2_context_backward, s1_context_backward[:, 0, :], self.m_full_backward_W)

        # (2) Maxpooling-Matching
        m_pair_f = self.matching_strategy_pairwise(s1_context_forward, s2_context_backward, self.m_maxpool_forward_W)
        m_pair_b = self.matching_strategy_pairwise(s1_context_backward, s2_context_backward, self.m_maxpool_backward_W)

        m_maxpool_s1_f, _ = m_pair_f.max(dim=2)
        m_maxpool_s1_b, _ = m_pair_b.max(dim=2)
        m_maxpool_s2_f, _ = m_pair_f.max(dim=1)
        m_maxpool_s2_b, _ = m_pair_b.max(dim=1)

        # (3) Attentive-Matching
        # cosine_f and cosine_b are batch x seq_len_1 x seq_len_2
        cosine_f = self.matching_strategy_attention(s1_context_forward, s2_context_forward)
        cosine_b = self.matching_strategy_attention(s1_context_backward, s2_context_backward)

        # attn_s1_f and others are batch x seq_len_1 x seq_len_2 x n_hidden
        attn_s1_f = s1_context_forward.unsqueeze(2) * cosine_f.unsqueeze(3)
        attn_s1_b = s1_context_forward.unsqueeze(2) * cosine_b.unsqueeze(3)
        attn_s2_f = s2_context_forward.unsqueeze(1) * cosine_f.unsqueeze(3)
        attn_s2_b = s2_context_forward.unsqueeze(1) * cosine_b.unsqueeze(3)

        attn_mean_vec_s2_f = attn_s1_f.sum(dim=1) / cosine_f.sum(1, keepdim=True).transpose(2, 1)  # batch x seq_len_2 x hidden
        attn_mean_vec_s2_b = attn_s1_b.sum(dim=1) / cosine_b.sum(1, keepdim=True).transpose(2, 1)  # batch x seq_len_2 x hidden
        attn_mean_vec_s1_f = attn_s2_f.sum(dim=2) / cosine_f.sum(2, keepdim=True)                  # batch x seq_len_1 x hidden
        attn_mean_vec_s1_b = attn_s2_b.sum(dim=2) / cosine_b.sum(2, keepdim=True)                  # batch x seq_len_1 x hidden

        m_attn_s1_f = self.matching_strategy_full(s1_context_forward, attn_mean_vec_s1_f, self.m_attn_forward_W)
        m_attn_s1_b = self.matching_strategy_full(s1_context_backward, attn_mean_vec_s1_b, self.m_attn_forward_W)
        m_attn_s2_f = self.matching_strategy_full(s2_context_forward, attn_mean_vec_s2_f, self.m_attn_forward_W)
        m_attn_s2_b = self.matching_strategy_full(s2_context_backward, attn_mean_vec_s2_b, self.m_attn_forward_W)

        # (4) Max-Attentive-Matching
        attn_max_vec_s2_f, _ = attn_s1_f.max(dim=1)  # batch x seq_len_2 x hidden
        attn_max_vec_s2_b, _ = attn_s1_b.max(dim=1)  # batch x seq_len_2 x hidden
        attn_max_vec_s1_f, _ = attn_s2_f.max(dim=2)  # batch x seq_len_1 x hidden
        attn_max_vec_s1_b, _ = attn_s2_b.max(dim=2)  # batch x seq_len_1 x hidden

        m_maxattn_s1_f = self.matching_strategy_full(s1_context_forward, attn_max_vec_s1_f, self.m_maxattn_forward_W)
        m_maxattn_s1_b = self.matching_strategy_full(s1_context_backward, attn_max_vec_s1_b, self.m_maxattn_forward_W)
        m_maxattn_s2_f = self.matching_strategy_full(s2_context_forward, attn_max_vec_s2_f, self.m_maxattn_forward_W)
        m_maxattn_s2_b = self.matching_strategy_full(s2_context_backward, attn_max_vec_s2_b, self.m_maxattn_forward_W)

        s1_combined_match_vec = torch.cat([m_full_s1_f, m_maxpool_s1_f, m_attn_s1_f, m_maxattn_s1_f,
                                           m_full_s1_b, m_maxpool_s1_b, m_attn_s1_b, m_maxattn_s1_b], dim=2)
        s2_combined_match_vec = torch.cat([m_full_s2_f, m_maxpool_s2_f, m_attn_s2_f, m_maxattn_s2_f,
                                           m_full_s2_b, m_maxpool_s2_b, m_attn_s2_b, m_maxattn_s2_b], dim=2)

        # Aggregation Layer
        s1_agg_out, (s1_agg_h, s1_agg_c) = self.aggregation_lstm(s1_combined_match_vec)
        s2_agg_out, (s2_agg_h, s2_agg_c) = self.aggregation_lstm(s2_combined_match_vec)

        # s1_agg_h and s2_agg_h are 2 x batch x n_hidden
        matching_vector = torch.cat([s1_agg_h.transpose(1, 0), s2_agg_h.transpose(1, 0)], dim=1).view(-1, 4 * self.n_hidden_units)

        return matching_vector




class MMIAN(nn.Module):
    def __init__(self, embedding_matrix, device):
        super(MMIAN, self).__init__()
        self.hidden_dim = 256
        self.device = device

        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))

        self.lstm_aspect = nn.LSTM(input_size = 100, hidden_size = 128, num_layers = 1, bidirectional = True, batch_first=True)
        self.lstm_context = nn.LSTM(input_size = 100, hidden_size = 128, num_layers = 1, bidirectional = True, batch_first=True)

        self.bimpm = BiMPM(256, 128, 128, dropout=0.1)
        self.placeholder = nn.LSTM(input_size = 256, hidden_size = 256, num_layers = 1, bidirectional = False, batch_first=True)
      
        self.vis2text = nn.Linear(2048, self.hidden_dim)
        self.fc_layer = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),
            nn.Linear(128, 4),
        )

    def forward(self, text, entity, visual_embeds_global):


        context = self.embed(text)
        aspect = self.embed(entity)
        context_lstm, (_, _) = self.lstm_context(context)
        aspect_lstm, (_, _) = self.lstm_aspect(aspect)


        text_representation = self.bimpm(aspect_lstm, context_lstm)
        converted_vis_embed = self.vis2text(torch.tanh(visual_embeds_global))

        x = torch.cat((text_representation, converted_vis_embed), dim=-1)
        out = self.fc_layer(x)

        return out, _, _