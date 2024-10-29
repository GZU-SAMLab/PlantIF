import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertModel, BertConfig
import os
import sys
import math
from einops.layers.torch import Rearrange, Reduce

sys.path.append(os.getcwd())
# from Vit_model import ViT
# from Vision_Transformer import VisionTransformer
from utils import ReverseLayerF


class LSTM(torch.nn.Module):
    def __init__(self, class_num, vocab_size, embedding_dim=128, hidden_dim=768, num_layers=1, dropout=0.5):
        super(LSTM, self).__init__()

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.LSTM = torch.nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, batch_first=True)

        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, class_num),
            torch.nn.ReLU()
        )

    def forward(self, inputs):
        inputs = self.word_embeddings(inputs)
        x, _ = self.LSTM(inputs, None)
        x = x[:, -1, :]
        x = self.classifier(x)

        return x


def masked_mean(tensor, mask, dim):
    """Finding the mean along dim"""
    masked = torch.mul(tensor, mask)
    return masked.sum(dim=dim) / mask.sum(dim=dim)


def masked_max(tensor, mask, dim):
    """Finding the max along dim"""
    masked = torch.mul(tensor, mask)
    neg_inf = torch.zeros_like(tensor)
    neg_inf[~mask] = -math.inf
    return (masked + neg_inf).max(dim=dim)


# let's define a simple model that can deal with multimodal variable length sequence
class MISA(nn.Module):
    def __init__(self, config):
        super(MISA, self).__init__()

        self.config = config
        self.text_size = config.embedding_size
        self.visual_size = config.embedding_size
        self.acoustic_size = config.embedding_size

        self.input_sizes = input_sizes = [self.text_size, self.visual_size, self.acoustic_size]
        self.hidden_sizes = hidden_sizes = [int(self.text_size), int(self.visual_size), int(self.acoustic_size)]
        self.output_size = output_size = config.num_classes
        self.dropout_rate = dropout_rate = config.dropout
        self.activation = self.config.activation()
        self.tanh = nn.Tanh()

        rnn = nn.LSTM if self.config.rnncell == "lstm" else nn.GRU
        if self.config.use_bert:
            self.bertmodel = LSTM(class_num=118, vocab_size=500000, embedding_dim=128, hidden_dim=768, num_layers=8, dropout=0.5)
            self.bertmodel.load_state_dict(torch.load('LSTM.mdl'))
            self.bertmodel.classifier = nn.Identity()

            # bertconfig = BertConfig.from_pretrained('/mnt/e/modelfiles/bert-base-uncased', output_hidden_states=True)
            # self.bertmodel = BertModel.from_pretrained('/mnt/e/modelfiles/bert-base-uncased', config=bertconfig)
        else:
            self.embed = nn.Embedding(len(config.word2id), input_sizes[0])
            self.trnn1 = rnn(input_sizes[0], hidden_sizes[0], bidirectional=True)
            self.trnn2 = rnn(2 * hidden_sizes[0], hidden_sizes[0], bidirectional=True)

        self.vrnn1 = rnn(input_sizes[1], hidden_sizes[1], bidirectional=True)
        self.vrnn2 = rnn(2 * hidden_sizes[1], hidden_sizes[1], bidirectional=True)

        self.arnn1 = rnn(input_sizes[2], hidden_sizes[2], bidirectional=True)
        self.arnn2 = rnn(2 * hidden_sizes[2], hidden_sizes[2], bidirectional=True)
        # model = VisionTransformer()
        # model.load_state_dict(torch.load('../Pritrain_vit_bert/VisionTransformer.mdl'))
        # self.vit = nn.Sequential(
        #     model.process[:-1],
        #     Reduce('b n e->b e', reduction='mean'),
        #     nn.Linear(768, 1024)
        # )
        from modelpy.ResNet_18_34 import ResNet18, ResNet34
        from modelpy.DenseNet import DenseNet169, DenseNet264
        from modelpy.ResNet_50_101_152 import ResNet50, ResNet101, ResNet152
        model = ResNet18()
        model.load_state_dict(torch.load('ResNet_18.mdl'))
        model.fc = nn.Linear(512, 1024)

        # model = ResNet34()
        # model.fc = nn.Linear(512, 1024)


        # model = DenseNet264(3, 118)
        # model.classifier = nn.Linear(2688, 1024)

        # model = ResNet50()
        # model.fc = nn.Linear(2048, 1024)

        # model = ResNet101()
        # model.fc = nn.Linear(2048, 1024)

        # model = ResNet152()
        # model.fc = nn.Linear(2048, 1024)

        self.vit = model

        ##########################################
        # mapping modalities to same sized space
        ##########################################
        if self.config.use_bert:
            self.project_t = nn.Sequential()
            self.project_t.add_module('project_t', nn.Linear(in_features=768, out_features=config.hidden_size))
            self.project_t.add_module('project_t_activation', self.activation)
            self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(config.hidden_size))
        else:
            self.project_t = nn.Sequential()
            self.project_t.add_module('project_t',
                                      nn.Linear(in_features=hidden_sizes[0] * 4, out_features=config.hidden_size))
            self.project_t.add_module('project_t_activation', self.activation)
            self.project_t.add_module('project_t_layer_norm', nn.LayerNorm(config.hidden_size))

        self.project_v = nn.Sequential()
        self.project_v.add_module('project_v', nn.Linear(in_features=1024, out_features=config.hidden_size))
        self.project_v.add_module('project_v_activation', self.activation)
        self.project_v.add_module('project_v_layer_norm', nn.LayerNorm(config.hidden_size))

        self.project_a = nn.Sequential()
        self.project_a.add_module('project_a',
                                  nn.Linear(in_features=hidden_sizes[2] * 4, out_features=config.hidden_size))
        self.project_a.add_module('project_a_activation', self.activation)
        self.project_a.add_module('project_a_layer_norm', nn.LayerNorm(config.hidden_size))

        ##########################################
        # private encoders
        ##########################################
        self.private_t = nn.Sequential()
        self.private_t.add_module('private_t_1',
                                  nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_t.add_module('private_t_activation_1', nn.Sigmoid())

        self.private_v = nn.Sequential()
        self.private_v.add_module('private_v_1',
                                  nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_v.add_module('private_v_activation_1', nn.Sigmoid())

        self.private_a = nn.Sequential()
        self.private_a.add_module('private_a_3',
                                  nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.private_a.add_module('private_a_activation_3', nn.Sigmoid())

        ##########################################
        # shared encoder
        ##########################################
        self.shared = nn.Sequential()
        self.shared.add_module('shared_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.shared.add_module('shared_activation_1', nn.Sigmoid())

        ##########################################
        # reconstruct
        ##########################################
        self.recon_t = nn.Sequential()
        self.recon_t.add_module('recon_t_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.recon_v = nn.Sequential()
        self.recon_v.add_module('recon_v_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
        self.recon_a = nn.Sequential()
        self.recon_a.add_module('recon_a_1', nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))

        ##########################################
        # shared space adversarial discriminator
        ##########################################
        if not self.config.use_cmd_sim:
            self.discriminator = nn.Sequential()
            self.discriminator.add_module('discriminator_layer_1',
                                          nn.Linear(in_features=config.hidden_size, out_features=config.hidden_size))
            self.discriminator.add_module('discriminator_layer_1_activation', self.activation)
            self.discriminator.add_module('discriminator_layer_1_dropout', nn.Dropout(dropout_rate))
            self.discriminator.add_module('discriminator_layer_2',
                                          nn.Linear(in_features=config.hidden_size, out_features=len(hidden_sizes)))

        ##########################################
        # shared-private collaborative discriminator
        ##########################################

        self.sp_discriminator = nn.Sequential()
        self.sp_discriminator.add_module('sp_discriminator_layer_1',
                                         nn.Linear(in_features=config.hidden_size, out_features=4))

        self.fusion = nn.Sequential()
        self.fusion.add_module('fusion_layer_1', nn.Linear(in_features=self.config.hidden_size * 4,
                                                           out_features=self.config.hidden_size * 3))
        self.fusion.add_module('fusion_layer_1_dropout', nn.Dropout(dropout_rate))
        self.fusion.add_module('fusion_layer_1_activation', self.activation)
        self.fusion.add_module('fusion_layer_3',
                               nn.Linear(in_features=self.config.hidden_size * 3, out_features=output_size))

        self.tlayer_norm = nn.LayerNorm((hidden_sizes[0] * 2,))
        self.vlayer_norm = nn.LayerNorm((hidden_sizes[1] * 2,))
        self.alayer_norm = nn.LayerNorm((hidden_sizes[2] * 2,))

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.config.hidden_size, nhead=2)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm):
        packed_sequence = pack_padded_sequence(sequence, lengths)

        if self.config.rnncell == "lstm":
            packed_h1, (final_h1, _) = rnn1(packed_sequence)
        else:
            packed_h1, final_h1 = rnn1(packed_sequence)

        padded_h1, _ = pad_packed_sequence(packed_h1)
        normed_h1 = layer_norm(padded_h1)
        packed_normed_h1 = pack_padded_sequence(normed_h1, lengths)

        if self.config.rnncell == "lstm":
            _, (final_h2, _) = rnn2(packed_normed_h1)
        else:
            _, final_h2 = rnn2(packed_normed_h1)

        return final_h1, final_h2

    def alignment(self, visual, input_ids, attention_masks, token_type_ids):

        batch_size = input_ids.size(0)

        bert_output = self.bertmodel(input_ids)

        bert_output = bert_output[0]

        # masked mean
        masked_output = torch.mul(attention_masks.unsqueeze(2), bert_output)
        mask_len = torch.sum(attention_masks, dim=1, keepdim=True)
        bert_output = torch.sum(masked_output, dim=1, keepdim=False) / mask_len

        utterance_text = bert_output

        utterance_video = self.vit(visual)

        # Shared-private encoders
        self.shared_private(utterance_text, utterance_video)

        if not self.config.use_cmd_sim:
            # discriminator
            reversed_shared_code_t = ReverseLayerF.apply(self.utt_shared_t, self.config.reverse_grad_weight)
            reversed_shared_code_v = ReverseLayerF.apply(self.utt_shared_v, self.config.reverse_grad_weight)
            reversed_shared_code_a = ReverseLayerF.apply(self.utt_shared_a, self.config.reverse_grad_weight)

            self.domain_label_t = self.discriminator(reversed_shared_code_t)
            self.domain_label_v = self.discriminator(reversed_shared_code_v)
            self.domain_label_a = self.discriminator(reversed_shared_code_a)
        else:
            self.domain_label_t = None
            self.domain_label_v = None
            self.domain_label_a = None

        self.shared_or_private_p_t = self.sp_discriminator(self.utt_private_t)
        self.shared_or_private_p_v = self.sp_discriminator(self.utt_private_v)
        # self.shared_or_private_p_a = self.sp_discriminator(self.utt_private_a)
        self.shared_or_private_s = self.sp_discriminator((self.utt_shared_t + self.utt_shared_v) / 2.0)

        # For reconstruction
        self.reconstruct()

        # 1-LAYER TRANSFORMER FUSION
        h = torch.stack((self.utt_private_t, self.utt_private_v, self.utt_shared_t, self.utt_shared_v), dim=0)
        h = self.transformer_encoder(h)
        h = torch.cat((h[0]*self.config.private_t_weight, h[1]*self.config.private_v_weight, h[2]*self.config.shared_t_weight, h[3]*self.config.shared_v_weight), dim=1)
        o = self.fusion(h)
        return o

    def reconstruct(self, ):

        self.utt_t = (self.utt_private_t + self.utt_shared_t)
        self.utt_v = (self.utt_private_v + self.utt_shared_v)
        # self.utt_a = (self.utt_private_a + self.utt_shared_a)

        self.utt_t_recon = self.recon_t(self.utt_t)
        self.utt_v_recon = self.recon_v(self.utt_v)
        # self.utt_a_recon = self.recon_a(self.utt_a)

    def shared_private(self, utterance_t, utterance_v, utterance_a=None):

        # Projecting to same sized space
        self.utt_t_orig = utterance_t = self.project_t(utterance_t)
        self.utt_v_orig = utterance_v = self.project_v(utterance_v)
        # self.utt_a_orig = utterance_a = self.project_a(utterance_a)

        # Private-shared components
        self.utt_private_t = self.private_t(utterance_t)
        self.utt_private_v = self.private_v(utterance_v)
        # self.utt_private_a = self.private_a(utterance_a)

        self.utt_shared_t = self.shared(utterance_t)
        self.utt_shared_v = self.shared(utterance_v)
        # self.utt_shared_a = self.shared(utterance_a)

    def forward(self, vision, input_ids, attention_masks, token_type_ids):
        batch_size = input_ids.size(0)
        o = self.alignment(vision, input_ids, attention_masks, token_type_ids)
        return o
