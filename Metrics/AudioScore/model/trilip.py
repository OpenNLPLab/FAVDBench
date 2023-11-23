from einops import reduce
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .load_passt import MyPasst


class FFN(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)  # position-wise
        self.w_2 = nn.Linear(d_hid, d_in)  # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class TriLip(nn.Module):

    def __init__(self, vl_model) -> None:
        super().__init__()

        # load clip and fix
        self.vl_model = vl_model
        for _, p in self.vl_model.named_parameters():
            p.requires_grad = False

        # load passt and fix
        self.a_model = MyPasst()
        self.a_model.freeze()

        self.logit_scale1 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale2 = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.vis_scaler = FFN(512, 2048)
        self.txt_scaler = FFN(512, 2048)
        self.aud_scaler = FFN(512, 2048)


    def forward(self, image, text, audio):
        # image: bx3x2224x224 text: bx100x77 audio: bx1x320000
        # img: bx512 txt: bx512 aud: bx473x512

        img = self.vis_scaler(self.vl_model.encode_image(image))
        txt = self.txt_scaler(self.vl_model.encode_text(text))
        aud = self.aud_scaler(self.a_model(audio))
        aud = reduce(aud, 'b l f -> b f', reduction='mean')

        # normalized features
        # img = img / img.norm(dim=1, keepdim=True)
        # txt = txt / txt.norm(dim=1, keepdim=True)
        # aud = aud / aud.norm(dim=1, keepdim=True)

        # cal logits
        logits_img_txt = self.logit_scale1 * ((img @ txt.t() + 1) / 2)
        logits_txt_img = logits_img_txt.t()

        logits_txt_aud = self.logit_scale2 * ((aud @ txt.t() + 1) / 2)
        logits_aud_txt = logits_txt_aud.t()

        logits = (logits_img_txt + logits_txt_aud) / 2
        logits_t = (logits_txt_img + logits_aud_txt) / 2

        return logits, logits_t
