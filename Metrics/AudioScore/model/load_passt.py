import torch.nn as nn

from .passt.passt import get_model
from .passt.preprocess import AugmentMelSTFT


class MyPasst(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.mel = AugmentMelSTFT(n_mels=128,
                                  sr=32000,
                                  win_length=800,
                                  hopsize=320,
                                  n_fft=1024,
                                  freqm=48,
                                  timem=192,
                                  htk=False,
                                  fmin=0.0,
                                  fmax=None,
                                  norm=1,
                                  fmin_aug_range=10,
                                  fmax_aug_range=2000)
        self.net = get_model(arch="passt_s_swa_p16_128_ap476",
                             pretrained=True,
                             n_classes=0,
                             in_channels=1,
                             fstride=10,
                             tstride=10,
                             input_fdim=128,
                             input_tdim=998,
                             u_patchout=0,
                             s_patchout_t=40,
                             s_patchout_f=4)
        self.dyn_norm = False
        self.linear = nn.Linear(in_features=768, out_features=512)

    def mel_forward(self, x):
        old_shape = x.size()
        x = x.reshape(-1, old_shape[2])
        x = self.mel(x)
        x = x.reshape(old_shape[0], old_shape[1], x.shape[1], x.shape[2])
        # if self.dyn_norm:
        #     if not hasattr(self, "tr_m") or not hasattr(self, "tr_std"):
        #         tr_m, tr_std = get_dynamic_norm(self)
        #         self.register_buffer('tr_m', tr_m)
        #         self.register_buffer('tr_std', tr_std)
        #     x = (x - self.tr_m) / self.tr_std
        return x

    def forward(self, x):
        x = self.mel_forward(x)
        embed = self.net(x)
        # feature downsample
        embed = self.linear(embed)
        return embed

    def freeze(self):
        for _, p in self.mel.named_parameters():
            p.requires_grad = False
        for _, p in self.net.named_parameters():
            p.requires_grad = False
