import torch

class PositionalEmbedding(torch.nn.Module):
    def __init__(self, num_channels, max_positions=10000, endpoint=False):
        super().__init__()
        self.num_channels = num_channels
        self.max_positions = max_positions
        self.endpoint = endpoint

    def forward(self, x):
        freqs = torch.arange(start=0, end=self.num_channels//2, dtype=torch.float32, device=x.device)
        freqs = freqs / (self.num_channels // 2 - (1 if self.endpoint else 0))
        freqs = (1 / self.max_positions) ** freqs
        x = x.ger(freqs.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

class Unet(torch.nn.Module):
    '''
    Unet taken from deep learning course.
    '''
    
    def __init__(
        self,
        chs: list[int], # list of channels including input channel size: (ch_in, ch_1, ..., ch_n), length n+1
        label_ch: int, # label dimension (class label/ time etc)
        noise_ch: int = 32, # embedding channel size 
        act_fn: torch.nn.Module = torch.nn.SiLU,
        debug: bool = False
    ):
        super().__init__()
        self.act_fn = act_fn
        self.debug = debug

        self.down_conv = torch.nn.ModuleList()
        self.up_conv = torch.nn.ModuleList()

        # Construct down-sampling blocks
        for i in range(len(chs) - 1):
            block = torch.nn.ModuleList()
            if i != 0:
                block.append(
                    torch.nn.MaxPool2d(kernel_size=2, stride=2)               
                )
            block.extend(
                (torch.nn.Conv2d(chs[i], chs[i+1], kernel_size=3, padding=1),
                self.act_fn())
            )
            self.down_conv.append(torch.nn.Sequential(*block))

        # Construct up-sampling blocks
        for i in range(len(chs) - 1, 0, -1):
            block = torch.nn.ModuleList()
            if i == len(chs) - 1:
                layer = torch.nn.ConvTranspose2d(chs[i], chs[i-1], kernel_size=3, stride=2, padding=1, output_padding=1)
            elif i == 1:
                layer = torch.nn.ConvTranspose2d(chs[i] * 2, chs[i], kernel_size=3, padding=1)  
            else:
                layer = torch.nn.ConvTranspose2d(chs[i] * 2, chs[i-1], kernel_size=3, stride=2, padding=1, output_padding=1)
            block.extend((layer, self.act_fn()))
            if i == 1:
                block.append(torch.nn.Conv2d(chs[i], chs[i-1], kernel_size=3, padding=1))
            self.up_conv.append(torch.nn.Sequential(*block))


        self.sigma_embedding = PositionalEmbedding(noise_ch)
        self.linear_label = torch.nn.Linear(label_ch, noise_ch)

        self.linear_embed = torch.nn.ModuleList([
            torch.nn.Linear(noise_ch, chs[i]) for i in range(1, len(chs), 1)
        ])

    def forward(self, x: torch.Tensor, sigma: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        #assume x has shape (b, ch, h, w) and t has shape (b, label_ch)

        emb = self.sigma_embedding(sigma)
        if t.ndim == 1:
            t = t[:, None]
        label_emb = self.linear_label(t)
        emb = emb + label_emb
        embs = [self.linear_embed[i](emb) for i in range(len(self.linear_embed))]

        signal = x
        signals = []
        for i, conv in enumerate(self.down_conv):
            signal = conv(signal)
            signal = signal  + embs[i][..., None, None]
            if i < len(self.down_conv) - 1:
                signals.append(signal)

            if self.debug: print(f"Down conv {i}: {signal.shape}")

        for i, tconv in enumerate(self.up_conv):
            if i == 0:
                signal = tconv(signal)
            else:
                signal = torch.cat((signal, signals[-i]), dim=-3)
                signal = tconv(signal)
            if i < len(self.up_conv) - 1:
                signal = signal + embs[-i-2][..., None, None]
                
            if self.debug: print(f"Up conv {i}: {signal.shape}")
        return signal
    

class EDMWrapper(torch.nn.Module):
    def __init__(self, 
        unet: torch.nn.Module,
        sigma_data: float = 0.5,
    ):
        super().__init__()
        self.unet = unet
        self.sigma_data = sigma_data

    def forward(self, x: torch.Tensor, sigma: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x has shape (b, ch, h, w) and sigma has shape (b,)
        # both should be dtype float32 for now
        sigma = torch.reshape(sigma, (-1, 1, 1, 1))

        # weights given by the EDM paper.
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)
        c_in = 1 / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)
        c_noise = torch.flatten(torch.log(sigma) / 4).to(torch.float32)

        F_x = self.unet(c_in * x, c_noise, t)   # output of u-net
        D_x = c_skip * x + c_out * F_x          # denoised data

        return D_x