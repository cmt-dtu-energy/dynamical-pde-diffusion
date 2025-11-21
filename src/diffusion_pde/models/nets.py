import torch
import torch.nn.functional as F


def init_weights(module, mode="kaiming_normal", zero_bias=True, nonlinearity="linear"):
    if not isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Linear)):
        return
    if mode == "kaiming_normal":
        torch.nn.init.kaiming_normal_(module.weight, a=0, mode="fan_in", nonlinearity=nonlinearity)
        if zero_bias and module.bias is not None:
            torch.nn.init.zeros_(module.bias)
        elif module.bias is not None:
            torch.nn.init.kaiming_normal_(module.bias.unsqueeze(1), a=0, mode="fan_in", nonlinearity=nonlinearity)
    elif mode == "kaiming_uniform":
        torch.nn.init.kaiming_uniform_(module.weight, a=0, mode="fan_in", nonlinearity=nonlinearity)
        if zero_bias and module.bias is not None:
            torch.nn.init.zeros_(module.bias)
        elif module.bias is not None:
            torch.nn.init.kaiming_uniform_(module.bias.unsqueeze(1), a=0, mode="fan_in", nonlinearity=nonlinearity)
    elif mode == "zeros":
        torch.nn.init.zeros_(module.weight)
        if module.bias is not None:
            torch.nn.init.zeros_(module.bias)
    else:
        raise ValueError(f"Unknown initialization mode: {mode}")


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
    

def get_conv_layer(
    in_ch: int, 
    out_ch: int, 
    kernel_size: int, 
    up: bool = False,
    down: bool = False,
    init_mode: str = "kaiming_normal",
):

    padding = max(0, (kernel_size - 1) // 2)
    if up:
        layer = torch.nn.ConvTranspose2d(in_ch, out_ch, kernel_size, stride=2, padding=padding, output_padding=1, padding_mode="zeros")
    elif down:
        layer = torch.nn.Conv2d(in_ch, out_ch, kernel_size, stride=2, padding=padding, padding_mode="reflect")
    else:
        layer = torch.nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, padding_mode="reflect")
    init_weights(layer, mode=init_mode)
    return layer


class ResBlock(torch.nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        emb_ch: int,
        up: bool = False,
        down: bool = False,
        dropout: float = 0.0,
        skip_scale: float = 2 ** -0.5,
    ):
        super().__init__()
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.skip_scale = skip_scale
        self.up = up
        self.down = down

        # Norms
        gn1_groups = 32 if in_ch >= 32 and in_ch % 32 == 0 else in_ch
        gn2_groups = 32 if out_ch >= 32 and out_ch % 32 == 0 else out_ch
        self.norm1 = torch.nn.GroupNorm(gn1_groups, in_ch)
        self.norm2 = torch.nn.GroupNorm(gn2_groups, out_ch)

        self.act = torch.nn.SiLU()

        # Convs
        self.conv1 = get_conv_layer(in_ch, out_ch, 3, up=up, down=down)
        self.conv2 = get_conv_layer(out_ch, out_ch, 3, init_mode="zeros")

        # Embedding → out_ch
        self.emb_layer = torch.nn.Linear(emb_ch, out_ch)
        init_weights(self.emb_layer)

        # Skip path
        self.skip = None
        if in_ch != out_ch or up or down:
            self.skip = get_conv_layer(in_ch, out_ch, 1, up=up, down=down)

        self.dropout = torch.nn.Dropout(dropout)


    def forward(self, x, emb):
        """
        x:   [B, C_in, H, W]
        emb: [B, emb_ch]
        """
        orig = x
        x = self.conv1(self.act(self.norm1(x)))

        # Broadcast embedding and add
        e = self.emb_layer(emb).unsqueeze(-1).unsqueeze(-1)  # [B, out_ch, 1, 1]
        x = x + e

        x = self.conv2(self.dropout(self.act(self.norm2(x))))
    
        x = x + (self.skip(orig) if self.skip is not None else orig)

        return x * self.skip_scale

# -----------------------------
# Small EDM-style UNet
# -----------------------------

class EDMUNet(torch.nn.Module):
    """
    EDM-style UNet, scaled down to be < 10M params for typical choices.

    - Conditioning: concatenates obs at input (x and obs same spatial size).
    - Embedding: sigma + label combined and fed into all ResBlocks.
    """

    def __init__(
        self,
        img_channels: int,      # number of channels of x
        obs_channels: int = 0,      # number of channels of obs
        label_dim: int = 0,     # dimension of labels (0 = no labels)
        base_channels: int = 64,
        channel_mults=(1, 2, 2),   # e.g. [1,2,2] or [1,2,2,2]
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        sigma_emb_dim: int = 64,
        emb_dim: int = 256,
        debug: bool = False,
    ):
        super().__init__()
        self.debug = debug
        self.img_channels = img_channels
        self.obs_channels = obs_channels
        in_channels = img_channels + obs_channels

        # --- time/noise embedding ---
        self.sigma_embed = PositionalEmbedding(sigma_emb_dim)
        init_weights(self.sigma_embed)

        self.time_mlp = torch.nn.Sequential(
            torch.nn.Linear(sigma_emb_dim, emb_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(emb_dim, emb_dim),
        )
        for m in self.time_mlp:
            init_weights(m)

        # --- label embedding (optional) ---
        if label_dim > 0:
            self.label_embed = torch.nn.Linear(label_dim, emb_dim)
            init_weights(self.label_embed)
        else:
            self.label_embed = None

        # ------------- Encoder -------------
        self.enc = torch.nn.ModuleList()
        ch = base_channels
        
        ch_list = []  # track feature dims for skip sizes
        for i, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            if i == 0:
                self.enc.append(get_conv_layer(in_channels, out_ch, 3))
            else:
                self.enc.append(ResBlock(ch, out_ch, emb_dim, down=True, dropout=dropout))
            ch = out_ch
            ch_list.append(ch)
            for _ in range(num_res_blocks):
                self.enc.append(ResBlock(ch, out_ch, emb_dim, dropout=dropout))
                ch = out_ch
                ch_list.append(ch)


        # ------------- Decoder -------------
        self.dec = torch.nn.ModuleList()
        # We'll iterate channel_mults in reverse
        for i, mult in reversed(list(enumerate(channel_mults))):
            #out_ch = base_channels * mult
            if i == len(channel_mults) - 1:
                self.dec.append(ResBlock(out_ch, out_ch, emb_dim, dropout=dropout))
                self.dec.append(ResBlock(out_ch, out_ch, emb_dim, dropout=dropout))
            else:
                self.dec.append(ResBlock(out_ch, out_ch, emb_dim, up=True, dropout=dropout))
            for _ in range(num_res_blocks + 1):  # +1 to consume skips properly
                # cat with skip features from encoder
                in_ch = out_ch + ch_list.pop()
                out_ch = base_channels * mult
                self.dec.append(ResBlock(in_ch, out_ch, emb_dim, dropout=dropout))
        
        self.final_block = torch.nn.Sequential(
            torch.nn.GroupNorm(num_channels=out_ch, num_groups=32 if (out_ch % 32) == 0 else out_ch),
            get_conv_layer(out_ch, img_channels, 3, init_mode="zeros")
        )

        if self.debug:
            total_params = sum(p.numel() for p in self.parameters())
            print(f"SmallEDMUNet params: {total_params/1e6:.2f}M")

    def forward(self, x, sigma, labels=None, obs=None):
        """
        x:      [B, Cx, H, W]   (noisy state)
        obs:    [B, Co, H, W]   (initial condition / BCs)
        sigma:  [B] or [B,1]    (noise level)
        labels: [B, label_dim]  (optional)
        """
        # concat conditioning at input
        if obs is not None and self.obs_channels > 0:
            assert obs.shape[1] == self.obs_channels, f"Expected obs with {self.obs_channels} channels, got {obs.shape[1]}"
            x = torch.cat([x, obs], dim=1)

        # build embedding
        emb_sigma = self.sigma_embed(sigma)        # [B, sigma_emb_dim]
        emb = self.time_mlp(emb_sigma)             # [B, emb_dim]
        if self.label_embed is not None and labels is not None:
            emb = emb + self.label_embed(labels)   # combine as in EDM/DhariwalUNet

        skips = []
        # Encoder
        for i, block in enumerate(self.enc):
            x = block(x, emb) if isinstance(block, ResBlock) else block(x)
            skips.append(x)

        # Decoder
        for i, block in enumerate(self.dec):
            if x.shape[1] != block.in_channels:
                x = torch.cat([x, skips.pop()], dim=1)
            x = block(x, emb)

        # Final conv block
        x = self.final_block(x)
    
        return x


class EDMWrapper(torch.nn.Module):
    def __init__(self, 
        unet: torch.nn.Module,
        sigma_data: float = 0.5,
    ):
        super().__init__()
        self.unet = unet
        self.sigma_data = sigma_data

    def forward(self, x: torch.Tensor, sigma: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        # x has shape (b, ch, h, w) and sigma has shape (b,)
        # both should be dtype float32 for now
        sigma = torch.reshape(sigma, (-1, 1, 1, 1))

        # weights given by the EDM paper.
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)
        c_in = 1 / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)
        c_noise = torch.flatten(torch.log(sigma) / 4).to(torch.float32)

        F_x = self.unet(c_in * x, c_noise, *args, **kwargs)   # output of u-net
        D_x = c_skip * x + c_out * F_x          # denoised data

        return D_x