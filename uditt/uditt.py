"""quantum_transformers.uditt.uditt
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
A PyTorch implementation of the U-Net-style Diffusion Transformer (UDiT)
as described in papers like UDiTQC (arXiv:2501.16380).

This version is the *classical baseline* and uses standard `nn.MultiheadAttention`.
It serves as the backbone for our hybrid Q-UDiT model.
"""
import torch
import torch.nn as nn
from einops import rearrange
from timm.models.vision_transformer import PatchEmbed, Mlp
import math
from quantum_transformers.quixer.quixer_block import QuixerBlock

# --------------------------------------------------------------------------
# Building Blocks
# --------------------------------------------------------------------------

class TimestepEmbedder(nn.Module):
    """Embeds scalar timesteps into vector representations."""
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)

class LabelEmbedder(nn.Module):
    """Embeds class labels into vector representations."""
    def __init__(self, num_classes, hidden_size, dropout_prob=0.1):
        super().__init__()
        self.embedding_table = nn.Embedding(num_classes, hidden_size)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, labels):
        embeddings = self.embedding_table(labels)
        return self.dropout(embeddings)

class UDiTBlock(nn.Module):
    """A single block of the U-Net Transformer."""
    def __init__(self, hidden_size, num_heads, use_quixer: bool = False, quixer_data_qubits: int = 4, quixer_max_seq_len: int = 256, qsvt_poly_degree: int = 5):
        super().__init__()
        self.use_quixer = use_quixer
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        if use_quixer:
            self.attn = QuixerBlock(
                hidden_size=hidden_size, 
                data_qubits=quixer_data_qubits, 
                max_seq_len=quixer_max_seq_len,
                qsvt_poly_degree=qsvt_poly_degree
            )
        else:
            self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(in_features=hidden_size, hidden_features=int(hidden_size * 4), act_layer=nn.GELU, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        # Adaptive LayerNorm modulation
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        
        # Attention block
        x_norm1 = self.norm1(x)
        x_norm1 = x_norm1 * (1 + scale_msa.unsqueeze(1)) + shift_msa.unsqueeze(1)
        
        if self.use_quixer:
            attn_output = self.attn(x_norm1)
        else:
            attn_output, _ = self.attn(x_norm1, x_norm1, x_norm1)
            
        x = x + gate_msa.unsqueeze(1) * attn_output

        # MLP block
        x_norm2 = self.norm2(x)
        x_norm2 = x_norm2 * (1 + scale_mlp.unsqueeze(1)) + shift_mlp.unsqueeze(1)
        mlp_output = self.mlp(x_norm2)
        x = x + gate_mlp.unsqueeze(1) * mlp_output
        
        return x

# --------------------------------------------------------------------------
# Main UDiT Model
# --------------------------------------------------------------------------

class UDiT(nn.Module):
    """The main U-Net-style Diffusion Transformer model."""
    def __init__(
        self,
        img_size=32,
        patch_size=2,
        in_channels=3,
        hidden_size=768,
        depth=12,
        num_heads=12,
        num_classes=10, # for classifier-free guidance
        context_dim=None,
        quantum_block_indices: list = None,
        quixer_data_qubits: int = 4,
        qsvt_poly_degree: int = 5
    ):
        super().__init__()
        if quantum_block_indices is None:
            quantum_block_indices = []
        
        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, hidden_size))

        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size)

        self.in_blocks = nn.ModuleList([
            UDiTBlock(
                hidden_size, 
                num_heads, 
                use_quixer=(i in quantum_block_indices),
                quixer_data_qubits=quixer_data_qubits,
                quixer_max_seq_len=self.patch_embed.num_patches,
                qsvt_poly_degree=qsvt_poly_degree
            ) for i in range(depth // 2)
        ])
        
        mid_block_idx = depth // 2
        self.mid_block = UDiTBlock(
            hidden_size, 
            num_heads, 
            use_quixer=(mid_block_idx in quantum_block_indices),
            quixer_data_qubits=quixer_data_qubits,
            quixer_max_seq_len=self.patch_embed.num_patches,
            qsvt_poly_degree=qsvt_poly_degree
        )

        self.out_blocks = nn.ModuleList([
            UDiTBlock(
                hidden_size, 
                num_heads, 
                use_quixer=((i + mid_block_idx + 1) in quantum_block_indices),
                quixer_data_qubits=quixer_data_qubits,
                quixer_max_seq_len=self.patch_embed.num_patches,
                qsvt_poly_degree=qsvt_poly_degree
            ) for i in range(depth // 2)
        ])
        
        self.final_norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.final_linear = nn.Linear(hidden_size, patch_size * patch_size * in_channels, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        # Weight initialization for other layers is handled by their default init

    def unpatchify(self, x):
        """(N, T, C) -> (N, C, H, W)"""
        c = self.patch_embed.proj.out_channels
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        return rearrange(x, 'n (h w) (p1 p2 c) -> n c (h p1) (w p2)', p1=p, p2=p, h=h, w=w)

    def forward(self, x, t, y):
        """
        x: (N, C, H, W) tensor of images
        t: (N,) tensor of timesteps
        y: (N,) tensor of class labels
        """
        x = self.patch_embed(x) + self.pos_embed
        t_embed = self.t_embedder(t)
        y_embed = self.y_embedder(y)
        c = t_embed + y_embed

        skips = []
        for block in self.in_blocks:
            x = block(x, c)
            skips.append(x)
        
        x = self.mid_block(x, c)
        
        for block in self.out_blocks:
            x = x + skips.pop() # Basic U-Net skip connection
            x = block(x, c)
            
        x = self.final_norm(x)
        x = self.final_linear(x)
        x = self.unpatchify(x)
        return x
