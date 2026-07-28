import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import Swin_B_Weights, swin_b
from transformer import TwoWayTransformer, LayerNorm2d

attribute_categories = {
            "Hairstyle category": 12,
            "Hair curliness": 3,
            "Hair length": 3,
            "Hair volume": 3,
            "Hair quality": 3,
            "Hair color": 6,
            "Bangs": 4,
        }

class HairXFormerMLP(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, 256)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        hidden_states = self.proj(hidden_states)
        return hidden_states


class PositionEmbeddingRandom(nn.Module):
    def __init__(self, num_pos_feats: int = 64, scale: float = 1.0) -> None:
        super().__init__()
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            scale * torch.randn((2, num_pos_feats)),
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        coords = 2 * coords - 1
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * torch.pi * coords
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: tuple) -> torch.Tensor:
        h, w = size
        device = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)


class HairDecoder(nn.Module):
    def __init__(self, transformer_dim=256, transformer=None, attribute_categories=None):
        super().__init__()
        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.attribute_categories = attribute_categories

        self.mask_token = nn.Embedding(1, transformer_dim)
        self.attribute_tokens = nn.Embedding(len(attribute_categories), transformer_dim)

        # Mask Prediction Head
        self.mask_prediction_head = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),
            LayerNorm2d(transformer_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(transformer_dim // 4, 1, kernel_size=2, stride=2),
        )

        # Attribute Prediction Heads
        self.attribute_heads = nn.ModuleDict({
            key: nn.Sequential(
                nn.Linear(transformer_dim, 128),
                nn.ReLU(),
                nn.Linear(128, num_classes)
            )
            for key, num_classes in attribute_categories.items()
        })

    def forward(self, image_embeddings: torch.Tensor, image_pe: torch.Tensor):
        # Initialize tokens
        mask_token = self.mask_token.weight.unsqueeze(0).expand(image_embeddings.size(0), -1, -1)
        attribute_tokens = self.attribute_tokens.weight.unsqueeze(0).expand(image_embeddings.size(0), -1, -1)
        tokens = torch.cat([mask_token, attribute_tokens], dim=1)

        # Pass through Transformer
        hs, src = self.transformer(image_embeddings, image_pe, tokens)
        return src



class HairXFormer(nn.Module):
    def __init__(self):
        super(HairXFormer, self).__init__()

        # Backbone: Swin Transformer
        swin_v2 = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*(list(swin_v2.children())[:-1]))
        self.multi_scale_features = []

        hidden_sizes = [128, 256, 512, 1024]
        decoder_hidden_size = 256

        # Multi-scale Feature Mapping
        self.linear_c = nn.ModuleList([HairXFormerMLP(input_dim=h) for h in hidden_sizes])

        # Feature Fusion
        self.linear_fuse = nn.Conv2d(
            in_channels=decoder_hidden_size * len(hidden_sizes),
            out_channels=decoder_hidden_size,
            kernel_size=1,
            bias=False,
        )

        # Positional Embedding
        self.pe_layer = PositionEmbeddingRandom(decoder_hidden_size // 2)

        # Register forward hooks for Swin Transformer layers
        for name, module in self.backbone.named_modules():
            if name in ['0.1', '0.3', '0.5', '0.7']:
                module.register_forward_hook(self.save_features_hook(name))

        # Hair Decoder
        self.hair_decoder = HairDecoder(
            transformer_dim=decoder_hidden_size,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=decoder_hidden_size,
                mlp_dim=2048,
                num_heads=8,
            ),
            attribute_categories=attribute_categories
        )

    def save_features_hook(self, name):
        def hook(module, input, output):
            self.multi_scale_features.append(output.permute(0, 3, 1, 2).contiguous())
        return hook

    def forward_encoder(self, x):
        self.multi_scale_features.clear()
        _ = self.backbone(x)

        batch_size = self.multi_scale_features[-1].shape[0]
        all_hidden_states = []

        for encoder_hidden_state, mlp in zip(self.multi_scale_features, self.linear_c):
            height, width = encoder_hidden_state.shape[2:]
            encoder_hidden_state = mlp(encoder_hidden_state)
            encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
            encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, height, width)
            encoder_hidden_state = nn.functional.interpolate(
                encoder_hidden_state, size=self.multi_scale_features[0].size()[2:], mode="bilinear"
            )
            all_hidden_states.append(encoder_hidden_state)

        fused_states = self.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))

        image_pe = self.pe_layer((fused_states.shape[2], fused_states.shape[3])).unsqueeze(0)
        src = self.hair_decoder(
            image_embeddings=fused_states,
            image_pe=image_pe
        )
        return src

    def forward(self, x):
        return self.forward_encoder(x)
