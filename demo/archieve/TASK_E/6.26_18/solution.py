"""ATEC Task E submission — self-contained ACT (Action Chunking with Transformers).

Single-file, no repository-package imports. The eval container ships only this
file + policy.pt + requirements.txt; server.py calls AlgSolution.predicts(obs, score).

Pipeline (validated in sim):
  * first WARMUP_STEPS predicts() drive the arm q0 -> collection HOME pose (the env
    resets joints to ~0 and demos were collected from HOME), then ACT takes over.
  * state  = obs['proprio'][:8] (joint_pos_rel) + DEFAULT_JOINT_POS -> absolute qpos,
    normalised with the checkpoint's norm_stats.
  * rgb    = obs['image']['video_rgb'] resized to 224 (kept 0..255; the ACT agent does
    /255 + ImageNet norm internally — no double normalisation).
  * action = ACT chunk, denormalised, combined by temporal ensembling.

The embedded model mirrors the training Agent/DETRVAE exactly so the checkpoint's
ema_agent state_dict loads with strict=True.
"""

from __future__ import annotations

import copy
import math
import os
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from torch import nn, Tensor
from torchvision.models._utils import IntermediateLayerGetter

_DIR = os.path.dirname(os.path.abspath(__file__))

# ------------------------------------------------------------------ #
# Task-E constants
# ------------------------------------------------------------------ #
ACTION_DIM = 8
NUM_QUERIES = 30
IMAGE_SIZE = 224
ACTION_SCALE = 0.5  # JointPositionActionCfg(scale=0.5, use_default_offset=True)
DEFAULT_JOINT_POS = (0.0, 1.2, -1.5, 0.0, 1.2, 0.0, 0.035, -0.035)
HOME_JOINT_POS = (0.0, 0.9245, -1.515, 0.0, 1.22, 0.0, 0.035, -0.035)
WARMUP_STEPS = 100
TEMPORAL_M = 0.01


# ================================================================== #
# Embedded ACT model (DETR-style CVAE; module structure / attribute names
# preserved so the trained checkpoint loads strict=True). Class names prefixed `_`.
# ================================================================== #

class _Cfg:
    position_embedding = "sine"
    backbone = "resnet18"
    lr_backbone = 1e-5
    masks = False
    dilation = False
    include_depth = False
    include_rgb = True
    enc_layers = 2
    dec_layers = 4
    dim_feedforward = 512
    hidden_dim = 256
    dropout = 0.1
    nheads = 8
    num_queries = NUM_QUERIES
    pre_norm = False


# ---- position encoding ---- #
class _PositionEmbeddingSine(nn.Module):
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor):
        x = tensor
        not_mask = torch.ones_like(x[0, [0]])
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


def _build_position_encoding(args):
    n_steps = args.hidden_dim // 2
    if args.position_embedding in ("v2", "sine"):
        return _PositionEmbeddingSine(n_steps, normalize=True)
    raise ValueError(f"not supported {args.position_embedding}")


# ---- backbone ---- #
class _FrozenBatchNorm2d(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class _BackboneBase(nn.Module):
    def __init__(self, backbone, train_backbone, num_channels, return_interm_layers):
        super().__init__()
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {"layer4": "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor):
        return self.body(tensor)


class _Backbone(_BackboneBase):
    def __init__(self, name, train_backbone, return_interm_layers, dilation, include_depth):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            weights=None, norm_layer=_FrozenBatchNorm2d)
        num_channels = 512 if name in ("resnet18", "resnet34") else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class _Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list):
        xs = self[0](tensor_list)
        out = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            pos.append(self[1](x).to(x.dtype))
        return out, pos


def _build_backbone(args):
    position_embedding = _build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = args.masks
    backbone = _Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation, args.include_depth)
    model = _Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model


# ---- transformer ---- #
def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class _TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src, src_mask=None, src_key_padding_mask=None, pos=None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class _TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, mask=None, src_key_padding_mask=None, pos=None):
        output = src
        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)
        if self.norm is not None:
            output = self.norm(output)
        return output


class _TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory, tgt_mask=None, memory_mask=None,
                     tgt_key_padding_mask=None, memory_key_padding_mask=None,
                     pos=None, query_pos=None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory, tgt_mask=None, memory_mask=None,
                    tgt_key_padding_mask=None, memory_key_padding_mask=None,
                    pos=None, query_pos=None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                pos=None, query_pos=None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class _TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None,
                pos=None, query_pos=None):
        output = tgt
        intermediate = []
        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        if self.return_intermediate:
            return torch.stack(intermediate)
        return output.unsqueeze(0)


class _Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()
        encoder_layer = _TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                 dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = _TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        decoder_layer = _TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                 dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = _TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                           return_intermediate=return_intermediate_dec)
        self._reset_parameters()
        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, query_embed, pos_embed, latent_input=None,
                proprio_input=None, additional_pos_embed=None):
        if src is None:
            bs = proprio_input.shape[0]
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
            pos_embed = additional_pos_embed.unsqueeze(1).repeat(1, bs, 1)
            src = torch.stack([latent_input, proprio_input], axis=0)
        elif len(src.shape) == 4:
            bs, c, h, w = src.shape
            src = src.flatten(2).permute(2, 0, 1)
            pos_embed = pos_embed.flatten(2).permute(2, 0, 1).repeat(1, bs, 1)
            query_embed = query_embed.unsqueeze(1).repeat(1, bs, 1)
            additional_pos_embed = additional_pos_embed.unsqueeze(1).repeat(1, bs, 1)
            pos_embed = torch.cat([additional_pos_embed, pos_embed], axis=0)
            addition_input = torch.stack([latent_input, proprio_input], axis=0)
            src = torch.cat([addition_input, src], axis=0)
        tgt = torch.zeros_like(query_embed)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask,
                          pos=pos_embed, query_pos=query_embed)
        hs = hs.transpose(1, 2)
        return hs


def _build_transformer(args):
    return _Transformer(
        d_model=args.hidden_dim,
        dropout=args.dropout,
        nhead=args.nheads,
        dim_feedforward=args.dim_feedforward,
        num_encoder_layers=args.enc_layers,
        num_decoder_layers=args.dec_layers,
        normalize_before=args.pre_norm,
        return_intermediate_dec=True,
    )


# ---- DETRVAE ---- #
def _reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = torch.autograd.Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def _get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class _DETRVAE(nn.Module):
    def __init__(self, backbones, transformer, encoder, state_dim, action_dim, num_queries):
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        if backbones is not None:
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
            self.backbones = nn.ModuleList(backbones)
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
        else:
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)
            self.backbones = None
        self.latent_dim = 32
        self.cls_embed = nn.Embedding(1, hidden_dim)
        self.encoder_state_proj = nn.Linear(state_dim, hidden_dim)
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim)
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)
        self.register_buffer("pos_table", _get_sinusoid_encoding_table(1 + 1 + num_queries, hidden_dim))
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)
        self.additional_pos_embed = nn.Embedding(2, hidden_dim)

    def forward(self, obs, actions=None):
        is_training = actions is not None
        state = obs["state"] if self.backbones is not None else obs
        bs = state.shape[0]

        if is_training:
            cls_embed = self.cls_embed.weight
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1)
            state_embed = self.encoder_state_proj(state)
            state_embed = torch.unsqueeze(state_embed, axis=1)
            action_embed = self.encoder_action_proj(actions)
            encoder_input = torch.cat([cls_embed, state_embed, action_embed], axis=1)
            encoder_input = encoder_input.permute(1, 0, 2)
            is_pad = torch.full((bs, encoder_input.shape[0]), False).to(state.device)
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0]
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = _reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(state.device)
            latent_input = self.latent_out_proj(latent_sample)

        if self.backbones is not None:
            vis_data = obs["rgb"]
            if "depth" in obs:
                vis_data = torch.cat([vis_data, obs["depth"]], dim=2)
            num_cams = vis_data.shape[1]
            all_cam_features = []
            all_cam_pos = []
            for cam_id in range(num_cams):
                features, pos = self.backbones[0](vis_data[:, cam_id])
                features = features[0]
                pos = pos[0]
                all_cam_features.append(self.input_proj(features))
                all_cam_pos.append(pos)
            proprio_input = self.input_proj_robot_state(state)
            src = torch.cat(all_cam_features, axis=3)
            pos = torch.cat(all_cam_pos, axis=3)
            hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input,
                                  proprio_input, self.additional_pos_embed.weight)[0]
        else:
            state = self.input_proj_robot_state(state)
            hs = self.transformer(None, None, self.query_embed.weight, None, latent_input,
                                  state, self.additional_pos_embed.weight)[0]

        a_hat = self.action_head(hs)
        return a_hat, [mu, logvar]


def _build_encoder(args):
    d_model = args.hidden_dim
    dropout = args.dropout
    nhead = args.nheads
    dim_feedforward = args.dim_feedforward
    num_encoder_layers = args.enc_layers
    normalize_before = args.pre_norm
    activation = "relu"
    encoder_layer = _TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                             dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    return _TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)


class _Agent(nn.Module):
    """Mirrors the training Agent: state_dict keys are `model.*` so the checkpoint
    ema_agent loads strict=True."""

    def __init__(self, state_dim, act_dim, cfg: _Cfg):
        super().__init__()
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.include_rgb = cfg.include_rgb
        backbones = [_build_backbone(cfg)] if cfg.include_rgb else None
        transformer = _build_transformer(cfg)
        encoder = _build_encoder(cfg)
        self.model = _DETRVAE(backbones, transformer, encoder,
                              state_dim=state_dim, action_dim=act_dim, num_queries=cfg.num_queries)

    def _preprocess_rgb(self, obs: dict) -> None:
        if self.include_rgb and "rgb" in obs:
            obs["rgb"] = obs["rgb"].float() / 255.0
            B, N, C, H, W = obs["rgb"].shape
            obs["rgb"] = self.normalize(obs["rgb"].view(B * N, C, H, W)).view(B, N, C, H, W)

    def _model_input(self, obs: dict):
        return obs if self.include_rgb else obs["state"]

    def get_action(self, obs: dict) -> torch.Tensor:
        self._preprocess_rgb(obs)
        a_hat, _ = self.model(self._model_input(obs))
        return a_hat


# ================================================================== #
# Runtime: preprocessing, temporal ensembling, AlgSolution
# ================================================================== #

def _to_tensor(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device)
    return torch.as_tensor(value, device=device)


def _absolute_state(proprio: Any, device: torch.device) -> torch.Tensor:
    rel = _to_tensor(proprio, device).to(dtype=torch.float32)
    if rel.ndim == 1:
        rel = rel.unsqueeze(0)
    default = torch.tensor(DEFAULT_JOINT_POS, dtype=torch.float32, device=device).view(1, ACTION_DIM)
    return rel[:, :ACTION_DIM] + default


def _prepare_rgb(rgb: Any, device: torch.device) -> torch.Tensor:
    x = _to_tensor(rgb, device)
    if x.ndim == 3:
        x = x.unsqueeze(0)
    if x.ndim != 4:
        raise ValueError(f"rgb must be 3 or 4 dims, got {tuple(x.shape)}")
    if x.shape[-1] in (3, 4):       # BHWC
        x = x[..., :3].permute(0, 3, 1, 2).contiguous()
    elif x.shape[1] in (3, 4):      # BCHW
        x = x[:, :3].contiguous()
    else:
        raise ValueError(f"cannot find rgb channel dim in {tuple(x.shape)}")
    x = x.to(dtype=torch.float32)
    x = T.Resize((IMAGE_SIZE, IMAGE_SIZE), antialias=True)(x)
    return x.unsqueeze(1)  # (B, 1, 3, 224, 224), 0..255 (Agent does /255 + imagenet)


class _TemporalEnsembler:
    """ACT temporal ensembling: every step predicts a fresh chunk; predictions for the
    current step from the last NUM_QUERIES chunks are combined oldest->newest with
    weights exp(-m * i)."""

    def __init__(self, num_queries: int, action_dim: int, m: float, device: torch.device):
        self.num_queries = int(num_queries)
        self.action_dim = int(action_dim)
        self.m = float(m)
        self.device = device
        self.reset()

    def reset(self) -> None:
        self._step = 0
        self._chunks: list[tuple[int, torch.Tensor]] = []

    def add_and_query(self, chunk: torch.Tensor) -> torch.Tensor:
        chunk = chunk.to(device=self.device, dtype=torch.float32)
        self._chunks.append((self._step, chunk))
        oldest_valid = self._step - self.num_queries + 1
        self._chunks = [(s, c) for s, c in self._chunks if s >= oldest_valid]
        preds = []
        for pred_step, c in self._chunks:
            offset = self._step - pred_step
            if 0 <= offset < c.shape[0]:
                preds.append(c[offset])
        stacked = torch.stack(preds, dim=0)
        idx = torch.arange(stacked.shape[0], device=self.device, dtype=torch.float32)
        weights = torch.exp(-self.m * idx).unsqueeze(1)
        action = (stacked * weights).sum(dim=0) / weights.sum().clamp(min=1e-8)
        self._step += 1
        return action


def _find_policy_path() -> str:
    for name in ("policy.pt", "best_loss.pt"):
        p = os.path.join(_DIR, name)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Task E ACT checkpoint not found in {_DIR} (expected policy.pt)")


class AlgSolution:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(_find_policy_path(), map_location=self.device)
        self.norm_stats = {k: v.to(device=self.device, dtype=torch.float32)
                           for k, v in ckpt["norm_stats"].items()}
        state_dim = int(self.norm_stats["state_mean"].shape[-1])
        action_dim = int(self.norm_stats["action_mean"].shape[-1])
        self.agent = _Agent(state_dim, action_dim, _Cfg()).to(self.device)
        weights = ckpt.get("ema_agent") or ckpt["agent"]
        self.agent.load_state_dict(weights, strict=True)
        self.agent.eval()

        # Warmup action: env applies joint_target = default + ACTION_SCALE*action, so
        # (HOME - DEFAULT)/ACTION_SCALE commands the collection HOME pose every step.
        default = torch.tensor(DEFAULT_JOINT_POS, dtype=torch.float32, device=self.device)
        home = torch.tensor(HOME_JOINT_POS, dtype=torch.float32, device=self.device)
        self._warmup_action = ((home - default) / ACTION_SCALE).detach().cpu().tolist()

        self.ensembler = _TemporalEnsembler(NUM_QUERIES, ACTION_DIM, TEMPORAL_M, self.device)
        self._step = 0

    def reset(self, **kwargs) -> None:
        self._step = 0
        self.ensembler.reset()

    def _fallback_action(self, obs) -> list:
        try:
            st = _absolute_state(obs["proprio"], self.device)[0]
            vals = [float(v) for v in st.detach().cpu().tolist()[:ACTION_DIM]]
            if all(math.isfinite(v) for v in vals):
                return vals
        except Exception:
            pass
        return [0.0] * ACTION_DIM

    def predicts(self, obs: dict, current_score: float):
        try:
            if self._step < WARMUP_STEPS:
                self._step += 1
                return {"action": [float(x) for x in self._warmup_action], "giveup": False}

            state = _absolute_state(obs["proprio"], self.device)
            state = (state - self.norm_stats["state_mean"]) / self.norm_stats["state_std"].clamp(min=1e-2)
            rgb = _prepare_rgb(obs["image"]["video_rgb"], self.device)
            with torch.inference_mode():
                chunk = self.agent.get_action({"state": state, "rgb": rgb})
            chunk = chunk[0] * self.norm_stats["action_std"].clamp(min=1e-2) + self.norm_stats["action_mean"]
            action = self.ensembler.add_and_query(chunk)
            self._step += 1
            vals = [float(x) for x in action.detach().cpu().tolist()[:ACTION_DIM]]
            if len(vals) < ACTION_DIM or not all(math.isfinite(v) for v in vals):
                return {"action": self._fallback_action(obs), "giveup": False}
            return {"action": vals, "giveup": False}
        except Exception:
            return {"action": self._fallback_action(obs), "giveup": False}
