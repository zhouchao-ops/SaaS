import math
import warnings
from typing import List, Optional, Tuple, Union
import os
import io
import torch
import gc
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from huggingface_hub import snapshot_download

from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers import Phi3Config, Phi3Model
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.utils import logging
from transformers.models.phi3.modeling_phi3 import Phi3Attention,apply_rotary_pos_emb,repeat_kv,Phi3DecoderLayer

import torch.nn.functional as F


from scipy.ndimage import gaussian_filter

import pickle
logger = logging.get_logger(__name__)
from .utils import intermediate_vars, frac_factors, masks

def get_gaussian_kernel2d(kernel_size: int = 7, sigma: float = 1.0, device='cpu'):
    """生成二维高斯核（用于模糊）"""
    coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    kernel1d = g / g.sum()

    kernel2d = kernel1d[:, None] @ kernel1d[None, :]
    kernel2d = kernel2d / kernel2d.sum()
    return kernel2d.to(device)

def apply_gaussian_blur_torch(attn_map: torch.Tensor, sigma: float = 1.0):
    """
    用纯 torch 实现的高斯模糊。
    输入 shape: [H, W]，返回相同 shape。
    """
    kernel_size = int(2 * round(3 * sigma) + 1)
    kernel = get_gaussian_kernel2d(kernel_size, sigma, device=attn_map.device).to(attn_map.dtype)
    kernel = kernel.view(1, 1, kernel_size, kernel_size)

    attn_map = attn_map.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    blurred = F.conv2d(attn_map, kernel, padding=kernel_size // 2)
    return blurred.squeeze(0).squeeze(0)  # [H, W]


def attn2mask(attn, thresh=0.0, y=0):
    """归一化并二值化注意力图"""
    for _ in range(y):
        attn = attn**2
        attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
    attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)
    mask = attn >= thresh
    return mask  # 已是 torch.bool 类型
import numpy as np
import torch
import cv2
def attn2mask_Otsu(attn, y=0):
    """
    输入:
        attn: attention map，Tensor 或 ndarray（支持 2D）
        y: 可选的幂次增强轮数
    输出:
        mask: 二值掩码，Torch Tensor (bool类型)
    """
    if isinstance(attn, torch.Tensor):
        attn = attn.to(torch.float32).detach().cpu().numpy()

    # 幂次增强（可选）
    for _ in range(y):
        attn = attn**2
        attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

    # 归一化
    attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

    # 将 float32 attention map 转成 8-bit 灰度图（0~255）
    attn_8bit = (attn * 255).astype(np.uint8)

    # 用 Otsu 方法自动寻找阈值
    otsu_thresh, mask = cv2.threshold(attn_8bit, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 转为 bool 类型 mask
    mask = mask.astype(bool)

    return torch.tensor(mask)

def store_attn_mask(attn_weight, thresh=0.4):
    thresh = os.getenv("MASK_THRESH","auto")

    conds = intermediate_vars['conds']
    res = intermediate_vars['res']
    spilt_idx = intermediate_vars['spilt_idx']
    H = int(res**0.5)

    attn_sum_image = attn_weight[..., conds[0][0]:conds[0][1]].sum(dim=-1).view(H, H)
    for i in range(len(spilt_idx)-1):
        if i == 0:
            attn_weight_sum = attn_weight[...,spilt_idx[i]:spilt_idx[i+1]].sum(dim=-1)
        else:
            attn_weight_sum = attn_weight[...,spilt_idx[i]+2:spilt_idx[i+1]].sum(dim=-1)
        attn_weight_sum = attn_weight_sum.view(H, H)
        text_attn_blur = apply_gaussian_blur_torch(attn_weight_sum, sigma=1.0)
        if thresh == "auto":
            mask = attn2mask_Otsu(text_attn_blur).to(text_attn_blur.device)
        else:
            mask = attn2mask(text_attn_blur, thresh=float(thresh))

        masks[i] = mask
        
        sum_text = torch.masked_select(attn_weight_sum,mask).sum()
        sum_image = torch.masked_select(attn_sum_image,mask).sum()
        frac_factor = sum_image/(sum_text+1e-6)
        frac_factors[i] = frac_factor

def attn_rescale(attn_weights,scale=1.0):

    conds = intermediate_vars['conds']
    res = conds[0][1]-conds[0][0]

    
    mask_type = os.getenv("MASK_TYPE", "step")

    if mask_type == "step":

        spilt_idx = intermediate_vars['spilt_idx']
        for i in range(len(spilt_idx) - 1):
            start = spilt_idx[i] + (0 if i == 0 else 2)
            end = spilt_idx[i + 1]
            text_attns = attn_weights[0, :, -res:, start:end]
            
            frac_factor = frac_factors[i]
            mask = masks[i]

            temp = text_attns.view(*text_attns.shape[:-2], int(res**0.5), int(res**0.5), text_attns.shape[-1])
            if mask.shape != temp.shape:
                mask_exp = mask.unsqueeze(0).unsqueeze(-1).expand_as(temp).to(temp.device)
            else:
                mask_exp = mask.to(temp.device)

            temp = temp * (~mask_exp) + temp * mask_exp * scale * frac_factor
            attn_weights[0, :, -res:, start:end] = temp.view_as(text_attns)
    # 归一化
    sums = attn_weights.sum(dim=-1,keepdim=True)
    attn_weights = attn_weights / sums
    return attn_weights



class Omni_Phi3Attention(Phi3Attention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        bsz, q_len, _ = hidden_states.size()

        qkv = self.qkv_proj(hidden_states)
        query_pos = self.num_heads * self.head_dim
        query_states = qkv[..., :query_pos]
        key_states = qkv[..., query_pos : query_pos + self.num_key_value_heads * self.head_dim]
        value_states = qkv[..., query_pos + self.num_key_value_heads * self.head_dim :]

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = self.rotary_emb(value_states, position_ids, seq_len=kv_seq_len)

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        
        scale = None
        scale_factor = 1 / math.sqrt(query_states.size(-1)) if scale is None else scale
        attn_weights = query_states @ key_states.transpose(-2, -1) * scale_factor

        cur_step = os.getenv("CUR_STEP", 0)
        cur_step = int(cur_step)
        begin_step = int(os.getenv("BEGIN_STEP", 0))
        thre_step = os.getenv("THRE_STEP", 50)
        thre_step = int(thre_step)


        if attention_mask is not None:
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights += causal_mask


        scale_layers = os.getenv("SCALE_LAYER", "-1")
        scale_layers = [int(i) for i in scale_layers.split(',')]
        scale = float(os.getenv("SCALE", 1.0))

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)


        if cur_step <= thre_step and self.layer_idx in scale_layers and cur_step > begin_step:

            attn_weights = attn_rescale(attn_weights,scale=scale)
        
        attn_output = attn_weights @ value_states
        if cur_step < thre_step and self.layer_idx >=8 and cur_step >= begin_step:
            res = intermediate_vars['res']
            attn_weights = attn_weights[0].mean(dim = 0)
            intermediate_vars['attn_weight'] += attn_weights[-res:,...]
            

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

            
        # if not output_attentions:
        #     attn_weights = None
        # if output_attentions:
        #     attn_weights = attn_weights[0].mean(dim = 0)
        #     attn_weights = attn_weights.detach().cpu()
        return attn_output, None, past_key_value


class Omni_Phi3DecoderLayer(Phi3DecoderLayer):
    def __init__(self, config: Phi3Config,layer_idx:int):
        super().__init__(config,layer_idx=layer_idx)
        self.config = config
        self.self_attn  = Omni_Phi3Attention(config,layer_idx=layer_idx)

class Omni_Phi3Model(Phi3Model):
    def __init__(self, config: Phi3Config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Omni_Phi3DecoderLayer(config,layer_idx=i) for i in range(config.num_hidden_layers)]
        )

class Phi3Transformer(Omni_Phi3Model):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Phi3DecoderLayer`]
    We only modified the attention mask
    Args:
        config: Phi3Config
    """
    def prefetch_layer(self, layer_idx: int, device: torch.device):
        "Starts prefetching the next layer cache"
        with torch.cuda.stream(self.prefetch_stream):
            # Prefetch next layer tensors to GPU
            for name, param in self.layers[layer_idx].named_parameters():
                param.data = param.data.to(device, non_blocking=True)

    def evict_previous_layer(self, layer_idx: int):
        "Moves the previous layer cache to the CPU"
        prev_layer_idx = layer_idx - 1
        for name, param in self.layers[prev_layer_idx].named_parameters():
            param.data = param.data.to("cpu", non_blocking=True)
            
    def get_offlaod_layer(self, layer_idx: int, device: torch.device):
        # init stream
        if not hasattr(self, "prefetch_stream"):
            self.prefetch_stream = torch.cuda.Stream()

        # delete previous layer
        torch.cuda.current_stream().synchronize()
        self.evict_previous_layer(layer_idx)
        
        # make sure the current layer is ready
        torch.cuda.synchronize(self.prefetch_stream)

        # load next layer
        self.prefetch_layer((layer_idx + 1) % len(self.layers), device)
        

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        offload_model: Optional[bool] = False,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        cur_step = int(os.getenv("CUR_STEP", 0))
        begin_step = int(os.getenv("BEGIN_STEP", 0))
        thre_step = int(os.getenv("THRE_STEP", 50))
        # if cur_step >= begin_step and cur_step < thre_step:
        #     output_attentions = True
        output_attentions = False
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # kept for BC (non `Cache` `past_key_values` inputs)
        return_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            return_legacy_cache = True
            if past_key_values is None:
                past_key_values = DynamicCache()
            else:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
                logger.warning_once(
                    "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                    "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                    "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
                )

        # if inputs_embeds is None:
        #     inputs_embeds = self.embed_tokens(input_ids)

        # if cache_position is None:
        #     past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        #     cache_position = torch.arange(
        #         past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        #     )
        # if position_ids is None:
        #     position_ids = cache_position.unsqueeze(0)

        if attention_mask is not None and attention_mask.dim() == 3:
            dtype = inputs_embeds.dtype
            min_dtype = torch.finfo(dtype).min
            attention_mask = (1 - attention_mask) * min_dtype
            attention_mask = attention_mask.unsqueeze(1).to(inputs_embeds.dtype)
        else:
            raise Exception("attention_mask parameter was unavailable or invalid")
            # causal_mask = self._update_causal_mask(
            #     attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
            # )

        hidden_states = inputs_embeds

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        layer_idx = -1

        intermediate_vars['attn_weight'] = 0
        for decoder_layer in self.layers:
            layer_idx += 1

            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                )
            else:
                if offload_model and not self.training:
                    self.get_offlaod_layer(layer_idx, device=inputs_embeds.device)
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
                )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            print('************')
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if return_legacy_cache:
            next_cache = next_cache.to_legacy_cache()

        cur_step = int(os.getenv("CUR_STEP", 0))
        thre_step = int(os.getenv("THRE_STEP", 50))
        if cur_step >= begin_step and cur_step < thre_step:
            ## caculate mask
            # all_hidden_states_cond = [all_self_attn for all_self_attn in all_self_attns]
            # attn_weight = torch.stack(all_hidden_states_cond[8:], dim=0)
            # attn_weight = attn_weight.mean(dim=0)
            attn_weight = intermediate_vars['attn_weight']
            store_attn_mask(attn_weight)
        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )

