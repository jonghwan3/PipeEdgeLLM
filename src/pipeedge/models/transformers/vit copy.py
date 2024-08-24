"""ViT Transformers."""
from collections.abc import Mapping
import logging
import math
import os
from typing import Optional, Union
import numpy as np
import requests
import torch
from torch import nn
from transformers import ViTConfig
from transformers.models.vit.modeling_vit import (
    ViTEmbeddings, ViTIntermediate, ViTOutput, ViTSelfAttention, ViTSelfOutput
)
from .. import ModuleShard, ModuleShardConfig
from . import TransformerShardData
import torch.nn.functional as F
import types

logger = logging.getLogger(__name__)

_WEIGHTS_URLS = {
    'google/vit-base-patch16-224': 'https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-B_16-224.npz',
    'google/vit-large-patch16-224': 'https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-L_16-224.npz',
    'google/vit-huge-patch14-224-in21k': 'https://storage.googleapis.com/vit_models/imagenet21k/ViT-H_14.npz',
}

PRUNED_SELF = False
PRUNED_LAST = False

class ViTLayerShard(ModuleShard):
    """Module shard based on `ViTLayer`."""

    def __init__(self, config: ViTConfig, shard_config: ModuleShardConfig):
        super().__init__(config, shard_config)
        self.layernorm_before = None
        self.self_attention = None
        self.self_output = None
        self.layernorm_after = None
        self.intermediate = None
        self.output = None
        self._build_shard()

    def _build_shard(self):
        if self.has_layer(0):
            self.layernorm_before = nn.LayerNorm(self.config.hidden_size,
                                                 eps=self.config.layer_norm_eps)
            self.self_attention = ViTSelfAttention(self.config)
        if self.has_layer(1):
            self.self_output = ViTSelfOutput(self.config)
        if self.has_layer(2):
            self.layernorm_after = nn.LayerNorm(self.config.hidden_size,
                                                eps=self.config.layer_norm_eps)
            self.intermediate = ViTIntermediate(self.config)
        if self.has_layer(3):
            self.output = ViTOutput(self.config)
    


    @torch.no_grad()
    def forward(self, data: TransformerShardData) -> TransformerShardData:
        """Compute layer shard."""
        if self.has_layer(0):
            data_norm = self.layernorm_before(data)
            data = (self.self_attention(data_norm)[0], data)
        if self.has_layer(1):
            skip = data[1]
            data = self.self_output(data[0], skip)
            data += skip
        if self.has_layer(2):
            data_norm = self.layernorm_after(data)
            data = (self.intermediate(data_norm), data)
        if self.has_layer(3):
            data = self.output(data[0], data[1])
        return data
    
    def forward_pruned(self, data: TransformerShardData) -> TransformerShardData:
        """Compute layer shard."""
        if self.has_layer(0):
            data_norm = self.layernorm_before(data)
            data = (self.self_attention(data_norm)[0], data)
        if self.has_layer(1):
            skip = data[1]
            data = self.self_output(data[0], skip)
            data += skip
        if self.has_layer(2):
            data_norm = self.layernorm_after(data)
            data = (self.intermediate(data_norm), data)
        if self.has_layer(3):
            data = self.output(data[0], data[1])
        return data

class ViTModelShard(ModuleShard):
    """Module shard based on `ViTModel` (no pooling layer)."""

    def __init__(self, config: ViTConfig, shard_config: ModuleShardConfig,
                 model_weights: Union[str, Mapping]):
        super().__init__(config, shard_config)
        self.embeddings = None
        # ViTModel uses an encoder here, but we'll just add the layers here instead.
        # Since we just do inference, a ViTEncoderShard class wouldn't provide real benefit.
        self.layers = nn.ModuleList()
        self.layernorm = None

        logger.debug(">>>> Model name: %s", self.config.name_or_path)
        if isinstance(model_weights, str):
            logger.debug(">>>> Load weight file: %s", model_weights)
            with np.load(model_weights) as weights:
                self._build_shard(weights)
        else:
            self._build_shard(model_weights)

    def _build_shard(self, weights):
        if self.shard_config.is_first:
            logger.debug(">>>> Load embeddings layer for the first shard")
            self.embeddings = ViTEmbeddings(self.config)
            self._load_weights_first(weights)

        layer_curr = self.shard_config.layer_start
        while layer_curr <= self.shard_config.layer_end:
            layer_id = math.ceil(layer_curr / 4) - 1
            sublayer_start = (layer_curr - 1) % 4
            if layer_id == math.ceil(self.shard_config.layer_end / 4) - 1:
                sublayer_end = (self.shard_config.layer_end - 1) % 4
            else:
                sublayer_end = 3
            logger.debug(">>>> Load layer %d, sublayers %d-%d",
                         layer_id, sublayer_start, sublayer_end)
            layer_config = ModuleShardConfig(layer_start=sublayer_start, layer_end=sublayer_end)
            layer = ViTLayerShard(self.config, layer_config)
            self._load_weights_layer(weights, layer_id, layer)
            self.layers.append(layer)
            layer_curr += sublayer_end - sublayer_start + 1

        if self.shard_config.is_last:
            logger.debug(">>>> Load layernorm for the last shard")
            self.layernorm = nn.LayerNorm(self.config.hidden_size, eps=self.config.layer_norm_eps)
            self._load_weights_last(weights)

    @torch.no_grad()
    def _load_weights_first(self, weights):
        self.embeddings.cls_token.copy_(torch.from_numpy(weights["cls"]))
        self.embeddings.position_embeddings.copy_(torch.from_numpy((weights["Transformer/posembed_input/pos_embedding"])))
        conv_weight = weights["embedding/kernel"]
        # O, I, J, K = conv_weight.shape
        # conv_weight = conv_weight.reshape(K,J,O,I)
        conv_weight = conv_weight.transpose([3, 2, 0, 1])
        self.embeddings.patch_embeddings.projection.weight.copy_(torch.from_numpy(conv_weight))
        self.embeddings.patch_embeddings.projection.bias.copy_(torch.from_numpy(weights["embedding/bias"]))

    @torch.no_grad()
    def _load_weights_last(self, weights):
        self.layernorm.weight.copy_(torch.from_numpy(weights["Transformer/encoder_norm/scale"]))
        self.layernorm.bias.copy_(torch.from_numpy(weights["Transformer/encoder_norm/bias"]))



    @torch.no_grad()
    def _load_weights_layer(self, weights, layer_id, layer, pruned=PRUNED_SELF):
        root = f"Transformer/encoderblock_{layer_id}/"
        hidden_size = self.config.hidden_size
        if layer.has_layer(0):
            layer.layernorm_before.weight.copy_(torch.from_numpy(weights[root + "LayerNorm_0/scale"]))
            layer.layernorm_before.bias.copy_(torch.from_numpy(weights[root + "LayerNorm_0/bias"]))
            if(pruned):
                with np.load('vit_layers_snip_pruned.npz') as weights_pruned:
                    for i, layer in enumerate(self.layers):
                        # print(1)
                        layer.self_attention.query.weight.copy_(torch.from_numpy(weights_pruned['vit.layer{}.self_attention.query.weight'.format(i)]))
                        layer.self_attention.query.bias.copy_(torch.from_numpy(weights_pruned['vit.layer{}.self_attention.query.bias'.format(i)]))
                        layer.self_attention.key.weight.copy_(torch.from_numpy(weights_pruned['vit.layer{}.self_attention.key.weight'.format(i)]))
                        layer.self_attention.key.bias.copy_(torch.from_numpy(weights_pruned['vit.layer{}.self_attention.key.bias'.format(i)]))
                        layer.self_attention.value.weight.copy_(torch.from_numpy(weights_pruned['vit.layer{}.self_attention.value.weight'.format(i)]))
                        layer.self_attention.value.bias.copy_(torch.from_numpy(weights_pruned['vit.layer{}.self_attention.value.bias'.format(i)]))
            else:
                layer.self_attention.query.weight.copy_(torch.from_numpy(weights[root + "MultiHeadDotProductAttention_1/query/kernel"]).view(hidden_size, hidden_size).t())
                layer.self_attention.key.weight.copy_(torch.from_numpy(weights[root + "MultiHeadDotProductAttention_1/key/kernel"]).view(hidden_size, hidden_size).t())
                layer.self_attention.value.weight.copy_(torch.from_numpy(weights[root + "MultiHeadDotProductAttention_1/value/kernel"]).view(hidden_size, hidden_size).t())
                layer.self_attention.query.bias.copy_(torch.from_numpy(weights[root + "MultiHeadDotProductAttention_1/query/bias"]).view(-1))
                layer.self_attention.key.bias.copy_(torch.from_numpy(weights[root + "MultiHeadDotProductAttention_1/key/bias"]).view(-1))
                layer.self_attention.value.bias.copy_(torch.from_numpy(weights[root + "MultiHeadDotProductAttention_1/value/bias"]).view(-1))
        if layer.has_layer(1):
            layer.self_output.dense.weight.copy_(torch.from_numpy(weights[root + "MultiHeadDotProductAttention_1/out/kernel"]).view(hidden_size, hidden_size).t())
            layer.self_output.dense.bias.copy_(torch.from_numpy(weights[root + "MultiHeadDotProductAttention_1/out/bias"]).view(-1))
        if layer.has_layer(2):
            layer.layernorm_after.weight.copy_(torch.from_numpy(weights[root + "LayerNorm_2/scale"]))
            layer.layernorm_after.bias.copy_(torch.from_numpy(weights[root + "LayerNorm_2/bias"]))
            layer.intermediate.dense.weight.copy_(torch.from_numpy(weights[root + "MlpBlock_3/Dense_0/kernel"]).t())
            layer.intermediate.dense.bias.copy_(torch.from_numpy(weights[root + "MlpBlock_3/Dense_0/bias"]).t())
        if layer.has_layer(3):
            layer.output.dense.weight.copy_(torch.from_numpy(weights[root + "MlpBlock_3/Dense_1/kernel"]).t())
            layer.output.dense.bias.copy_(torch.from_numpy(weights[root + "MlpBlock_3/Dense_1/bias"]).t())

    @torch.no_grad()
    def forward(self, data: TransformerShardData) -> TransformerShardData:
        """Compute shard layers."""
        if self.shard_config.is_first:
            data = self.embeddings(data)
        for layer in self.layers:
            data = layer(data)
        if self.shard_config.is_last:
            data = self.layernorm(data)
        return data

    @staticmethod
    def save_weights(model_name: str, model_file: str, url: Optional[str]=None,
                     timeout_sec: Optional[float]=None) -> None:
        """Save the model weights file."""
        if url is None:
            url = _WEIGHTS_URLS[model_name]
        logger.info('Downloading model: %s: %s', model_name, url)
        req = requests.get(url, stream=True, timeout=timeout_sec)
        req.raise_for_status()
        with open(model_file, 'wb') as file:
            for chunk in req.iter_content(chunk_size=8192):
                if chunk:
                    file.write(chunk)
                    file.flush()
                    os.fsync(file.fileno())


class ViTShardForImageClassification(ModuleShard):
    """Module shard based on `ViTForImageClassification`."""

    def __init__(self, config: ViTConfig, shard_config: ModuleShardConfig,
                 model_weights: Union[str, Mapping]):
        super().__init__(config, shard_config)
        self.vit = None
        self.classifier = None

        logger.debug(">>>> Model name: %s", self.config.name_or_path)
        if isinstance(model_weights, str):
            logger.debug(">>>> Load weight file: %s", model_weights)
            with np.load(model_weights) as weights:
                self._build_shard(weights)
        else:
            self._build_shard(model_weights)

    def _build_shard(self, weights):
        ## all shards use the inner ViT model
        self.vit = ViTModelShard(self.config, self.shard_config, weights)

        if self.shard_config.is_last:
            logger.debug(">>>> Load classifier for the last shard")
            self.classifier = nn.Linear(self.config.hidden_size, self.config.num_labels) if self.config.num_labels > 0 else nn.Identity()
            self._load_weights_last(weights)

    @torch.no_grad()
    def _load_weights_last(self, weights, pruned=PRUNED_LAST):
        if(pruned):
            with np.load('vit_snip_pruned.npz') as weights_pruned:
                # print(1)
                self.classifier.weight.copy_(torch.from_numpy(weights_pruned['classifier.weight']))
                self.classifier.bias.copy_(torch.from_numpy(weights_pruned['classifier.bias']))
        else:
            self.classifier.weight.copy_(torch.from_numpy(np.transpose(weights["head/kernel"])))
            self.classifier.bias.copy_(torch.from_numpy(weights["head/bias"]))

    @torch.no_grad()
    def forward(self, data: TransformerShardData) -> TransformerShardData:
        """Compute shard layers."""
        data = self.vit(data)
        if self.shard_config.is_last:
            data = self.classifier(data[:, 0, :])
        return data
    
    def prune_snip_transformer(self, ubatch, ubatch_labels):
        ## layers' masks
        for layer in self.vit.layers:
            layer.self_attention.query.weight_mask = nn.Parameter(torch.ones_like(layer.self_attention.query.weight))
            layer.self_attention.key.weight_mask = nn.Parameter(torch.ones_like(layer.self_attention.key.weight))
            layer.self_attention.value.weight_mask = nn.Parameter(torch.ones_like(layer.self_attention.value.weight))
            layer.self_attention.query.forward = types.MethodType(snip_forward_linear, layer.self_attention.query)
            layer.self_attention.key.forward = types.MethodType(snip_forward_linear, layer.self_attention.key)
            layer.self_attention.value.forward = types.MethodType(snip_forward_linear, layer.self_attention.value)

        self.zero_grad()

        # forward
        data = self.vit.embeddings(ubatch)
        for layer in self.vit.layers:
            data = layer.forward_pruned(data)
        data = self.vit.layernorm(data)
        data = self.classifier(data[:, 0, :])

        loss = F.nll_loss(data, ubatch_labels)
        loss.backward()
        
        print(loss)
        print(self.vit.layers[0].self_attention.query.weight.grad)
        weights = {}
        for i, layer in enumerate(self.vit.layers):
            #TODO : whole model
            keep_masks = snip_keep_masks(layer.self_attention.query.weight_mask.grad)
            layer.self_attention.query.weight.data[keep_masks[0] == 0.] = 0.
            weights['vit.layer{}.self_attention.query.weight'.format(i)] = layer.self_attention.query.weight.detach().numpy()
            weights['vit.layer{}.self_attention.query.bias'.format(i)] = layer.self_attention.query.bias.detach().numpy()

            keep_masks = snip_keep_masks(layer.self_attention.key.weight_mask.grad)
            layer.self_attention.key.weight.data[keep_masks[0] == 0.] = 0.
            weights['vit.layer{}.self_attention.key.weight'.format(i)] = layer.self_attention.key.weight.detach().numpy()
            weights['vit.layer{}.self_attention.key.bias'.format(i)] = layer.self_attention.key.bias.detach().numpy()

            keep_masks = snip_keep_masks(layer.self_attention.value.weight_mask.grad)
            layer.self_attention.value.weight.data[keep_masks[0] == 0.] = 0.
            weights['vit.layer{}.self_attention.value.weight'.format(i)] = layer.self_attention.value.weight.detach().numpy()
            weights['vit.layer{}.self_attention.value.bias'.format(i)] = layer.self_attention.value.bias.detach().numpy()
        #TODO : print sparsity/density
        model_file = 'vit_layers_snip_pruned.npz'
        np.savez(model_file, **weights)



    # self.vit.layers[0].self_attention.query.forward = types.MethodType(snip_forward_linear, self.)
    def prune_snip(self, ubatch, ubatch_labels):
        self.classifier.weight_mask = nn.Parameter(torch.ones_like(self.classifier.weight)) 
        self.classifier.forward = types.MethodType(snip_forward_linear, self.classifier)
        self.zero_grad()
        #forward
        ubatch = self.vit(ubatch)
        ubatch = self.classifier(ubatch[:, 0, :])
        loss = F.nll_loss(ubatch, ubatch_labels)
        loss.backward()
        print(loss)
        print(self.classifier.weight_mask.grad)
        grads_abs = []
        grads_abs.append(torch.abs(self.classifier.weight_mask.grad))
        all_scores = torch.cat([torch.flatten(x) for x in grads_abs])
        norm_factor = torch.sum(all_scores)
        all_scores.div_(norm_factor)
        keep_ratio = 0.0001
        num_params_to_keep = int(len(all_scores) * keep_ratio)
        threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
        acceptable_score = threshold[-1]
        keep_masks = []
        for g in grads_abs:
            keep_masks.append(((g / norm_factor) >= acceptable_score).float())
        print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks])))
        self.classifier.weight.data[keep_masks[0] == 0.] = 0.
        model_file = 'vit_snip_pruned.npz'
        weights = {}
        weights['classifier.weight'] = self.classifier.weight.detach().numpy()
        weights['classifier.bias'] = self.classifier.bias.detach().numpy()
        # print(1)
        np.savez(model_file, **weights)

        # with np.load('vit_snip_pruned.npz') as weights_pruned:
        #     print(1)
        #     self.classifier.weight.copy_(torch.from_numpy(weights_pruned['classifier.weight']))
        #     self.classifier.bias.copy_(torch.from_numpy(weights_pruned['classifier.bias']))

# def save_weights(model_name: str, model_file: str) -> None:
#     """Save the model weights file."""
#     model = BertForSequenceClassification.from_pretrained(model_name)
#     state_dict = model.state_dict()
#     weights = {}
#     for key, val in state_dict.items():
#         weights[key] = val
#     np.savez(model_file, **weights)


# @staticmethod
# def save_weights(model_name: str, model_file: str, url: Optional[str]=None,
#                     timeout_sec: Optional[float]=None) -> None:
#     """Save the model weights file."""
#     if url is None:
#         url = _WEIGHTS_URLS[model_name]
#     logger.info('Downloading model: %s: %s', model_name, url)
#     req = requests.get(url, stream=True, timeout=timeout_sec)
#     req.raise_for_status()
#     with open(model_file, 'wb') as file:
#         for chunk in req.iter_content(chunk_size=8192):
#             if chunk:
#                 file.write(chunk)
#                 file.flush()
#                 os.fsync(file.fileno())

    
    @staticmethod
    def save_weights(model_name: str, model_file: str, url: Optional[str]=None,
                     timeout_sec: Optional[float]=None) -> None:
        """Save the model weights file."""
        ViTModelShard.save_weights(model_name, model_file, url=url, timeout_sec=timeout_sec)

def snip_forward_linear(self, x):
    return F.linear(x, self.weight * self.weight_mask, self.bias)

def snip_keep_masks(grad, keep_ratio=0.9):
    grads_abs = []
    grads_abs.append(grad)
    all_scores = torch.cat([torch.flatten(x) for x in grads_abs])
    norm_factor = torch.sum(all_scores)
    all_scores.div_(norm_factor)
    num_params_to_keep = int(len(all_scores) * keep_ratio)
    threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
    acceptable_score = threshold[-1]
    keep_masks = []
    for g in grads_abs:
        keep_masks.append(((g / norm_factor) >= acceptable_score).float())
    return keep_masks
