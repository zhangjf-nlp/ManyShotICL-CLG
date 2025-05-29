import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, Union, List

from transformers import Cache, DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaModel, LlamaPreTrainedModel, LlamaForCausalLM, \
    LlamaRMSNorm


class LlamaConfigForPrefix(LlamaConfig):
    def __init__(self, num_p=10, **kwargs):
        super().__init__(**kwargs)
        self.num_p = num_p


class MLP(nn.Module):
    def __init__(self, input_size, intermediate_size, output_size, output_norm=1.0, bias=False):
        super().__init__()
        self.gate_proj = nn.Linear(input_size, intermediate_size, bias=bias)
        self.in_proj = nn.Linear(input_size, intermediate_size, bias=bias)
        self.out_proj = nn.Linear(intermediate_size, output_size, bias=bias)
        self.act_fn = nn.SiLU()
        self.ln = LlamaRMSNorm(output_size, eps=1e-6)
        self.ln.weight.data.fill_(output_norm / np.sqrt(output_size))

    def forward(self, x):
        return self.ln(self.out_proj(self.act_fn(self.gate_proj(x)) * self.in_proj(x)))


class LlamaForPrefixTuning(LlamaPreTrainedModel):
    config_class = LlamaConfigForPrefix
    main_input_name = "input_ids"
    is_parallelizable = False

    def __init__(self, config: LlamaConfigForPrefix):
        super().__init__(config)
        self.num_p = self.config.num_p
        self.embed_dim = config.hidden_size
        self.model = LlamaModel(config)
        self.model.requires_grad_(False)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.lm_head.requires_grad_(False)
        self.prefix_embedding = nn.Embedding(self.num_p, self.embed_dim)
        self.print_once = True
        self.step = 0
        self.nll_loss = True

    def get_prefix_embeds(self, batch_size=1):
        prefix_indices = torch.arange(self.num_p).to(self.device)
        prefix_embeds = self.prefix_embedding(prefix_indices)
        return prefix_embeds.unsqueeze(0).repeat(batch_size, 1, 1)

    def convert_right_to_left_padding(self, input_ids, attention_mask):
        for mask in attention_mask:
            zero_indices = (mask == 0).nonzero(as_tuple=True)[0]
            if zero_indices.numel() > 0:
                first_zero_idx = zero_indices[0]
                if torch.any(mask[first_zero_idx:] == 1):
                    raise AssertionError("The attention_mask is not strictly right padded.")
        encoder_input_ids = torch.zeros_like(input_ids)
        for i, (ids, mask) in enumerate(zip(input_ids, attention_mask)):
            non_padding_ids = ids[mask == 1]
            encoder_input_ids[i, -len(non_padding_ids):] = non_padding_ids
        encoder_attention_mask = torch.zeros_like(attention_mask)
        for i, mask in enumerate(attention_mask):
            non_padding_len = torch.sum(mask == 1).item()
            encoder_attention_mask[i, -non_padding_len:] = 1
        return encoder_input_ids, encoder_attention_mask

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None, # [batch_size, seq_len]
        attention_mask: Optional[torch.FloatTensor] = None, # [batch_size, seq_len]
        labels: Optional[torch.LongTensor] = None, # [batch_size, seq_len]
        return_dict: bool = None,
        **kwargs,
    ):
        if self.training:
            self.step += 1

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        batch_size, seq_len = input_ids.shape # [batch_size, seq_len]

        inputs_embeds = self.model.embed_tokens(input_ids).detach()
        prefix_embeds = self.get_prefix_embeds(batch_size=batch_size)
        inputs_embeds = torch.cat([prefix_embeds, inputs_embeds], dim=1)

        prefix_attention_mask = attention_mask.new_ones(size=(batch_size, self.num_p))
        attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)

        transformer_outputs = self.model.forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=return_dict,
            **kwargs
        )
        hidden_states = transformer_outputs[0]
        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            if self.nll_loss:
                # Negative Log-Likelihood
                loss = F.cross_entropy(
                    input=lm_logits[:, self.num_p:-1, :].contiguous().transpose(1, 2),
                    target=labels[:, 1:].contiguous(),
                    reduction='none'
                ).sum(dim=-1)
            else:
                # token-avg lm_loss
                loss = F.cross_entropy(
                    input=lm_logits[:, self.num_p:-1, :].contiguous().transpose(1, 2),
                    target=labels[:, 1:].contiguous(),
                    reduction='mean'
                )
            loss = loss.mean()

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
        )

    def merge_and_save(self, lm: LlamaForCausalLM, save_directory: str, start_idx=None):
        if start_idx is None:
            start_idx = lm.vocab_size - self.num_p
        prefix_merge_idx = torch.arange(self.num_p, device=self.device) + start_idx
        lm.model.embed_tokens.weight.data[prefix_merge_idx, :] = self.prefix_embedding.weight.data
        lm.save_pretrained(save_directory)