import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple, Union, List

from transformers import Cache, DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.qwen2.modeling_qwen2 import Qwen2Config, Qwen2Model, Qwen2PreTrainedModel, Qwen2ForCausalLM, Qwen2RMSNorm


class Qwen2ConfigForPrefix(Qwen2Config):
    def __init__(self, num_p=4, **kwargs):
        super().__init__(**kwargs)
        self.num_p = num_p


class Qwen2ForPrefixTuning(Qwen2PreTrainedModel):
    config_class = Qwen2ConfigForPrefix
    main_input_name = "input_ids"
    is_parallelizable = False
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: Qwen2ConfigForPrefix):
        super().__init__(config)
        self.num_p = self.config.num_p
        self.embed_dim = config.hidden_size
        self.model = Qwen2Model(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.model.requires_grad_(False)
        self.lm_head.requires_grad_(False)
        self.prefix_embedding = nn.Embedding(self.num_p, self.embed_dim)
        self.dropout_rate = 0.0

        self.print_once = True
        self.step = 0
        self.nll_loss = True

        # Initialize weights and apply final processing
        self.post_init()

    def get_prefix_embeds(self, batch_size=1):
        prefix_indices = torch.arange(self.num_p).to(self.device)
        prefix_embeds = self.prefix_embedding(prefix_indices)
        prefix_embeds = prefix_embeds.unsqueeze(0).repeat(batch_size, 1, 1)
        if self.training and self.dropout_rate > 0.0:
            prefix_embeds = F.dropout(prefix_embeds, self.dropout_rate)
        return prefix_embeds

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None, # [batch_size, seq_len]
        inputs_embeds: Optional[torch.FloatTensor] = None, # [batch_size, seq_len, hidden_size]
        attention_mask: Optional[torch.FloatTensor] = None, # [batch_size, seq_len]
        labels: Optional[torch.LongTensor] = None, # [batch_size, seq_len]
        return_dict: bool = None,
        **kwargs,
    ):
        if self.training:
            self.step += 1

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
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
            **kwargs,
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

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def merge_and_save(self, lm: Qwen2ForCausalLM, save_directory: str, start_idx=None):
        if start_idx is None:
            start_idx = lm.vocab_size - self.num_p
        prefix_merge_idx = torch.arange(self.num_p, device=self.device) + start_idx
        lm.model.embed_tokens.weight.data[prefix_merge_idx, :] = self.prefix_embedding.weight.data
        lm.save_pretrained(save_directory)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        # 如果存在缓存，我们需要调整 input_ids 以仅包含未处理的 tokens
        if past_key_values is not None:
            if inputs_embeds is not None:  # 如果传递了 inputs_embeds，input_ids 可能缺少条目
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # 默认情况
                input_ids = input_ids[:, cache_position]

        # 如果传递了 inputs_embeds，我们只在第一步生成时使用它们
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        # 处理前缀嵌入
        batch_size = input_ids.shape[0]
        if cache_position[0] == 0:  # 只在第一步生成时添加前缀
            prefix_embeds = self.get_prefix_embeds(batch_size=batch_size)
            if model_inputs["inputs_embeds"] is not None:
                model_inputs["inputs_embeds"] = torch.cat([prefix_embeds, model_inputs["inputs_embeds"]], dim=1)
            else:
                inputs_embeds = self.model.embed_tokens(input_ids)
                model_inputs["inputs_embeds"] = torch.cat([prefix_embeds, inputs_embeds], dim=1)
                model_inputs["input_ids"] = None

            cache_position = torch.cat([cache_position, torch.arange(self.num_p).to(self.device)+cache_position[-1]+1], dim=-1)
        else:
            assert cache_position.shape == (1,), cache_position
            cache_position = cache_position + self.num_p
            model_inputs["inputs_embeds"] = self.model.embed_tokens(model_inputs["input_ids"])
            model_inputs["input_ids"] = None

        # 调整 attention_mask 以包含前缀
        prefix_attention_mask = attention_mask.new_ones(size=(batch_size, self.num_p))
        attention_mask = torch.cat([prefix_attention_mask, attention_mask], dim=1)

        # 如果 attention_mask 不为 None 且 position_ids 为 None，则动态创建 position_ids
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # 更新 model_inputs
        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        #print(f"model_inputs: {model_inputs}")
        #print(f"attention_mask.shape: {attention_mask.shape}")
        return model_inputs