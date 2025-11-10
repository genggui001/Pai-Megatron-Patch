# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
import os
import torch
import torch
import json
import gzip
import pickle
import tempfile
import ahocorasick
import dataclasses
import sys
import traceback

from typing import Any, Dict, List, Optional, Tuple, Union

import webdataset as wds

from megatron.energon import DefaultTaskEncoder, Batch, TextSample, Sample, SkipSample, stateless
from megatron.energon.task_encoder.cooking import Cooker, basic_sample_keys


import torch.nn.functional as F

from transformers import AutoTokenizer


def find_subsequences(data: list[int], start_flag: list[int], end_flag: list[int]) -> list[tuple[int, int]]:
    """
    优化版本：在data中查找所有以start_flag开始，end_flag结束的子序列的位置
    采用非贪婪匹配，并使用KMP思想提高效率
    
    参数:
        data: 主数据列表
        start_flag: 开始标志序列
        end_flag: 结束标志序列
        
    返回:
        包含(开始位置, 结束位置)的列表，其中结束位置是end_flag最后一个元素的位置+1
    """
    results = []
    data_len = len(data)
    start_len = len(start_flag)
    end_len = len(end_flag)
    
    if not start_len or not end_len or data_len < start_len + end_len:
        return results
    
    # 预处理函数：查找所有start_flag的出现位置
    def find_all_patterns(pattern, text):
        positions = []
        m, n = len(pattern), len(text)
        if m == 0 or n < m:
            return positions
            
        # 计算失效函数 (KMP算法的一部分)
        lps = [0] * m
        j = 0
        for i in range(1, m):
            while j > 0 and pattern[j] != pattern[i]:
                j = lps[j-1]
            if pattern[j] == pattern[i]:
                j += 1
            lps[i] = j
            
        # 搜索模式
        j = 0
        for i in range(n):
            while j > 0 and pattern[j] != text[i]:
                j = lps[j-1]
            if pattern[j] == text[i]:
                j += 1
            if j == m:
                positions.append(i - m + 1)
                j = lps[j-1]
                
        return positions
    
    # 找到所有start_flag出现的位置
    start_positions = find_all_patterns(start_flag, data)
    
    for start_pos in start_positions:
        # 从start_flag之后开始查找第一个end_flag
        search_start = start_pos + start_len
        
        # 如果剩余空间不足以包含end_flag，跳过此start_pos
        if search_start > data_len - end_len:
            continue
            
        # 在剩余区域中查找end_flag
        j = 0
        for i in range(search_start, data_len):
            if data[i] == end_flag[j]:
                j += 1
                if j == end_len:
                    # 找到完整的end_flag
                    end_pos = i + 1  # 包含end_flag的最后一个元素
                    results.append((start_pos, end_pos))
                    break
            else:
                # 如果匹配中断，重置j并从当前位置继续
                j = 0
                # 优化：如果数据与end_flag的第一个元素匹配，重新检查
                if data[i] == end_flag[0]:
                    j = 1
    
    return results




@dataclasses.dataclass
class SFTSample(Sample):
    platform: dict
    messages: List[dict]
    

@dataclasses.dataclass
class EncodedSample:
    __key__: str
    __restore_key__: Tuple[Union[str, int, tuple], ...]

    input_ids: torch.Tensor
    position_ids: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int

    # 输出
    labels: Optional[torch.Tensor]



@dataclasses.dataclass
class TrainBatch(Batch):
    # __keys__: List[str]

    input_ids: torch.Tensor
    position_ids: torch.Tensor
    attention_mask: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seqlen: int

    # 输出
    labels: Optional[torch.Tensor]
    loss_mask: Optional[torch.Tensor]


class MyTaskEncoder(
    DefaultTaskEncoder[
        Union[SFTSample, TextSample], 
        EncodedSample, 
        TrainBatch, 
        dict
    ]
):
    def __init__(
            self,
            tokenizer_path,
            seq_length,
            sensitive_words_path = None,
            ignore_decoder_errors = False,
            # video_max_pixels = 360 * 420,
            # video_fps = 1.0,
        ):
        self.cookers = [
            Cooker(self.cook_sft, has_subflavors={"data": "data"}),
            Cooker(self.cook_pretrain, has_subflavors={"gztext": "gztext"}),
        ]
        self.ignore_decoder_errors = ignore_decoder_errors

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, model_max_length=seq_length)
        self.pad_token_id = self.tokenizer.pad_token_id
        
        # self.no_think_start_flag = self.tokenizer("<|im_start|>assistant\n<think>\n\n</think>\n\n").input_ids 
        # self.think_start_flag = self.tokenizer("<|im_start|>assistant\n<think>\n").input_ids
        # self.end_flag = self.tokenizer('<|im_end|>').input_ids
        self.no_think_start_flag = [151644, 77091, 198, 151667, 271, 151668, 271]
        self.think_start_flag = [151644, 77091, 198, 151667, 198]
        self.end_flag = [151645]
        # self.start_offset = 2
        self.english_reasoning_prompt = "\n\n\nUse English in your thinking process."
        
        #  tokenizer deal
        self.train_max_len = seq_length
        # self.train_max_vision_length = vision_seq_length

        # self.sft_role_mapper = {
        #     "to user": "<|aim_start|>",
        #     "from terminal": "<|fim_start|>",
        # }

        # self.base_role_mapper = {
        #     "from user": "User:\n",
        #     "to user": "Assistant:\n",
        #     "from terminal": "Function:\n",
        # }

        self.sensitive_word_a = None
        if sensitive_words_path is not None:
            with open(sensitive_words_path, "r", encoding="utf8") as f:
                sensitive_words = json.load(f) 

            sensitive_word_a = ahocorasick.Automaton()
            for idx, key in enumerate(sensitive_words):
                sensitive_word_a.add_word(key, (idx, key))
            sensitive_word_a.make_automaton()
            print(sensitive_words_path + " load sensitive words success.")

        super().__init__()
    
    def _decode_error_handler(self, exc: Exception) -> bool:
        if self.ignore_decoder_errors:
            return True
        raise exc
    
    @stateless
    def cook_pretrain(self, sample):
        d_sample = gzip.decompress(sample['gztext']).decode("utf8")

        return TextSample(
            **basic_sample_keys(sample),
            text=d_sample
        )
    
    @stateless
    def cook_sft(self, sample):
        d_sample = pickle.loads(gzip.decompress(sample['data']))

        return SFTSample(
            **basic_sample_keys(sample),
            platform=d_sample['platform'],
            messages=d_sample['messages'],
        )
    
    @stateless
    def encode_sample(self, sample: Union[SFTSample, TextSample]):
        if isinstance(sample, SFTSample):
            input_modality = sample.platform.get("input_modality", None)
            if input_modality == 'text':
                yield from self.encode_text_sft_iter(sample, is_reasoning=False)
            elif input_modality == 'same_language_reasoning':
                yield from self.encode_text_sft_iter(sample, is_reasoning=True)
            elif input_modality == 'english_reasoning':
                # 特殊处理思考过程+英文指令
                sample.platform['content'] = (sample.platform['content'] + self.english_reasoning_prompt).strip()
                yield from self.encode_text_sft_iter(sample, is_reasoning=True)
            else:
                raise NotImplementedError('Sample format not supported')
        elif isinstance(sample, TextSample):
            yield self.encode_text(sample)

    def _deal_sft_text(
        self,
        text,
    ):
        # text修复
        text = text.replace("</s>", "\n").strip()
        has_sensitive_word = False
        if (
            self.sensitive_word_a is not None 
            and len(list(self.sensitive_word_a.iter_long(text))) > 0
        ):
            has_sensitive_word = True

        # 合并对象
        return text, has_sensitive_word
            

    def encode_text_sft_iter(self, sample: SFTSample, is_reasoning: bool = False):

        system = sample.platform.get("content", "")
        if len(system) > 0:
            used_messages = [{
                "role": "system", 
                "content": system
            }]
        else:
            used_messages = []
            
        for message in sample.messages:
            # 非推理数据删除所有思考过程
            if not is_reasoning:
                message['reasoning_content'] = None
                
            used_messages.append(message)

            if message['role'] != 'assistant':
                continue

            # 处理 input
            build_text = self.tokenizer.apply_chat_template(
                used_messages,
                tools=sample.platform.get("tools", []),
                tokenize=False,
            )
            
            build_text, has_sensitive_word = self._deal_sft_text(
                build_text,
            )

            if has_sensitive_word:
                # raise SkipSample()
                # print(f"skip {sample.__key__}")
                continue
                    
            input_ids = self.tokenizer(build_text).input_ids
            labels = [-100 for _ in input_ids]


            if is_reasoning:
                reasoning_content = message.get("reasoning_content", None)
                if reasoning_content is not None and len(reasoning_content) > 0:
                    # 处理思考过程
                    for start, end in find_subsequences(input_ids, self.think_start_flag, self.end_flag):
                        # train include <think>\n
                        for idx in range(start + len(self.think_start_flag) - 2, end):
                            labels[idx] = input_ids[idx]
                else:
                    # 推理数据没思考过程跳过
                    continue
            else:
                for start, end in find_subsequences(input_ids, self.no_think_start_flag, self.end_flag):
                    for idx in range(start + len(self.no_think_start_flag), end):
                        labels[idx] = input_ids[idx]

            input_ids = torch.LongTensor([input_ids])
            labels = torch.LongTensor([labels[1:] + [-100]])

            # 长度裁剪
            if input_ids.shape[-1] > self.train_max_len:
                input_ids = input_ids[:, :self.train_max_len]
                labels = labels[:, :self.train_max_len]

            # 无label直接返回
            if torch.any(labels >= 0) == False:
                # raise SkipSample()
                # print(f"skip {sample.__key__}")
                continue

            if input_ids.dtype == torch.float32:
                # print(f"skip {sample.__key__}")
                # raise SkipSample()
                continue
                
            yield EncodedSample(
                __key__=sample.__key__,
                __restore_key__=sample.__restore_key__,

                input_ids=input_ids,
                position_ids=torch.arange(input_ids.shape[-1], device=input_ids.device)[None, :],
                cu_seqlens=torch.LongTensor([0, input_ids.shape[-1]]),
                max_seqlen=input_ids.shape[-1],

                labels=labels,
            )
            


    def encode_text(self, sample: TextSample):
        if (
            self.sensitive_word_a is not None 
            and len(list(self.sensitive_word_a.iter_long(sample.text))) > 0
        ):
            raise SkipSample()

        if len(sample.text) == 0:
            raise SkipSample()
        
        build_inputs = self.tokenizer(sample.text)
        input_ids = torch.LongTensor([build_inputs['input_ids']])
        labels = torch.LongTensor([build_inputs['input_ids'][1:] + [-100]])

        # 长度裁剪
        if input_ids.shape[-1] > self.train_max_len:
            input_ids = input_ids[:, :self.train_max_len]
            labels = labels[:, :self.train_max_len]

        if input_ids.dtype == torch.float32:
            raise SkipSample()

        return EncodedSample(
            __key__=sample.__key__,
            __restore_key__=sample.__restore_key__,

            input_ids=input_ids,
            position_ids=torch.arange(input_ids.shape[-1], device=input_ids.device)[None, :],
            cu_seqlens=torch.LongTensor([0, input_ids.shape[-1]]),
            max_seqlen=input_ids.shape[-1],

            labels=labels,
        )
    
    def select_samples_to_pack(self, samples: List[EncodedSample]) -> List[List[EncodedSample]]:
        # 统一合并逻辑
        group_seq_lens = [0]

        group_samples: List[List[EncodedSample]] = [[]]

        for sample in samples:
            
            tmp_len = sample.input_ids.shape[-1]
            found_insert = False

            for idx in range(len(group_seq_lens)):

                if (
                    group_seq_lens[idx] + tmp_len <= self.train_max_len
                ):
                    group_seq_lens[idx] += tmp_len
                    group_samples[idx].append(sample)

                    found_insert = True
                    break

            if found_insert == False:
                group_seq_lens.append(tmp_len)
                group_samples.append([sample])

        # 最后一个可能为空
        if group_seq_lens[-1] == 0:
            group_seq_lens.pop(-1)
            group_samples.pop(-1)

        return group_samples
    
    @stateless
    def pack_selected_samples(self, group_samples: List[EncodedSample]) -> EncodedSample:
        pack_input_ids = torch.concat(tuple(sample.input_ids for sample in group_samples), dim=-1)
        pack_labels = torch.concat(tuple(sample.labels for sample in group_samples), dim=-1)
        pack_position_ids = torch.concat(tuple(sample.position_ids for sample in group_samples), dim=-1)

        # 处理长度问题
        pack_max_seqlen = max(sample.max_seqlen for sample in group_samples)
        pack_cu_seqlens = [
            torch.LongTensor([0])
        ]
        tmp_len = 0
        for sample in group_samples:
            pack_cu_seqlens.append(
                sample.cu_seqlens[1:] + tmp_len
            )
            tmp_len += sample.cu_seqlens[-1].item()
        pack_cu_seqlens = torch.concat(pack_cu_seqlens, dim=-1)

    
        return EncodedSample(
            __key__=",".join([sample.__key__ for sample in group_samples]),
            __restore_key__=(),

            input_ids=pack_input_ids,
            position_ids=pack_position_ids,
            cu_seqlens=pack_cu_seqlens,
            max_seqlen=pack_max_seqlen,

            labels=pack_labels,
        )


    def batch(self, samples: List[EncodedSample]):
        assert len(samples) == 1

        sample = samples[0]


        return TrainBatch(
            __key__=[sample.__key__],
            __restore_key__=(),

            input_ids=sample.input_ids,
            position_ids=sample.position_ids,
            attention_mask=torch.ones_like(sample.input_ids),
            cu_seqlens=sample.cu_seqlens.to(torch.int32),
            max_seqlen=sample.max_seqlen,

            # 输出
            labels=sample.labels,
            loss_mask=(sample.labels >= 0).float(),
        )
        

    def encode_batch(self, batch: TrainBatch) -> dict:
        raw = dataclasses.asdict(batch)
        return raw



def print_error_handler(exc: Exception, key: Optional[str]):
    print(
        f"The following exception occurred in the dataloader for sample {key} and is skipped",
        file=sys.stderr,
    )
    traceback.print_exc()

