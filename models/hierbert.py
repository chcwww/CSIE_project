# from dataclasses import dataclass
# from typing import Optional, Tuple
# from transformers.file_utils import ModelOutput

import torch
import numpy as np
from torch import nn
from transformers import AutoModel

DEBUG = False

# @dataclass
# class SimpleOutput(ModelOutput):
#     last_hidden_state: torch.FloatTensor = None
#     past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
#     hidden_states: Optional[Tuple[torch.FloatTensor]] = None
#     attentions: Optional[Tuple[torch.FloatTensor]] = None
#     cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


def sinusoidal_init(num_embeddings: int, embedding_dim: int):
    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * i / embedding_dim) for i in range(embedding_dim)]
        if pos != 0 else np.zeros(embedding_dim) for pos in range(num_embeddings)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])  # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])  # dim 2i+1
    return torch.from_numpy(position_enc).type(torch.FloatTensor)


class HierarchicalBert(nn.Module):

    def __init__(self, encoder, max_segments=64, max_segment_length=128):
        super(HierarchicalBert, self).__init__()
        supported_models = ['bert', 'roberta', 'deberta']
        assert encoder.config.model_type in supported_models  # other model types are not supported so far
        # Pre-trained segment (token-wise) encoder, e.g., BERT
        self.encoder = encoder
        # Specs for the segment-wise encoder
        self.hidden_size = encoder.config.hidden_size
        self.max_segments = max_segments
        self.max_segment_length = max_segment_length
        # Init sinusoidal positional embeddings
        self.seg_pos_embeddings = nn.Embedding(max_segments + 1, encoder.config.hidden_size,
                                               padding_idx=0,
                                               _weight=sinusoidal_init(max_segments + 1, encoder.config.hidden_size))
        # Init segment-wise transformer-based encoder
        self.seg_encoder = nn.Transformer(d_model=encoder.config.hidden_size,
                                          nhead=encoder.config.num_attention_heads,
                                          batch_first=True, dim_feedforward=encoder.config.intermediate_size,
                                          activation=encoder.config.hidden_act,
                                          dropout=encoder.config.hidden_dropout_prob,
                                          layer_norm_eps=encoder.config.layer_norm_eps,
                                          num_encoder_layers=2, num_decoder_layers=0).encoder
        # Linear Classifier
        self.classifier = torch.nn.Linear(encoder.config.hidden_size, 2)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                ):
        # Hypothetical Example
        # Batch of 4 documents: (batch_size, n_segments, max_segment_length) --> (4, 64, 128)
        # BERT-BASE encoder: 768 hidden units

        # Squash samples and segments into a single axis (batch_size * n_segments, max_segment_length) --> (256, 128)
        # breakpoint()
        input_ids_reshape = input_ids.contiguous().view(-1, input_ids.size(-1))
        attention_mask_reshape = attention_mask.contiguous().view(-1, attention_mask.size(-1))
        if token_type_ids is not None:
            token_type_ids_reshape = token_type_ids.contiguous().view(-1, token_type_ids.size(-1))
        else:
            token_type_ids_reshape = None

        # Extend it to batch_size * max_segments
        # Encode segments with BERT --> (256, 128, 768)
        encoder_outputs = self.encoder(input_ids=input_ids_reshape,
                                       attention_mask=attention_mask_reshape,
                                       token_type_ids=token_type_ids_reshape)[0]

        # Reshape back to (batch_size, n_segments, max_segment_length, output_size) --> (4, 64, 128, 768)
        encoder_outputs = encoder_outputs.contiguous().view(input_ids.size(0), self.max_segments,
                                                            self.max_segment_length,
                                                            self.hidden_size)

        # Gather CLS outputs per segment --> (4, 64, 768)
        # Use [CLS] to extract the encode vector of the whole segment (block, sentence)
        encoder_outputs = encoder_outputs[:, :, 0] # it only consider first logit (with BERT model -> extract segment VECTOR)

        # Infer real segments, i.e., mask paddings
        seg_mask = (torch.sum(input_ids, 2) != 0).to(input_ids.dtype)
        # Infer and collect segment positional embeddings
        seg_positions = torch.arange(1, self.max_segments + 1).to(input_ids.device) * seg_mask
        # Add segment positional embeddings to segment inputs
        encoder_outputs += self.seg_pos_embeddings(seg_positions)

        # Encode segments with segment-wise transformer
        seg_encoder_outputs = self.seg_encoder(encoder_outputs) # get the transformer encoder output

        # Collect document representation
        # We don't need this
        # outputs, _ = torch.max(seg_encoder_outputs, 1) # get the final result (for dim 1 which is per segment? need further check)

        # return SimpleOutput(last_hidden_state=outputs, hidden_states=outputs) # 回這些應該是為了給下面那層classifier用
        # return outputs, seg_encoder_outputs, encoder_outputs
        linear_outputs = self.classifier(seg_encoder_outputs)
        return linear_outputs

if DEBUG:
    fake_input = torch.randint(1, 1000, (4, 64, 128))
    fake_attn = torch.ones(4, 64, 128)
    # encoder = AutoModel.from_pretrained('bert-base-uncased')
    encoder = AutoModel.from_pretrained("bert-base-chinese")
    fake_model = HierarchicalBert(encoder)
    fake_output = fake_model(fake_input, fake_attn)

    fake_output.shape # [ 4(batch_size), 64(block_size), 2(two_class) ]
    # 看起來不錯哦 只要資料處理一下就OK了