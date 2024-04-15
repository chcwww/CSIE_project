import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, BertPreTrainedModel, RobertaConfig, RobertaModel, RobertaForSequenceClassification
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import TokenClassifierOutput

# check branch
class Introspector(BertPreTrainedModel):
    
    config_class = RobertaConfig
    pretrained_model_archive_map = None
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(Introspector, self).__init__(config)
        self.roberta = RobertaModel(config)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(config.hidden_size, 1)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = logits
        if labels is not None:
            labels = labels.type_as(logits)
            loss_fct = torch.nn.BCEWithLogitsLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1), labels.view(-1))
            outputs = (loss, logits)

        return outputs  # (loss), scores, (hidden_states), (attentions)



class Reasoner(object): # Interface
    def export_labels(self, bufs, device):
        raise NotImplementedError
        # return (labels: consistent with forward, crucials: list of list of blks)
    def forward(self, ids, attn_masks=None, type_ids=None, labels=None, **kwargs):
        raise NotImplementedError
        # return (loss, ) if labels is not None else ...

class QAReasoner(Reasoner, BertPreTrainedModel):
    config_class = RobertaConfig
    pretrained_model_archive_map = None
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(QAReasoner, self).__init__(config)

        self.roberta = RobertaModel(config)
        self.qa_outputs = torch.nn.Linear(config.hidden_size, 2)

        self.init_weights()

    @classmethod
    def export_labels(cls, bufs, device):
        labels = torch.zeros(2, len(bufs), dtype=torch.long, device=device)
        crucials = []
        for i, buf in enumerate(bufs):
            t, crucial = 0, []
            for b in buf.blocks:
                if hasattr(b, 'start'):
                    labels[0, i] = t + b.start[0]
                if hasattr(b, 'end'):
                    labels[1, i] = t + b.end[0]
                if hasattr(b, 'start') or hasattr(b, 'end') or b.blk_type == 0:
                    crucial.append(b)
                t += len(b)
            crucials.append(crucial)
        return labels, crucials

    def forward(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        position_ids=None,
        head_mask=None,
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
        )

        sequence_output = outputs[0] # batch_size * max_len * hidden_size

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if labels is not None:
            start_positions, end_positions = labels
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index, reduction='none')
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = total_loss

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)


class ClassificationReasoner(RobertaForSequenceClassification, Reasoner):
    
    def __init__(self, config):
        super(ClassificationReasoner, self).__init__(config)

    @classmethod
    def export_labels(cls, bufs, device):
        labels = torch.zeros(len(bufs), dtype=torch.long, device=device)
        for i, buf in enumerate(bufs):
            labels[i] = int(buf[0].label)
        return labels, [[b for b in buf if b.blk_type == 0] for buf in bufs]

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
    ):
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )
        sequence_output = outputs[0]
        # 可考慮加dropout
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss(reduction='none')
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs # 回傳損失值跟隱藏層?

        return outputs  # (loss), logits, (hidden_states), (attentions)
    

class ALLonBert(BertPreTrainedModel, Reasoner) :

    def __init__(self, config) :
        super(ALLonBert, self).__init__(config)
        from transformers import AutoTokenizer
        self.dropouts = torch.nn.Dropout(0.1)
        self.roberta = RobertaModel(config)
        self.tokenizer = AutoTokenizer.from_pretrained("luhua/chinese_pretrain_mrc_roberta_wwm_ext_large")
        self.classifier = torch.nn.Linear(config.hidden_size, 2)
        self.all_config = config

    @classmethod
    def export_labels(cls, bufs, device): # TODO 根據新的標籤類型來更改
        # labels = torch.zeros(len(bufs), dtype=torch.long, device=device)
        labels = []
        for i, buf in enumerate(bufs):
            # print(f"This is type : {type(buf)} and {type(buf[0])} and {len(buf)}")
            # print(buf[0])
            # print(buf[1])
            # print(buf[2])
            # print(buf[-1])
            # print(buf[0].label)
            labels.append(buf[0].label)
            # labels[i] = int(buf[0].label)
        return labels, [[b for b in buf if b.blk_type == 0] for buf in bufs]

    def forward(
            self,
            input_ids = None,
            attention_mask = None,
            token_type_ids = None,
            position_ids = None,
            head_mask = None,
            inputs_embeds = None,
            labels = None,
            # position of cls token
            pos = None, # important : should modify "reasoner_module.py"
    ) :
        outputs = self.roberta(
            input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            position_ids = position_ids,
            head_mask = head_mask,
            inputs_embeds = inputs_embeds,
        )
        

        print()
        cls_list = []
        for ina in input_ids :
            cls_tmp = []
            text_tmp = []
            place = 0
            for ink in ina :
                if int(ink.detach().cpu()) == 101 : # equals to [CLS]
                    cls_tmp.append(place) # cls_tmp.append(place)
                    # print(int(ink.detach().cpu()), end = " ")
                place += 1
                text_tmp.append(ink)
            cls_list.append(cls_tmp)
            # print(cls_tmp)
            print("".join(self.tokenizer.convert_ids_to_tokens(text_tmp)))
            print("\n========================\n")
        print("################################")

        print("我是labels :")
        print(labels)
        print(f"POS here : {pos}") # 他們各是第幾句

        sequence_outputs = outputs[0] # last_hidden_state

        sequence_outputs = self.dropouts(sequence_outputs)
        
        print("我是cls_list :")
        print(cls_list)
        s_o = sequence_outputs.shape
        logits_list = []
        for nu, c_l in enumerate(cls_list) : # 為了避免每組的Block數不一樣 所以全部分開預測
            out_tensor = sequence_outputs[nu, [com + 1 for com in c_l], :] # 拿到[CLS](或是+1就是第一個字)的BERT向量
            logits = self.classifier(out_tensor) # problem 1 , 大家長度會不一樣 但在data_helper控制block數可以mitigate這個問題
            logits_list.append(logits)

        # print(f"size... : {sequence_outputs.shape} and {sequence_outputs[:, cls_list, :].shape} and {logits.shape}")
        # logits = self.classifier(sequence_outputs[:, [0, 12], :].view(-1, self.all_config.hidden_size))

        # outputs = (logits,) + outputs[2:]
        # if labels is not None :
        #     loss_fct = CrossEntropyLoss(reduction = 'none')
        #     loss = loss_fct(logits.view(-1), labels.view(-1))
        #     outputs = (loss,) + outputs
        
        # return outputs
        
        losses = []
        for ba, logit in enumerate(logits_list) : # 一樣每篇分開跑
            loss_func = CrossEntropyLoss()
            soft_max = torch.max(logit, dim = 1).indices # 某一個的預測結果
            pred = torch.zeros(len(labels[ba]), device = self.device) # 先開一個跟原始文章一樣多block的全0的tensor
            for sn, ou in enumerate(pos[ba]) :
                if soft_max[sn] == 1 and ou != -1 : # 如果預測成1(重點block)以及這不是qbuf的[CLS]那塊
                    pred[ou] = 1 # 相應預測之block為1
            pred.requires_grad = True # 讓loss可以backward
            print(f"pred : {pred.view(-1)}, label : {torch.tensor(labels[ba]).view(-1)}")
            loss = loss_func(pred.float().view(-1).to(self.device), torch.tensor(labels[ba]).float().view(-1).to(self.device)) # 算cross entropy
            losses.append(loss)
        # pred = torch.zeros(len(labels), len(labels[0]), dtype=torch.long, device=self.device)
        # print(f"我是logits :\n{logits}")
        # soft_max = torch.max(logits, dim=1).indices
        # print(f"我是soft_max :\n{soft_max}")

        #=== for test
        # labels = torch.ones([1, logits.view(-1).size(0)])
        #=== for test
        # logits = self.classifier(sequence_outputs[:, [0, 12], :].view(-1, self.all_config.hidden_size))
        # labels = torch.ones([1, logits.view(-1).size(0)])
        # loss = loss_func(logits.view(-1), labels.view(-1))
            
        loss = sum(losses) # 每一組的loss加起來
        outputs = loss
        
        # loss_func = CrossEntropyLoss()
        # outputs = loss_func(torch.tensor([1., 0.]), torch.tensor([0., 0.]))
        # outputs.requres_grad = True
        return outputs

        # return TokenClassifierOutput(
        #     loss = loss,
        #     logits = logits,
        #     hidden_states = outputs.hidden_states,
        #     attentions = outputs.attentions
        #     )
        

# aLLonBert.from_pretrained('roberta-base')