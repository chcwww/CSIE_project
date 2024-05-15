import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoTokenizer, AutoModel, BertPreTrainedModel, RobertaConfig, RobertaModel, RobertaForSequenceClassification
from torch.nn import CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import TokenClassifierOutput
import logging
import os
# print(f"Dir now : {os.getcwd()}")
# from utils.util import MODEL_NAME

# check branch
class Introspector(torch.nn.Module):
    def __init__(self, m_name):
        super(Introspector, self).__init__()
        self.roberta = AutoModel.from_pretrained(m_name)
        self.dropout = torch.nn.Dropout(0.1)
        bert_dim = self.roberta.config.hidden_size
        self.classifier = torch.nn.Linear(bert_dim, 1)

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
            # CrossEntropy : softmax, BCE : sigmoid
            loss_fct = torch.nn.BCEWithLogitsLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                # 他label其實也是全局的 所以也要選出activate的
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
        # self.roberta = AutoModel.from_pretrained(MODEL_NAME)
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
            labels.append(buf[0].label) # 第一塊(qbuf)身上的label
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
        # input_ids, attention_mask, token_type_ids, labels, pos = *inputs, labels, blk_pos
        outputs = self.roberta(
            input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            position_ids = position_ids,
            head_mask = head_mask,
            inputs_embeds = inputs_embeds,
        )
        
        # outputs[0].shape [1, 512, 768]
        # outputs[1].shape [1, 768]

        logging.debug('\n')
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
            logging.debug("".join(self.tokenizer.convert_ids_to_tokens(text_tmp)))
            logging.debug("\n========================\n")
        logging.debug("################################")

        logging.debug("我是labels :")
        logging.debug(labels)
        logging.debug(f"POS here : {pos}") # 他們各是第幾句

        sequence_outputs = outputs[0] # last_hidden_state

        sequence_outputs = self.dropouts(sequence_outputs)
        
        logging.debug("我是cls_list :")
        logging.debug(cls_list)
        s_o = sequence_outputs.shape
        logits_list = []

        ## DEBUG
        # batch, length, dim = 4, 512, 768
        # sequence_outputs = torch.randn(batch, length, dim)
        ## DEBUG
        for nu, c_l in enumerate(cls_list) : # 為了避免每組的Block數不一樣 所以全部分開預測
            # start = [com for com in c_l]
            # end =
            try:
                out_tensor = sequence_outputs[nu, [com + 1 for com in c_l], :] # 拿到[CLS](或是+1就是第一個字)的BERT向量
            except:
                breakpoint()
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

        set_local = True
        if set_local :
            for ba, logit in enumerate(logits_list) :
                loss_func = CrossEntropyLoss()
                soft_max = torch.max(logit, dim = 1).indices
                pred = torch.zeros(len(logit), device = self.device)
                lab = torch.zeros(len(logit), device = self.device)
                for sn, ou in enumerate(pos[ba]) :
                    if soft_max[sn] == 1 and ou != -1 :
                        pred[sn] = 1
                    if ou != -1 :
                        lab[sn] = labels[ba][ou]
                pred.requires_grad = True # 讓loss可以backward
                lab.requires_grad = True # 讓loss可以backward
                logging.debug(f"pred : {pred.view(-1)}, label : {lab.view(-1)}")
                loss = loss_func(pred.float().view(-1).to(self.device), lab.float().view(-1).to(self.device)) # 算cross entropy
                losses.append(loss)
        else :
            for ba, logit in enumerate(logits_list) : # 一樣每篇分開跑
                loss_func = CrossEntropyLoss()
                soft_max = torch.max(logit, dim = 1).indices # 某一個的預測結果
                pred = torch.zeros(len(labels[ba]), device = self.device) # 先開一個跟原始文章一樣多block的全0的tensor
                for sn, ou in enumerate(pos[ba]) :
                    if soft_max[sn] == 1 and ou != -1 : # 如果預測成1(重點block)以及這不是qbuf的[CLS]那塊
                        pred[ou] = 1 # 相應預測之block為1
                pred.requires_grad = True # 讓loss可以backward
                logging.debug(f"pred : {pred.view(-1)}, label : {torch.tensor(labels[ba]).view(-1)}")
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
            
        # loss = sum(losses) # 每一組的loss加起來
        # outputs = torch.FloatTensor(losses)
        outputs = losses
        
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
        
# m_name = "bert-base-chinese"
# input_ids = inputs[0]
# attention_mask = inputs[1]
# token_type_ids = inputs[2]
# position_ids = None
# head_mask = None
# inputs_embeds = None
# labels = labels
# pos = blk_pos 
# roberta = AutoModel.from_pretrained(MODEL_NAME)
# tokenizer = AutoTokenizer.from_pretrained("luhua/chinese_pretrain_mrc_roberta_wwm_ext_large")
# classifier = torch.nn.Linear(768, 2)
# dropouts = torch.nn.Dropout(0.1)


class ALLonBert_v2(torch.nn.Module, Reasoner) :

    def __init__(self, m_name) :
        super(ALLonBert_v2, self).__init__()
        self.dropouts = torch.nn.Dropout(0.1)
        self.roberta = AutoModel.from_pretrained(m_name)
        bert_dim = self.roberta.config.hidden_size
        self.tokenizer = AutoTokenizer.from_pretrained(m_name)
        self.classifier = torch.nn.Linear(bert_dim, 2)

    @classmethod
    def export_labels(cls, bufs, device): # TODO 根據新的標籤類型來更改
        labels = []
        for i, buf in enumerate(bufs):
            labels.append(buf[0].label) # 第一塊(qbuf)身上的label
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
            device = None,
            debug_buf = None,
    ) :
        # input_ids, attention_mask, token_type_ids, labels, pos = *inputs, labels, blk_pos
        self.roberta = self.roberta.to(device)
        # outputs = roberta(
        outputs = self.roberta(
            input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            position_ids = position_ids,
            head_mask = head_mask,
            inputs_embeds = inputs_embeds,
        )
        
        # outputs[0].shape [1, 512, 768]
        # outputs[1].shape [1, 768]

        sep_list = []
        for ina in input_ids :
            sep_tmp = []
            text_tmp = []
            place = 0
            for ink in ina :
                if int(ink.detach().cpu()) == 102 :
                    sep_tmp.append(place)
                place += 1
                text_tmp.append(ink)
            sep_list.append(sep_tmp)

        sequence_outputs = outputs[0] # last_hidden_state

        sequence_outputs = self.dropouts(sequence_outputs)
        # sequence_outputs = dropouts(sequence_outputs)
        
        # sequence_outputs.shape [ 1 (batch size), 512 (token len limit), 768 (bert dim) ]
        logits_list = []
        for nu, n_list in enumerate(sep_list) : # 為了避免每組的Block數不一樣 所以全部分開預測
            try:
                out_tensor = torch.mean(sequence_outputs[nu, 1:n_list[0], :], 0).view(1, -1)
                # 想直接改成token-wise的跑loss
                for i in range(len(n_list)-1) :
                    # 整句取平均
                    temp_tensor = torch.mean(sequence_outputs[nu, n_list[i]+1:n_list[i+1], :], 0).view(1, -1)
                    out_tensor = torch.cat((out_tensor, temp_tensor), 0)
                # out_tensor.shape [7 (num of block), 768]
                logits = self.classifier(out_tensor)
                if (logits==logits).sum()!=len(logits)*2: # check nan
                    breakpoint()
                # logits.shape [7 (num of block), 2]
                logits_list.append(logits)
            except:
                breakpoint()
                print('Weird buffer exist..')

        outputs = (logits_list, )
        
        if labels is not None :
            losses = []
            mu_label = []
            loss_func = CrossEntropyLoss()
            for batch_id, logit in enumerate(logits_list) :
                # 原來之前是這裡寫錯
                soft_max = F.softmax(logit, dim = 1) # F.log_softmax
                local_label = torch.zeros(len(logit), device = device)
                local_pos = pos[batch_id].copy()
                local_pos.pop(0) # 第一個是[CLS]
                local_true = labels[batch_id].copy()
                for local_id, blk_id in enumerate(local_pos) :
                    if local_true[blk_id-1] == 1 :
                        local_label[local_id] = 1
                local_label = local_label.long()
                # 算cross entropy
                l_loss = loss_func(logit, local_label.view(-1).to(device))
                if l_loss != l_loss:
                    breakpoint()
                losses.append(l_loss)
                mu_label.append(local_label.view(-1))
            outputs = (losses, mu_label, ) + outputs
            # breakpoint()
        return outputs
    
    
### We use token-wise loss for ALLonBert_v3
# class ALLonBert_v3(torch.nn.Module, Reasoner) :

#     def __init__(self, m_name) :
#         super(ALLonBert_v2, self).__init__()
#         self.dropouts = torch.nn.Dropout(0.1)
#         self.roberta = AutoModel.from_pretrained(m_name)
#         bert_dim = self.roberta.config.hidden_size
#         self.tokenizer = AutoTokenizer.from_pretrained(m_name)
#         self.classifier = torch.nn.Linear(bert_dim, 2)

#     @classmethod
#     def export_labels(cls, bufs, device): # TODO 根據新的標籤類型來更改
#         labels = []
#         for i, buf in enumerate(bufs):
#             labels.append(buf[0].label) # 第一塊(qbuf)身上的label
#         return labels, [[b for b in buf if b.blk_type == 0] for buf in bufs]

#     def forward(
#             self,
#             input_ids = None,
#             attention_mask = None,
#             token_type_ids = None,
#             position_ids = None,
#             head_mask = None,
#             inputs_embeds = None,
#             labels = None,
#             # position of cls token
#             pos = None, # important : should modify "reasoner_module.py"
#             device = None,
#             debug_buf = None,
#     ) :
#         # input_ids, attention_mask, token_type_ids, labels, pos = *inputs, labels, blk_pos
#         self.roberta = self.roberta.to(device)
#         # outputs = roberta(
#         outputs = self.roberta(
#             input_ids,
#             attention_mask = attention_mask,
#             token_type_ids = token_type_ids,
#             position_ids = position_ids,
#             head_mask = head_mask,
#             inputs_embeds = inputs_embeds,
#         )
        
#         # outputs[0].shape [1, 512, 768]
#         # outputs[1].shape [1, 768]

#         sep_list = []
#         for ina in input_ids :
#             sep_tmp = []
#             text_tmp = []
#             place = 0
#             for ink in ina :
#                 if int(ink.detach().cpu()) == 102 :
#                     sep_tmp.append(place)
#                 place += 1
#                 text_tmp.append(ink)
#             sep_list.append(sep_tmp)

#         sequence_outputs = outputs[0] # last_hidden_state

#         sequence_outputs = self.dropouts(sequence_outputs)
#         # sequence_outputs = dropouts(sequence_outputs)
        
#         # sequence_outputs.shape [ 4 (batch size), 512 (token len limit), 768 (bert dim) ]
#         logits = self.classifier(sequence_outputs) # logits: [ 4, 512, 2 ]

#         outputs = (logits, )

#         mu_label = 0
#         if labels is not None :
#             relevance = torch.zeros((logits.shape[0], logits.shape[1]), device=device)
#             relevance = relevance.type_as(logits)
#             for batch_id, n_list in enumerate(sep_list):
#                 t = 0
#                 for idx, ls in enumerate(n_list):
#                     if labels[batch_id][idx] == 1:
#                         relevance[batch_id, t: ls] = 1
#                     t = ls + 1
                    
#                 local_label = torch.zeros(len(n_list), device = device)
#                 local_pos = pos[batch_id].copy()
#                 local_pos.pop(0) # 第一個是[CLS]
#                 local_true = labels[batch_id].copy()
#                 for local_id, blk_id in enumerate(local_pos) :
#                     if local_true[blk_id-1] == 1 :
#                         local_label[local_id] = 1
#                 local_label = local_label.long()
#                 mu_label.append(local_label.view(-1))
                
#             loss_fct = CrossEntropyLoss()
#             loss = loss_fct(logits.view(-1), labels.view(-1))  
#             outputs = ([loss], mu_label, ) + outputs
#             # breakpoint()
#         return outputs




class ALLonBert_v3(torch.nn.Module, Reasoner) :

    def __init__(self, m_name) :
        super(ALLonBert_v3, self).__init__()
        self.dropouts = torch.nn.Dropout(0.1)
        self.roberta = AutoModel.from_pretrained(m_name)
        bert_dim = self.roberta.config.hidden_size
        self.tokenizer = AutoTokenizer.from_pretrained(m_name)
        self.classifier = torch.nn.Linear(bert_dim, 2)

    @classmethod
    def export_labels(cls, bufs, device): # TODO 根據新的標籤類型來更改
        labels = []
        for i, buf in enumerate(bufs):
            labels.append(buf[0].label) # 第一塊(qbuf)身上的label
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
            device = None,
            debug_buf = None,
    ) :
        # input_ids, attention_mask, token_type_ids, labels, pos = *inputs, labels, blk_pos
        self.roberta = self.roberta.to(device)
        # outputs = roberta(
        outputs = self.roberta(
            input_ids,
            attention_mask = attention_mask,
            token_type_ids = token_type_ids,
            position_ids = position_ids,
            head_mask = head_mask,
            inputs_embeds = inputs_embeds,
        )
        
        # outputs[0].shape [1, 512, 768]
        # outputs[1].shape [1, 768]

        sep_list = []
        for ina in input_ids :
            sep_tmp = []
            text_tmp = []
            place = 0
            for ink in ina :
                if int(ink.detach().cpu()) == 102 :
                    sep_tmp.append(place)
                place += 1
                text_tmp.append(ink)
            sep_list.append(sep_tmp)

        sequence_outputs = outputs[0] # last_hidden_state

        sequence_outputs = self.dropouts(sequence_outputs)
        # sequence_outputs = dropouts(sequence_outputs)
        
        # sequence_outputs.shape [ 1 (batch size), 512 (token len limit), 768 (bert dim) ]
        logits_list = []
        for nu, n_list in enumerate(sep_list) : # 為了避免每組的Block數不一樣 所以全部分開預測
            try:
                out_tensor = torch.mean(sequence_outputs[nu, 1:n_list[0], :], 0).view(1, -1)
                # 想直接改成token-wise的跑loss
                for i in range(len(n_list)-1) :
                    # 整句取平均
                    temp_tensor = torch.mean(sequence_outputs[nu, n_list[i]+1:n_list[i+1], :], 0).view(1, -1)
                    out_tensor = torch.cat((out_tensor, temp_tensor), 0)
                # out_tensor.shape [7 (num of block), 768]
                logits = self.classifier(out_tensor)
                if (logits==logits).sum()!=len(logits)*2: # check nan
                    breakpoint()
                # logits.shape [7 (num of block), 2]
                logits_list.append(logits)
            except:
                breakpoint()
                print('Weird buffer exist..')
                
        true_logits = self.classifier(sequence_outputs) # logits: [ 4, 512, 2 ]
        
        outputs = (logits_list, )
        
        if labels is not None :
            mu_label = []
            for batch_id, logit in enumerate(logits_list) :
                # 原來之前是這裡寫錯
                # soft_max = F.softmax(logit, dim = 1) # F.log_softmax
                local_label = torch.zeros(len(logit), device = device)
                local_pos = pos[batch_id].copy()
                local_pos.pop(0) # 第一個是[CLS]
                local_true = labels[batch_id].copy()
                for local_id, blk_id in enumerate(local_pos) :
                    if local_true[blk_id-1] == 1 :
                        local_label[local_id] = 1
                local_label = local_label.long()
                mu_label.append(local_label.view(-1))
                
        if labels is not None :
            relevance = torch.zeros((true_logits.shape[0], true_logits.shape[1]), device=device)
            relevance = relevance.type_as(logits)
            for batch_id, n_list in enumerate(sep_list):
                t = 0
                for idx, ls in enumerate(n_list):
                    if labels[batch_id][idx] == 1:
                        relevance[batch_id, t: ls] = 1
                    t = ls + 1
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(true_logits.view(-1), labels.view(-1))  
            outputs = ([loss], mu_label, ) + outputs
            # breakpoint()
        return outputs