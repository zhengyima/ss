import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class BertContrastive(nn.Module):
    def __init__(self, bert_model, temperature, use_mlm=False):
        super(BertContrastive, self).__init__()
        self.bert_model = bert_model
        # self.dropout = nn.Dropout(0.1)
        # self._use_mlm = use_mlm
        self.temperature = temperature
        # if self._use_mlm:
        #     self.cls = BertOnlyMLMHead(self.bert_model.config)
        #     self.mlm_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.cl_loss = torch.nn.CrossEntropyLoss(reduction='none')

    def forward(self, batch_data):
        """
        Args:
            input_ids ([type]): [description]
            attention_mask ([type]): [description]
            token_type_ids ([type]): [description]
        """
        batch_size = batch_data["input_ids1"].size(0)

        # masked_lm_loss = 0
        # if self._use_mlm:
        #     input_ids_mlm = batch_data["input_ids_mlm"]
        #     attention_mask_mlm = batch_data["attention_mask_mlm"]
        #     token_type_ids_mlm = batch_data["token_type_ids_mlm"]
        #     masked_lm_labels = batch_data["masked_lm_labels"]
        #     bert_inputs_mlm = {'input_ids': input_ids_mlm, 'attention_mask': attention_mask_mlm, 'token_type_ids': token_type_ids_mlm}
        #     bert_outputs_mlm = self.bert_model(**bert_inputs_mlm)[0]
        #     prediction_scores = self.cls(bert_outputs_mlm)
        #     masked_lm_loss = self.mlm_loss(prediction_scores.view(-1, self.bert_model.config.vocab_size), masked_lm_labels.view(-1))

        input_ids1 = batch_data["input_ids1"]
        attention_mask1 = batch_data["attention_mask1"]
        token_type_ids1 = batch_data["token_type_ids1"]
        bert_inputs1 = {'input_ids': input_ids1, 'attention_mask': attention_mask1, 'token_type_ids': token_type_ids1}
        sent_rep1 = self.bert_model(**bert_inputs1)[1]

        input_ids2 = batch_data["input_ids2"]
        attention_mask2 = batch_data["attention_mask2"]
        token_type_ids2 = batch_data["token_type_ids2"]
        bert_inputs2 = {'input_ids': input_ids2, 'attention_mask': attention_mask2, 'token_type_ids': token_type_ids2}
        sent_rep2 = self.bert_model(**bert_inputs2)[1]

        sent_norm1 = sent_rep1.norm(dim=-1, keepdim=True)  # [batch]
        sent_norm2 = sent_rep2.norm(dim=-1, keepdim=True)  # [batch]
        batch_self_11 = torch.einsum("ad,bd->ab", sent_rep1, sent_rep1) / (torch.einsum("ad,bd->ab", sent_norm1, sent_norm1) + 1e-6)  # [batch, batch]
        batch_cross_12 = torch.einsum("ad,bd->ab", sent_rep1, sent_rep2) / (torch.einsum("ad,bd->ab", sent_norm1, sent_norm2) + 1e-6)  # [batch, batch]
        batch_self_11 = batch_self_11 / self.temperature
        batch_cross_12 = batch_cross_12 / self.temperature
        batch_first = torch.cat([batch_self_11, batch_cross_12], dim=-1)  # [batch, batch * 2]
        batch_arange = torch.arange(batch_size).to(torch.cuda.current_device())
        mask = F.one_hot(batch_arange, num_classes=batch_size * 2) * -1e10
        batch_first += mask
        batch_label1 = batch_arange + batch_size  # [batch]

        batch_self_22 = torch.einsum("ad,bd->ab", sent_rep2, sent_rep2) / (torch.einsum("ad,bd->ab", sent_norm2, sent_norm2) + 1e-6)  # [batch, batch]
        batch_cross_21 = torch.einsum("ad,bd->ab", sent_rep2, sent_rep1) / (torch.einsum("ad,bd->ab", sent_norm2, sent_norm1) + 1e-6)  # [batch, batch]
        batch_self_22 = batch_self_22 / self.temperature
        batch_cross_21 = batch_cross_21 / self.temperature
        batch_second = torch.cat([batch_self_22, batch_cross_21], dim=-1)  # [batch, batch * 2]
        batch_second += mask
        batch_label2 = batch_arange + batch_size  # [batch]

        batch_predict = torch.cat([batch_first, batch_second], dim=0)
        batch_label = torch.cat([batch_label1, batch_label2], dim=0)  # [batch * 2]
        contras_loss = self.cl_loss(batch_predict, batch_label)

        batch_logit = batch_predict.argmax(dim=-1)
        acc = torch.sum(batch_logit == batch_label).float() / (batch_size * 2)

        # loss = masked_lm_loss + contras_loss

        return contras_loss, acc
