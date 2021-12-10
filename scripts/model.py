
import torch
import torch.nn as nn

class AttentionHead(nn.Module):
    def __init__(self, h_size, hidden_dim=512):
        super().__init__()
        self.W = nn.Linear(h_size, hidden_dim)
        self.V = nn.Linear(hidden_dim, 1)
        
    def forward(self, features):
        att = torch.tanh(self.W(features))
        score = self.V(att)
        attention_weights = torch.softmax(score, dim=1)
        context_vector = attention_weights * features
        context_vector = torch.sum(context_vector, dim=1)

        return context_vector

class CLRPModel(nn.Module):
    def __init__(self,transformer,config):
        super(CLRPModel,self).__init__()
        self.h_size = config.hidden_size
        self.transformer = transformer
        self.head = AttentionHead(self.h_size*4)
        self.linear = nn.Linear(self.h_size*2, 1)
        self.linear_out = nn.Linear(self.h_size*8, 1)

              
    def forward(self, input_ids, attention_mask):
        transformer_out = self.transformer(input_ids, attention_mask)
        # tranformer_out是三个键值的字典，第一部分是last_hidden_states，最后一层transformer的输出/b*s*dim
        # 第二部分是pooler层的输出/b*1*dim
        # 第三部分是所有transformer层的输出，是个列表。
        all_hidden_states = torch.stack(transformer_out.hidden_states)
        cat_over_last_layers = torch.cat(
            (all_hidden_states[-1], all_hidden_states[-2], all_hidden_states[-3], all_hidden_states[-4]),-1
        )
        
        cls_pooling = cat_over_last_layers[:, 0]   
        head_logits = self.head(cat_over_last_layers)
        y_hat = self.linear_out(torch.cat([head_logits, cls_pooling], -1))
        # 使用最后四层的输出拼接并做self-attention，然后和[CLS]位置的输出拼接，最后过全连接。
        return y_hat