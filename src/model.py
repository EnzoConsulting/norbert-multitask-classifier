from transformers import AutoModel, AutoConfig
import torch.nn as nn

class MultiHeadNorBERT(nn.Module):
    def __init__(self, model_name, num_labels_dict):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        self.dropout = nn.Dropout(0.3)
        self.sentiment = nn.Linear(hidden_size, num_labels_dict['sentiment'])
        self.priority = nn.Linear(hidden_size, num_labels_dict['priority'])
        self.category = nn.Linear(hidden_size, num_labels_dict['category'])

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # CLS token

        pooled = self.dropout(pooled)

        return {
            "sentiment": self.sentiment(pooled),
            "priority": self.priority(pooled),
            "category": self.category(pooled)
        }
