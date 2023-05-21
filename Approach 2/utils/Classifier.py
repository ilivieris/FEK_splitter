import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from utils.loss_function import *

        
        
class Classifier(nn.Module):
    def __init__(self, config=None, pretrained_model='lighteternal/stsb-xlm-r-greek-transfer', args=None):
        super(Classifier, self).__init__()
        self.args = args
        self.pretrained_model = pretrained_model

        # Backbone sentence-transformer for creating the embeddings
        self.backbone = SentenceTransformer(pretrained_model)

        if len(self.args.hidden_size) == 1:
            self.lin_layer1 = nn.Linear(768, self.args.hidden_size[0])
        else:
            self.lin_layer1 = nn.Linear(768, self.args.hidden_size[0])
            self.lin_layer2 = nn.Linear(self.args.hidden_size[0], self.args.hidden_size[1])

            
    def forward(self, text=None, labels=None):
        # Output from pre-trained model
        output = torch.Tensor( self.backbone.encode(text) )

        # Layer-1
        if len(self.args.hidden_size) == 1:
            # Dropout
            if (self.args.dropout_rate > 0): output = torch.nn.Dropout(self.args.dropout_rate)(output)            
            # Output
            output = self.lin_layer1(output)

        # Layer-2 (if exists)
        if len(self.args.hidden_size) == 2:
            output = self.lin_layer1(output)
            output = torch.nn.ReLU()(output)        
            # Dropout
            if (self.args.dropout_rate > 0): output = torch.nn.Dropout(self.args.dropout_rate)(output)   
            # Output
            output = self.lin_layer2(output)

        # Convert output & label type to Float        
        output = output.type(torch.FloatTensor)
        labels = labels.type(torch.FloatTensor)

        # Output layer
        if (self.args.loss_function in ['BCE', 'weighted_BCE']):
            logits = torch.sigmoid(output)
        else:
            logits = output

        if labels is not None:
            if self.args.loss_function == 'Focal':
                loss_fct = FocalLoss()
            elif self.args.loss_function == 'BCE':                
                loss_fct = nn.BCELoss()
            elif self.args.loss_function == 'weighted_BCE':
                weight = torch.tensor([self.args.weight[0] if x == 0 else self.args.weight[1] for x in labels]).float().to(self.args.device)
                loss_fct = nn.BCELoss(weight=weight)
                
            loss = loss_fct(output.view(-1), labels)
            return loss, logits
        else:
            return logits