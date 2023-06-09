import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from utils.loss_function import *

        
        
class Classifier(nn.Module):
    def __init__(self, args=None):
        super(Classifier, self).__init__()
        self.args = args

        # Backbone sentence-transformer for creating the embeddings
        self.backbone = SentenceTransformer(model_name_or_path=args.model_name, device=args.device)
        # Embeddings dimension
        embedding_dimension = self.backbone.get_sentence_embedding_dimension()

        if len(self.args.hidden_size) == 1:
            self.lin_layer1 = nn.Linear(embedding_dimension, self.args.hidden_size[0])
        else:
            self.lin_layer1 = nn.Linear(embedding_dimension, self.args.hidden_size[0])
            self.lin_layer2 = nn.Linear(self.args.hidden_size[0], self.args.hidden_size[1])

            
    def forward(self, text=None, labels=None):
        # Output from pre-trained model
        output = torch.Tensor( self.backbone.encode(text, show_progress_bar=False) ).to(self.args.device)

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

        # Output layer
        if (self.args.loss_function in ['BCE', 'weighted_BCE']):
            logits = torch.sigmoid(output)
        else:
            logits = output


        
        if labels is not None:

            # Convert output & label type to Float        
            logits = logits.type(torch.FloatTensor)
            labels = labels.type(torch.FloatTensor)

            if self.args.loss_function == 'Focal':
                loss_fct = FocalLoss()
            elif self.args.loss_function == 'BCE':                
                loss_fct = nn.BCELoss()
            elif self.args.loss_function == 'weighted_BCE':
                weight = torch.tensor([self.args.weight[0] if x == 0 else self.args.weight[1] for x in labels]).float()
                loss_fct = nn.BCELoss(weight=weight)
            
            loss = loss_fct(logits.view(-1), labels)
            return loss, logits
        else:
            # Convert output & label type to Float        
            logits = logits.type(torch.FloatTensor)

            return logits