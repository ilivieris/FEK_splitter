import numpy as np
from sklearn import metrics
import torch

def roc_auc_score_multiclass(actual_class, pred_class, average = "macro"):

  #creating a set of all the unique classes using the actual class list
  unique_class = set(actual_class)
  roc_auc_dict = {}
  AUC = []
  for per_class in unique_class:
    #creating a list of all the classes except the current class 
    other_class = [x for x in unique_class if x != per_class]

    #marking the current class as 1 and all other classes as 0
    new_actual_class = [0 if x in other_class else 1 for x in actual_class]
    new_pred_class = [0 if x in other_class else 1 for x in pred_class]

    #using the sklearn metrics method to calculate the roc_auc_score
    roc_auc = metrics.roc_auc_score(new_actual_class, new_pred_class, average = average)
    roc_auc_dict[per_class] = roc_auc
    AUC.append(roc_auc)
  return np.mean(AUC)


def performance_evaluation(labels:torch.Tensor=None, predictions:torch.Tensor=None):
  '''
    Performance evaluation metrics (Accuracy, AUC, Precision, Recall, GM, CM)
    Parameters
    ----------
    labels: true tokens
    predictions: predicted tokens
    attention_mask: attention mask
    Returns
    -------
    Accuracy (float)
    AUC (float)
    Precision (float)
    Recall (float)
    GM (float)
    CM (ndarray)
  '''
  y = np.array([0 if x < 0.5 else 1 for x in labels])
  pred = np.array([0 if x < 0.5 else 1 for x in predictions])

  Accuracy = 100.0 * metrics.accuracy_score(y, pred)
  try:
      AUC = roc_auc_score_multiclass(y, pred)
  except:
      AUC = 0.0
  # Recall = metrics.recall_score(y, pred, average='macro')
  # Precision = metrics.precision_score(y, pred, average='macro')    
  CM = metrics.confusion_matrix(y, pred)

  # GM = np.prod(np.diag(CM)) ** (1.0/CM.shape[0])

  return Accuracy, AUC, CM #Precision, Recall, GM

