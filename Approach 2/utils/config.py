import torch

class Parameters():
    def __init__(self) -> None:
        # Data & tokenizer parameters
        self.dataset_path = '../Data/Dataset.csv'

        # # Model parameters
        self.output_dir = 'model' 
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.dropout_rate = 0.5
        self.hidden_size = [16, 1]

        # Training parameters
        self.test_size = 0.2 # Testing size as percentage of data size
        self.valid_size = 0.1 # Validation size as percentage of the training data size        
        self.train_batch_size = 3
        self.test_batch_size = 64
        self.loss_function = 'BCE' # 'Focal', 'BCE', 'weighted_BCE'
        self.weight = [1, 1.2] # Used in case loss function is 'weighted_BCE'
        self.optimizer = 'SGD' # 'Adam', 'AdamW', 'SGD', 'BertAdam'
        self.momentum = 0.9 # Used for SGD optimizer
        self.weight_decay = 0.01 # 0.001
        self.learning_rate = 1e-5
        self.epochs = 500
        self.seed = 1983

