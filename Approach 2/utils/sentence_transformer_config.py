import torch

class Parameters():
    def __init__(self) -> None:
        # Data & tokenizer parameters
        self.dataset_path = '../Data/Dataset.csv'
        self.number_of_iterations = 100
        self.number_training_percentage = 0.5

        # Sentence-Transformer parameters
        self.model_name = 'lighteternal/stsb-xlm-r-greek-transfer'
        self.output_dir = 'sentence_transformer'
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


        # Training parameters
        self.test_size = 0.2 # Testing size as percentage of data size
        self.batch_size = 4
        self.loss_function = 'CosineSimilarityLoss' # ['CosineSimilarityLoss', 'ContrastiveLoss', 'OnlineContrastiveLoss', 'SoftmaxLoss']
        self.weight_decay = 0.01 # 0.001
        self.learning_rate = 1e-4
        self.epochs = 1
        self.seed = 1983

