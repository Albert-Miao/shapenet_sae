import argparse

class AEOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='options')
        
        self.parser.add_argument("--gpu",
                                 default=0,
                                 type=int)
        self.parser.add_argument("--dataset",
                                 default="shapenet")
        self.parser.add_argument("--batch_size",
                                 default=32,
                                 type=int)
        self.parser.add_argument("--super_batch_size",
                                 default=250,
                                 type=int)
        self.parser.add_argument("--lr",
                                 default=0.004,
                                 type=float)
        self.parser.add_argument("--momentum",
                                 default=0.9,
                                 type=float)
        self.parser.add_argument("--num_model_epochs",
                                 default=50,
                                 type=int)
        self.parser.add_argument("--num_sae_epochs",
                                 default=20,
                                 type=int)
        
        self.parser.add_argument("--l1_lambda",
                                 default=0.02,
                                 type=float)
        self.parser.add_argument("--codebook_size",
                                 default=320,
                                 type=int)
        self.parser.add_argument("--hidden_rep_size",
                                 default=50,
                                 type=int)
        
        self.parser.add_argument("--model_path",
                                 default="./03_12_25.pth",
                                 type=str)
        self.parser.add_argument("--sae_model_path",
                                 default="./03_12_25_sae.pth",
                                 type=str)
        
    
    def parse(self, args=None):
        self.options = self.parser.parse_args(args)
        return self.options