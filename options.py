import argparse

class AEOptions:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='options')
        
        self.parser.add_argument("--model",
                                 default="naive",
                                 type=str,
                                 help="'pointnet' or 'naive'")
        self.parser.add_argument("--load_model_for_sae",
                                 default=False,
                                 type=bool)
        self.parser.add_argument("--visualize",
                                 default=False,
                                 type=bool)
        
        self.parser.add_argument("--gpu",
                                 default=0,
                                 type=int)
        self.parser.add_argument("--dataset",
                                 default="shapenet")
        self.parser.add_argument("--num_points",
                                 default=2048,
                                 type=int)
        self.parser.add_argument("--batch_size",
                                 default=32,
                                 type=int)
        self.parser.add_argument("--super_batch_size",
                                 default=250,
                                 type=int)
        self.parser.add_argument("--lr",
                                 default=0.001,
                                 type=float)
        self.parser.add_argument("--momentum",
                                 default=0.9,
                                 type=float)
        self.parser.add_argument("--num_model_epochs",
                                 default=1000,
                                 type=int)
        self.parser.add_argument("--num_sae_epochs",
                                 default=600,
                                 type=int)
        
        self.parser.add_argument("--l1_lambda",
                                 default=2,
                                 type=float)
        self.parser.add_argument("--dead_lambda",
                                 default=1/32,
                                 type=float)
        self.parser.add_argument("--codebook_size",
                                 default=4096,
                                 type=int)
        self.parser.add_argument("--hidden_rep_dim",
                                 default=512,
                                 type=int)
        self.parser.add_argument("--batch_topk",
                                 default=True,
                                 type=bool)
        self.parser.add_argument("--k",
                                 default=16,
                                 type=bool)
        self.parser.add_argument("--dead_k",
                                 default=64,
                                 type=bool)
        self.parser.add_argument("--pointnet_sae_level",
                                 default=1,
                                 type=bool)
        
        self.parser.add_argument("--model_path",
                                 default="./04_07_25_hd_128_codebook_512.pth",
                                 type=str)
        self.parser.add_argument("--sae_model_path",
                                 default="./04_16_25_hd_128_codebook_512.pth",
                                 type=str)
        self.parser.add_argument("--encoder_save_path",
                                 default="./04_17_25_encoder.pth",
                                 type=str)
        self.parser.add_argument("--encoder_load_path",
                                 default="./04_16_25_encoder.pth",
                                 type=str)       
        self.parser.add_argument("--decoder_save_path",
                                 default="./04_17_25_decoder.pth",
                                 type=str)
        self.parser.add_argument("--decoder_load_path",
                                 default="./04_16_25_decoder.pth",
                                 type=str)       
        self.parser.add_argument("--sae_save_path",
                                 default="./04_17_25_sae.pth",
                                 type=str)
        self.parser.add_argument("--sae_load_path",
                                 default="./04_16_25_sae.pth",
                                 type=str)       
        self.parser.add_argument("--classifier_save_path",
                                 default="./04_17_25_classifier.pth",
                                 type=str)
        self.parser.add_argument("--classifier_load_path",
                                 default="./04_16_25_classifier.pth",
                                 type=str)       
        self.parser.add_argument("--load_encoder",
                                 default=True,
                                 type=bool)
        self.parser.add_argument("--load_decoder",
                                 default=True,
                                 type=bool)
        self.parser.add_argument("--load_sae",
                                 default=False,
                                 type=bool)       
        self.parser.add_argument("--load_classifier",
                                 default=False,
                                 type=bool)       
        
        self.parser.add_argument("--train_encoder",
                                 default=True,
                                 type=bool)
        self.parser.add_argument("--train_sae",
                                 default=False,
                                 type=bool)
        self.parser.add_argument("--train_classifier",
                                 default=False,
                                 type=bool)       
        self.parser.add_argument("--h_decoder",
                                 default=False,
                                 type=bool)
         
        self.parser.add_argument('--cates', type=str, nargs='+', default=['all'],
                                 help="Categories to be trained (useful only if 'shapenet' is selected)")
        self.parser.add_argument("--tr_max_sample_points", type=int, default=2048,
                                 help='Max number of sampled points (train)')
        self.parser.add_argument("--te_max_sample_points", type=int, default=0,
                                 help='Max number of sampled points (test)')
        self.parser.add_argument('--data_dir', type=str, default="ShapeNetCore.v2.PC15k",
                                 help="Path to the training data")
        self.parser.add_argument('--dataset_scale', type=float, default=1.,
                                 help='Scale of the dataset (x,y,z * scale = real output, default=1).')
        self.parser.add_argument('--random_rotate', action='store_true',
                                 help='Whether to randomly rotate each shape.')
        self.parser.add_argument('--normalize_per_shape', action='store_true',
                                 help='Whether to perform normalization per shape.')
        self.parser.add_argument('--normalize_std_per_axis', action='store_true',
                                 help='Whether to perform normalization per axis.')
    
    def parse(self, args=None):
        self.options = self.parser.parse_args(args)
        return self.options