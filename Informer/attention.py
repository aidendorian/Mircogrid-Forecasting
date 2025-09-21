import torch

class ProbSparseAttention(torch.nn.Module):
    def __init__(self,
                 masked:bool=True,
                 factor:int=5,
                 scale=None,
                 attn_dropout=0.1,
                 output_attn=False):
        super().__init__()
        
        self.factor = factor
        self.scale = scale
        self.mask_flag = masked
        self.output_attention = output_attn
        self.dropout = torch.nn.Dropout(attn_dropout)
        
        