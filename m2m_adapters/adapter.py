from torch import nn

class Adapter(nn.Module):
    """
    Adapter for model finetuning, as descripted in:
    https://arxiv.org/pdf/1909.08478.pdf
    """
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config.d_model
        self.layer_norm = nn.LayerNorm(self.embed_dim)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(self.embed_dim, config.adapter_size)
        self.fc2 = nn.Linear(config.adapter_size, self.embed_dim)
       
    def forward(self,x,encoder_padding_mask):
        #Keep values for residual connection
        normalized = self.layer_norm(x)
        proj_down = self.relu(self.fc1(normalized))
        proj_up = self.fc2(proj_down)
        return proj_up