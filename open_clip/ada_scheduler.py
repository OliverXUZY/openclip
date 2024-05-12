import torch
import torch.nn as nn

from dataclasses import dataclass

class LatencyEncoder(nn.Module):
    # Embed scalar to 32 dimensions

    def __init__(self, out_dim = 32, B_key = None, gaussian_scale = 1.):
        super(LatencyEncoder, self).__init__()
        
        # self.device = torch.device("cpu")  # Default device
        
        if B_key is None:
            print("B-key is None")
            self.B = None
            in_dim = 1
        elif B_key == "basic":
            self.B = torch.tensor(1.0)  # Make B a tensor
            in_dim = 2
        elif B_key == "gaussian":
            B_gauss = torch.randn(16)  ## m = 16 so embedding size of eta(v) wil be 32, (eq5) of https://arxiv.org/pdf/2006.10739.pdf
            self.B = B_gauss * gaussian_scale # Make B a tensor
            in_dim = out_dim
        else:
            raise NotImplementedError("B_key functionality not implemented.")
        if self.B is not None:
            self.B = self.B.cuda()
        
        self.fc = nn.Linear(in_dim, out_dim)  # Embed scalar to 32 dimensions
    
    # Fourier feature mapping
    def input_mapping(self, x, B):
        '''
        Args: 
            x (Tensor): batch of latencies, shape [bs, ].
            B (Tensor): embedding vector or scalar, shape [d,] or scalar.
        '''
        if B is None:
            return x
        else:
            # print(x.device)
            # print(B.device)
            # assert False
            x_proj = (2.*torch.pi*x) * B
            # print("np.sin(x_proj): ", np.sin(x_proj))
            return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], -1)

    def forward(self, x):
        # print("self.B", self.B.device)
        x = self.input_mapping(x, self.B)
        x = self.fc(x)
        return x

@dataclass
class ada_SchedulerCfg:
    latency_dim: int = 32
    latency_Bkey: str = None 
    latency_gaussian_scale: float = 1.0
    
    content_inp_dim: int = 768 ## feature received from first a few layers of CLIP vision enc
    content_dim: int = 128


class ada_Scheduler(nn.Module):
    def __init__(self, ada_ScheCfg: ada_SchedulerCfg):
        super().__init__()
        self.n_knobs = 11  ## hard code for now
        self.latency_encoder = LatencyEncoder(
            ada_ScheCfg.latency_dim,
            ada_ScheCfg.latency_Bkey,
            ada_ScheCfg.latency_gaussian_scale,
        )
        self.content_encoder = nn.Linear(ada_ScheCfg.content_inp_dim, ada_ScheCfg.content_dim, bias = False)

        # Combine image and scalar features
        self.combined_fc = nn.Sequential(
            nn.Linear(ada_ScheCfg.latency_dim + ada_ScheCfg.content_dim, 256),
            # nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),  # Adjust dropout rate as needed
            nn.Linear(256, 64),
            # nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),  # Adjust dropout rate as needed
        )

        self.out = nn.Linear(64, self.n_knobs)
    
    def construct_embeddings(self, image_features, scalar_embedding):
        """
        Args:
            image_features (float tensor, (bs, ada_ScheCfg.content_dim)): feature maps.
            scalar_embedding (float tensor, (bs, ada_ScheCfg.latency_dim)): latency for each input.
        """
        return torch.cat((image_features, scalar_embedding), dim=1)

    def forward(self, x, latency):
        """
        Args:
            x (torch.float16, [bs, dim]): input features. [4(64), 768]
            latency (float tensor, (bs,)): latency for each input.
        """
        print(x.device, x.shape)
        print(latency.device, latency.shape)
        x = self.content_encoder(x)
        latency = self.latency_encoder(latency.view(-1, 1))
        print("x.shape: ", x.shape)
        print("latency.shape: ", latency.shape)
        # assert False

        embeddings = self.construct_embeddings(x, latency)
        # print("embeddings.shape: ", embeddings.shape)
        # assert False

        hidden = self.combined_fc(embeddings)
        # print("hidden.shape: ", hidden.shape)
        logits = self.out(hidden)
        print("logits.shape: ", logits.shape)

        return logits