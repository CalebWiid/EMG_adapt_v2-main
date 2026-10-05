"""
CNN_model is the main script to train the CNN. 

A few things to note that differ from the paper: 
- The Neaural Network is uses Pytorch instead of Tensorflow
- Pytorch is fine to use as the weigh updates use a first order method so no need for higher order gradient machinery. 

The structure of the code is as follows: 
    - Convelutional block -> applies the filters and creates the feature maps.\
    - 

"""
import torch
import torch.nn as nn

def conv_block(in_channels: int, out_channels: int, kernel_size: int) -> nn.Sequential: 
    """ First conveluitional block. 
        - conv1d -> batchNorm1d -> relu 
        - No padding applied 
    """
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size, stride=1, padding=0),
        nn.BatchNorm1d(out_channels),
        nn.ReLU()
    )

def build_conv_extractor(in_channels: int = 1) -> nn.Sequential: 
    return nn.Sequential(
        nn.BatchNorm1d(in_channels), 
        conv_block(in_channels, 64, 3), 
        conv_block(64, 64, 3), 
        conv_block(64, 64, 1), 
        conv_block(64, 64, 1), 
    )    

def dense_block(in_features: int, out_features: int, activation: bool = True) -> nn.Sequential: 
    """ Build the fully connected blocks: 
    Linear -> BatchNorm1d -> ReLU (optional) 
    """
    layers = [nn.Linear(in_features, out_features), nn.BatchNorm1d(out_features)]
    if activation: 
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)

def build_embedding_head(flatten_dim: int, embedding_dim: int = 128) -> nn.Sequential: 
    """ Build the embedding head. 
    Flattern the conv maps -> 3 dense (FC) layers 
    """

    return nn.Sequential(
        nn.Flatten(), 
        dense_block(flatten_dim, 512),
        dense_block(512, 512), 
        dense_block(512, embedding_dim, activation=False),
    )

class EMGAdapt(nn.Module): 
    """ This is the full 1D CNN. 
    
    Inputs -> CCA feature matrix 
    Output -> 128 dimensional embedding vector 
    """
    def __init__(self, e: int, n_classes: int, embedding_dim: int = 128): 
        super().__init__()
        self.e = e
        self.n_classes = n_classes
        self.embedding_dim = embedding_dim
        flatten_dim = 64 * (e - 4)
        self.features = build_conv_extractor(in_channels=1)
        self.embedding = build_embedding_head(flatten_dim, embedding_dim=embedding_dim)
        self.classifier = nn.Linear(embedding_dim, n_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def embed(self, x: torch.Tensor) -> torch.Tensor: 
        """Return the 128-d embedding f_theta(x). x: (batch, e) or (batch, 1, e)."""
        if x.dim() == 2: 
            x = x.unsqueeze(1)  
        z = self.features(x)
        z = self.embedding(z)

        return z 

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        """Return (embedding, logits)"""
        z = self.embed(x) 
        logits = self.classifier(z) 

        return z, logits     

if __name__ == "__main__": 
    e = 16
    x = torch.rand(4, 1, e)
    block = conv_block(1, 64, 3)
    y = block(x)
    print("input : ", tuple(x.shape))
    print("output : ", tuple(y.shape))
    assert y.shape == (4, 64, e - 2) 
    print("STEP 1 OK")

    # step 2 
    for e in (16, 8): 
        x = torch.rand(4, 1, e)
        extractor = build_conv_extractor(in_channels=1)
        extractor.eval()
        y = extractor(x)
        print(f"STEP 2 extractor e = {e:>2}", tuple(x.shape), "->", tuple(y.shape))
        assert y.shape == (4, 64, e - 4), y.shape
    print("STEP 2 OK")

    #step 3 
    for e in (16, 8): 
        flatten_dim = 64 * (e - 4)
        net = nn.Sequential(build_conv_extractor(1), build_embedding_head(flatten_dim))
        net.eval()
        x = torch.rand(4, 1, e)
        z = net(x)
        print(f"STEP 3  embedding e={e:>2}: {tuple(x.shape)} -> {tuple(z.shape)}  "
        f"(flatten_dim={flatten_dim})")
        assert z.shape == (4, 128), z.shape
    print("STEP 3 OK")

    # Step 4 
    for e, n_classes in [(16, 53), (8, 12)]:      # DB5-like, then MindRove-like
        model = EMGAdapt(e=e, n_classes=n_classes)
        model.eval()
        x = torch.randn(4, e)                      # CCA feature matrix (batch, e)
        emb, logits = model(x)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"STEP 4  e={e:>2}, classes={n_classes:>2}: x{tuple(x.shape)} -> "
              f"embedding{tuple(emb.shape)}, logits{tuple(logits.shape)}  "
              f"| params={n_params:,}")
        assert emb.shape == (4, 128)
        assert logits.shape == (4, n_classes)
        # .embed() alone gives just the embedding (deployment path)
        assert model.embed(x).shape == (4, 128)
        # a pre-shaped (batch, 1, e) input also works
        assert model(torch.randn(4, 1, e))[1].shape == (4, n_classes)
    print("STEP 4 OK")

    # Step 5 
    import math
    model = EMGAdapt(e=16, n_classes=53)
    checked = 0
    for name, m in model.named_modules():
        if isinstance(m, (nn.Conv1d, nn.Linear)):
            fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = math.sqrt(6.0 / (fan_in + fan_out))          # Xavier-uniform range
            assert m.weight.abs().max().item() <= bound + 1e-6, name
            if m.bias is not None:
                assert m.bias.abs().max().item() == 0.0, name    # biases zeroed
            checked += 1
    print(f"STEP 5  checked {checked} Conv1d/Linear layers: all within Xavier bound, biases 0")
    print("STEP 5 OK")

    # step 6 
    torch.manual_seed(0)
    model = EMGAdapt(e=16, n_classes=5)                 # 5-way -> paper param count
    assert sum(p.numel() for p in model.parameters()) == 746_439   # regression guard
    model.train()
 
    x = torch.randn(32, 16)                                # dummy CCA batch (batch, e)
    y = torch.randint(0, 5, (32,))                         # dummy labels
    criterion = nn.CrossEntropyLoss()
 
    # one forward + backward: a gradient must reach every trainable tensor
    emb, logits = model(x)
    assert emb.shape == (32, 128) and logits.shape == (32, 5)
    loss0 = criterion(logits, y)
    model.zero_grad()
    loss0.backward()
    no_grad = [n for n, p in model.named_parameters() if p.requires_grad and p.grad is None]
    assert not no_grad, f"no gradient reached: {no_grad}"
    grad_norm = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
    assert grad_norm > 0 and all(torch.isfinite(p.grad).all()
                                 for p in model.parameters() if p.grad is not None)
    print(f"STEP 6  backward: gradient reached all {sum(1 for _ in model.parameters())} "
          f"tensors, total grad-norm={grad_norm:.3f}, all finite")
 
    # tiny overfit on the fixed batch: loss must fall -> the model can learn
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(100):
        opt.zero_grad()
        _, logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        opt.step()
    print(f"STEP 6  overfit loss: {loss0.item():.3f} -> {loss.item():.3f} (100 Adam steps)")
    assert loss.item() < 0.5 * loss0.item()