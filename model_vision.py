import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# ResNet Implementation (Adapted for CIFAR-10: 32x32 input)
# ============================================================================

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # CIFAR-10 specific: 3x3 conv, stride 1, no maxpool
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, targets=None):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        logits = self.linear(out)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss
    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Separate parameters for Muon (Matrices) and AdamW (Vectors).
        """
        muon_params = []
        adam_params = []
        
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
                
            # Conv2d weights (4D) and Linear weights (2D) -> Muon
            # BatchNorm weights (1D), Biases (1D) -> AdamW
            if p.ndim >= 2:
                muon_params.append(p)
            else:
                adam_params.append(p)
                
        optim_groups = [
            {'params': muon_params, 'use_muon': True, 'lr': 0.02, 'weight_decay': weight_decay, 'momentum': 0.95},
            {'params': adam_params, 'use_muon': False, 'lr': learning_rate, 'weight_decay': 0.0}
        ]
        return optim_groups

def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])

# ============================================================================
# Vision Transformer (ViT) Implementation (Small for CIFAR)
# ============================================================================

class ViT(nn.Module):
    def __init__(self, image_size=32, patch_size=4, num_classes=10, dim=512, depth=6, heads=8, mlp_dim=1024, dropout=0.1, emb_dropout=0.1):
        super().__init__()
        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2
        
        self.patch_size = patch_size
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        self.to_cls_token = nn.Identity()
        
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img, targets=None):
        p = self.patch_size
        
        # Extract patches
        # (B, C, H, W) -> (B, N, P*P*C)
        x = img.unfold(2, p, p).unfold(3, p, p)
        x = x.permute(0, 2, 3, 1, 4, 5).contiguous()
        x = x.view(x.size(0), -1, p * p * 3)
        
        # Embed patches
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # Add position embedding
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        
        # Transformer
        x = self.transformer(x)
        
        # Classification head
        x = self.to_cls_token(x[:, 0])
        logits = self.mlp_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """
        Separate parameters for Muon (Matrices) and AdamW (Vectors).
        """
        muon_params = []
        adam_params = []
        
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
                
            # Linear weights (2D) -> Muon
            # LayerNorm, Biases, Embeddings (1D/2D but treated as vectors often) -> AdamW
            # Note: ViT embeddings are 2D (1, N, D) or (N, D). 
            # Muon paper suggests keeping embeddings in AdamW or treating them carefully.
            # Here we'll put Linear layers (ndim=2) in Muon, and everything else in AdamW.
            # Specifically, we want QKV projections and MLP weights in Muon.
            
            if p.ndim == 2 and 'embedding' not in name and 'cls_token' not in name and 'pos_embedding' not in name:
                muon_params.append(p)
            else:
                adam_params.append(p)
                
        optim_groups = [
            {'params': muon_params, 'use_muon': True, 'lr': 0.02, 'weight_decay': weight_decay, 'momentum': 0.95},
            {'params': adam_params, 'use_muon': False, 'lr': learning_rate, 'weight_decay': 0.0}
        ]
        return optim_groups
