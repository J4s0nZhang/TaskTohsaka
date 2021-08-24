import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SiameseNet(nn.Module):
    """
    Siemese Neural Network with resnet 50 backbone for convolution section
    """
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 10),  
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  
            nn.Conv2d(64, 128, 7),
            nn.ReLU(),    
            nn.MaxPool2d(2),   
            nn.Conv2d(128, 128, 4),
            nn.ReLU(), 
            nn.MaxPool2d(2), 
            nn.Conv2d(128, 256, 4),
            nn.ReLU(),   
        )
        self.linear = nn.Sequential(nn.Linear(9216, 4096), nn.Sigmoid())
        self.out = nn.Linear(4096, 1) # no final sigmoid layer, applied by BCE with logits loss
    
    def forward_path(self, x):
        x = self.conv(x)
        x = x.view(x.size()[0], -1) 
        x = self.linear(x)
        return x 

    def forward(self, x1, x2):
        encoded_l = self.forward_path(x1)
        encoded_r = self.forward_path(x2)
        abs_diff = torch.abs(encoded_l - encoded_r)
        out = self.out(abs_diff)
        return out 

class ConvNet(nn.Module):
    """
    Convolutional Neural Network with resnet50 backbone.
    Supports Imprinted Weights Low-shot learning architecture for small amount of
    classes as well. 
    """
    def __init__(self, n_classes=3, in_features=2048,d_emb=0):
        super().__init__()
        self.n_classes = n_classes
        self.d_emb = d_emb
        self.extractor = Extractor()
        if d_emb != 0:
            self.embedding = Embedding(in_features, d_emb, False)
            self.classifier = nn.Linear(d_emb, n_classes)
        else:
            self.classifier = nn.Linear(in_features, n_classes)
        
    def forward(self, x):
        x = self.extractor(x)
        if self.d_emb != 0:
            x = self.embedding(x)
        x = self.classifier(x)
        return x 

    def extract(self, x):
        x = self.extractor(x)
        if self.d_emb > 0:
            x = self.embedding(x)
        return x
        
class Extractor(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        basenet = models.resnet50(pretrained=pretrained)
        self.extractor = nn.Sequential(*list(basenet.children())[:-1])

    def forward(self, x):
        x = self.extractor(x)
        x = x.view(x.size(0), -1)
        return x

class Embedding(nn.Module):
    def __init__(self, in_features=2048, d_emb=64, normalize=False):
        super().__init__()
        self.normalize = normalize
        self.d_emb = d_emb
        self.fc = nn.Linear(in_features, d_emb)

    def forward(self, x):
        x = self.fc(x)
        if self.normalize:
            x = F.normalize(x, p=2, dim=1)
        return x

class Classifier(nn.Module):
    def __init__(self, n_classes, d_emb=64, normalize=False, bias=True):
        super().__init__()
        self.n_classes = n_classes
        self.d_emb = d_emb
        self.normalize = normalize
        self.fc = nn.Linear(d_emb, n_classes, bias=bias)

    def forward(self, x):
        if self.normalize:
            w = self.fc.weight
            w = F.normalize(w, dim=1, p=2)
            x = F.linear(x, w)
        else: 
            x = self.fc(x)
        return x
