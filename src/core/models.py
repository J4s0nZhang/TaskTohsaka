import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models



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
