import torch
from torch import nn
from torch.nn import functional as F


class ConvModule(nn.Module):
    """Module consists of several convolutional layers with different kernel sizes"""
    def __init__(self, emb_dim, feature_maps, filters_per_region = 2, drop_prob = 0.5, *region_sizes):
        super(ConvModule, self).__init__()
        layers = []
        for size in region_sizes:
            for filter in range(filters_per_region):
                layers.append(nn.Conv1d(in_channels = 1, out_channels = feature_maps, kernel_size=(size, emb_dim)))
        self.layers = nn.Sequential(*layers)
        self.activ = nn.ReLU()
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, inputs):
        results = None
        for layer in self.layers:
            x = self.activ(layer(inputs)) # [Batch_Size x Feature_Maps x (MaxLen - Kernel+1 ) x 1]
            x = x.squeeze(3) #  [Batch_Size x (Feature_Maps x (MaxLen - Kernel+1))//Emb_Size]
            x = F.max_pool1d(x, x.size(2)).squeeze(2)
            if results == None:
                results = x
            else:
                results = torch.cat([results,x], 1)
            out = self.dropout(results) # [BatchSize x Feature_Map * (len(self.layers)) ]
            return out


class ConvModel(nn.Module):
    """Convolutional model with embeddings"""
    def __init__(self, vocab_size, num_classes, emb_size=32, pad_idx = 0, kernel_size=3, feature_maps = 100, drop_prob=0.5):
        super(ConvModel, self).__init__()
        self.region_sizes = [2,3,4,5]
        self.emb = nn.Embedding(vocab_size, emb_size, padding_idx=pad_idx)
        self.convolutions = ConvModule(emb_size, 100, 2, drop_prob, *self.region_sizes)
        self.fc = nn.Linear(feature_maps*len(self.region_sizes) * 2, num_classes)

    def forward(self, inputs):
        embeds = self.emb(inputs).unsqueeze(1) # [Batch_size x 1 x MaxLen x EmbSize]
        features = self.convolutions(embeds)
        output = self.fc(features)
        return output