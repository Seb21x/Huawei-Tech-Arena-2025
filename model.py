import torch
from torch import nn
import torch.nn.functional as F


class SentenceEncoder(nn.Module):
    def __init__(self, input_size=768, hidden_size=128, output_size=256):
        super().__init__()

        # projection layer, squeeze data to hidden_size
        self.proj = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=False),
            # normalize the output from the Linear layer
            nn.LayerNorm(hidden_size),
            # GELU as activation function
            nn.GELU(),
        )

        # attention pooling, assign weights to tokens
        self.att = nn.Linear(hidden_size, 1, bias=False)

        # final projection to output
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, output_size, bias=False),
            nn.LayerNorm(output_size),  # normalization at the end
        )

        # initialize weights
        self._init_weights()

    def _init_weights(self):
        """
        Helper function to set initial weights.
        """
        # iterate over all layers
        for module in self.modules():
            # if Linear, use Xavier
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            # if LayerNorm, weight = 1, bias = 0
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)

    def forward(self, x):
        """
        Input `x` has shape: (Batch_Size, Tokens, Input_Size)
        """

        # padding mask
        mask = (x.sum(-1) != 0)

        # projection layer
        h = self.proj(x)

        # attention layer
        att_logits = self.att(h).squeeze(-1)

        # masking
        att_logits = att_logits.masked_fill(~mask, -1e9)

        # change scores to percentages (weights)
        att_weights = F.softmax(att_logits, dim=-1)

        # Attention Pooling
        # Multiply each token vector (h) by its weight (att_weights) and sum everything.
        att_pooled = torch.einsum('bsd,bs->bd', h, att_weights)

        # Mean Pooling
        mask_sum = mask.sum(dim=1, keepdim=True).clamp(min=1)
        mean_pooled = (h * mask.unsqueeze(-1)).sum(dim=1) / mask_sum

        # mixing results, 70% from attention and 30% from mean
        pooled = 0.7 * att_pooled + 0.3 * mean_pooled

        # output layer
        output = self.fc(pooled)

        # L2 normalization
        return F.normalize(output, p=2, dim=-1)


def get_model():
    return SentenceEncoder()