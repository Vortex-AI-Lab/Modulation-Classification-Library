import torch
from torch import nn


class MCformer(nn.Module):
    """`MCformer <https://ieeexplore.ieee.org/abstract/document/9685815>`_ backbone
    The input for MCformer is a 1*2*L frame
    Args:
        frame_length (int): the frame length equal to number of sample points
        num_classes (int): number of classes for classification.
            The default value is -1, which uses the backbone as
            a feature extractor without the top classifier.
    """

    def __init__(
        self,
        fea_dim: int = 32,
        frame_length: int = 128,
        num_classes: int = -1,
        init_cfg=None,
    ) -> None:
        super(MCformer, self).__init__(init_cfg)
        self.frame_length = frame_length
        self.num_classes = num_classes
        self.cnn = nn.Sequential(
            nn.Conv1d(2, fea_dim, kernel_size=65, padding="same"),
            nn.ReLU(inplace=True),
        )
        self.fea_dim = fea_dim

        # Create one transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            fea_dim, 4, dim_feedforward=fea_dim, batch_first=True
        )
        # Stack multiple layers to create the transformer encoder
        self.tnn = nn.TransformerEncoder(encoder_layer, num_layers=4)

        if self.num_classes > 0:
            self.classifier = nn.Sequential(
                nn.Linear(4 * self.fea_dim, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(128, self.num_classes),
            )

    def forward(self, x_enc: torch.FloatTensor) -> tuple:
        x_enc = self.cnn(x_enc)
        x_enc = torch.squeeze(x_enc, dim=2)
        x_enc = torch.transpose(x_enc, 1, 2)
        x_enc = self.tnn(x_enc)
        x_enc = x_enc[:, :4, :]
        x_enc = torch.reshape(x_enc, [-1, 4 * self.fea_dim])
        if self.num_classes > 0:
            x_enc = self.classifier(x_enc)

        return x_enc