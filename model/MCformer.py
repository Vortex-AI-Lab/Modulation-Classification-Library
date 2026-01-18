import torch
from torch import nn


class Model(nn.Module):
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
        configs,
    ) -> None:
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.n_classes = configs.n_classes

        # The demension of features model and feedforward in transformer
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        # The number of heads in multi-head attention
        self.n_heads = configs.n_heads

        # The number of the transformer layers
        self.n_layers = configs.n_layers

        # The rate of dropout layers
        self.dropout = configs.dropout

        # Create the CNN layer for embedding
        self.embedding = nn.Sequential(
            nn.Conv1d(
                in_channels=2, out_channels=self.d_model, kernel_size=65, padding="same"
            ),
            nn.ReLU(inplace=True),
        )

        # Create one transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            self.d_model, self.n_heads, dim_feedforward=self.d_ff, batch_first=True
        )

        # Stack multiple layers to create the transformer encoder
        self.backbone = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)

        # TODO: 后续可以统一去整合classifier部分
        self.classifier = nn.Sequential(
            nn.Linear(4 * self.d_model, self.d_ff),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.d_ff, self.n_classes),
        )

    def forward(self, x_enc: torch.FloatTensor) -> tuple:
        # Get the embedding of data
        x_enc = self.embedding(x_enc)
        x_enc = torch.squeeze(x_enc, dim=2)
        x_enc = torch.transpose(x_enc, 1, 2)

        # Pass through the transformer encoder layers
        x_dec = self.backbone(x_enc)

        # Get the outputs of the first 4 tokens
        x_dec = x_dec[:, :4, :]
        x_dec = torch.reshape(x_dec, [-1, 4 * self.d_model])

        # Classifier
        return self.classifier(x_dec)


if __name__ == "__main__":

    class Configs:
        seq_len = 128
        n_classes = 11
        d_model = 64
        d_ff = 256
        n_heads = 8
        n_layers = 4
        dropout = 0.1

    model = Model(configs=Configs())

    inputs = torch.rand((4, 2, 128))
    outputs = model(inputs)

    print(outputs.shape)
