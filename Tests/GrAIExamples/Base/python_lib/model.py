import torch


class SimpleAutoEncoder(torch.nn.Module):
    """
    Simple auto encoder model.
    """

    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(
                3, 12,
                kernel_size=3, stride=2, padding=1,
                bias=True
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                12, 24,
                kernel_size=3, stride=2, padding=1,
                bias=True
            ),
            torch.nn.ReLU(),
            torch.nn.Conv2d(
                24, 48,
                kernel_size=3, stride=2, padding=1,
                bias=True
            ),
            torch.nn.ReLU(),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(48, 24, kernel_size=2, stride=2),
            torch.nn.ConvTranspose2d(24, 12, kernel_size=2, stride=2),
            torch.nn.ConvTranspose2d(12, 3, kernel_size=2, stride=2),
            torch.nn.Sigmoid(),
        )

        self.encoder.apply(self.weight_init)
        self.decoder.apply(self.weight_init)

    @staticmethod
    def weight_init(module: torch.nn.Module):
        """
        Initialize weights and biases.

        Parameters
        ----------
        module: torch.nn.Module
            The module to initialize.
        """
        if isinstance(module, torch.nn.Conv2d) or \
           isinstance(module, torch.nn.ConvTranspose2d) or \
           isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_normal_(module.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            The input tensor.

        Returns
        -------
        _: torch.Tensor
            The output tensor.
        """
        x = self.encoder(x)
        x = self.decoder(x)
        return x
