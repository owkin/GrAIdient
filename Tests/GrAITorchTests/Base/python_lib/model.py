import torch
import numpy as np


class ModelTest1(torch.nn.Module):
    """
    Model to test.
    Principle features:
        - Convolution with stride and biases
        - MaxPool with ceil_model
        - Linear with biases
    """

    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(
                3, 5,
                kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                bias=True
            ),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(
                kernel_size=2, stride=2,
                ceil_mode=True
            ),
            torch.nn.Conv2d(
                5, 10,
                kernel_size=(1, 1), stride=(2, 2),
                bias=True
            ),
            torch.nn.ReLU(),
        )
        self.avgpool = torch.nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=490, out_features=10),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=10, out_features=1)
        )
        self.features.apply(self.weight_init)
        self.classifier.apply(self.weight_init)

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
           isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight)

            if module.bias is not None:
                torch.nn.init.normal_(module.bias)

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
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class ModelTest2(torch.nn.Module):
    """
    Model to test.
    Principle features:
        - Convolution with batch normalization and no biases
        - MaxPool with overlapping and no ceil_mode
        - ResNet like shortcut
    """

    def __init__(self):
        super().__init__()
        self.features1 = torch.nn.Sequential(
            torch.nn.Conv2d(
                3, 5,
                kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                bias=False
            ),
            torch.nn.BatchNorm2d(5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(
                kernel_size=3, stride=2, padding=(1, 1),
                ceil_mode=False
            ),
        )
        self.features2 = torch.nn.Sequential(
            torch.nn.Conv2d(
                5, 5,
                kernel_size=(3, 3), padding=(1, 1),
                bias=False
            ),
            torch.nn.BatchNorm2d(5),
            torch.nn.ReLU(),
        )
        self.avgpool = torch.nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=245, out_features=10),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=10, out_features=1)
        )
        self.features1.apply(self.weight_init)
        self.features2.apply(self.weight_init)
        self.classifier.apply(self.weight_init)

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
                isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight)

            if module.bias is not None:
                torch.nn.init.normal_(module.bias)

        elif isinstance(module, torch.nn.BatchNorm2d):
            torch.nn.init.normal_(module.weight)
            torch.nn.init.normal_(module.bias)
            torch.nn.init.normal_(module.running_mean)
            torch.nn.init.uniform_(module.running_var)

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
        x = self.features1(x)
        x = x + self.features2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def _fft2d_freqs(h, w):
    """
    Compute 2d spectrum frequences.
    """
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[:]
    return np.sqrt(fx*fx + fy*fy)


def _irfft_image(x):
    """
    Compute Inverse of Fast Fourier Transform.
    """
    real = torch.concat([
        x[0, 0][None],
        x[0, 2][None],
        x[0, 4][None]
    ], dim=0)
    img = torch.concat([
        x[0, 1][None],
        x[0, 3][None],
        x[0, 5][None]
    ], dim=0)
    x = torch.complex(real, img)
    x = torch.fft.ifft2(x).real
    return x/4.


def _linear_decorrelate_color(x):
    """
    Multiply input by sqrt of empirical (ImageNet) color correlation matrix.
    """
    color_correlation_svd_sqrt = np.asarray([
        [0.26, 0.09, 0.02],
        [0.27, 0.00, -0.05],
        [0.27, -0.09, 0.03]]
    ).astype("float32")
    max_norm_svd_sqrt = np.max(
        np.linalg.norm(color_correlation_svd_sqrt, axis=0)
    )

    x = torch.transpose(x, 0, 1)
    x = torch.transpose(x, 1, 2)
    x_flat = torch.reshape(x, [-1, 3])
    color_correlation_normalized = \
        color_correlation_svd_sqrt / max_norm_svd_sqrt
    color_correlation_normalized = torch.Tensor(
        color_correlation_normalized.T.astype("float32")
    )
    x_flat = torch.matmul(x_flat, color_correlation_normalized)
    x = torch.reshape(x_flat, x.shape)
    x = torch.transpose(x, 1, 2)
    x = torch.transpose(x, 0, 1)
    return x


class ModelTest3(torch.nn.Module):
    """
    Model to test.
    Principle features:
        - 2D Frequences & scale
        - IRFFT
        - Decorrelate color
    """

    def __init__(self, size):
        super().__init__()
        freqs = _fft2d_freqs(size, size)
        scale = 1.0 / np.maximum(freqs, 1.0 / max(size, size))
        scale *= np.sqrt(size * size)
        self.scale = torch.Tensor(scale.astype("float32"))

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
        x = x * self.scale
        x = _irfft_image(x)
        x = _linear_decorrelate_color(x)
        x = torch.nn.Sigmoid()(x)
        x = -1 + 2 * x
        x = x[None]
        return x
