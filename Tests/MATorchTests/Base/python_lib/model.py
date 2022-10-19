import torch


class ModelTest1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 5, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=True),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            torch.nn.Conv2d(5, 10, kernel_size=(1, 1), stride=(2, 2), bias=True),
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
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight)

            if module.bias is not None:
                torch.nn.init.normal_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class ModelTest2(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.features1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 5, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            torch.nn.BatchNorm2d(5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=(1, 1), ceil_mode=False),
        )
        self.features2 = torch.nn.Sequential(
            torch.nn.Conv2d(5, 5, kernel_size=(3, 3), padding=(1, 1), bias=False),
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
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight)

            if module.bias is not None:
                torch.nn.init.normal_(module.bias)

        elif isinstance(module, torch.nn.BatchNorm2d):
            torch.nn.init.normal_(module.weight)
            torch.nn.init.normal_(module.bias)
            torch.nn.init.normal_(module.running_mean)
            torch.nn.init.uniform_(module.running_var)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features1(x)
        x = x + self.features2(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
