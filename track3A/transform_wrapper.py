from albumentations.pytorch import ToTensorV2


class Atw():
    def __init__(self, aug):
        self.aug = aug
        self.to_tenor = ToTensorV2()

    def __call__(self, tensor):
        img = tensor.permute(1, 2, 0).numpy()
        img =  self.aug(image=img)["image"]
        tensor = self.to_tenor(image=img)["image"]
        return tensor

    def __repr__(self):
        repr = f"{self.__class__.__name__  }(aug={self.aug})"
        return repr