
import torch
from typing import Tuple, Union, List
import torchvision.transforms as T
from toDayGan.options.test_options import TestOptions
from toDayGan.models.combogan_model import ComboGANModel
from toDayGan.data.base_dataset import get_transform


class DeviceAgnosticColorJitter(T.ColorJitter):
    def __init__(self, brightness: float = 0., contrast: float = 0., saturation: float = 0., hue: float = 0.):
        """This is the same as T.ColorJitter but it only accepts batches of images and works on GPU"""
        super().__init__(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        assert len(images.shape) == 4, f"images should be a batch of images, but it has shape {images.shape}"
        B, C, H, W = images.shape
        # Applies a different color jitter to each image
        color_jitter = super(DeviceAgnosticColorJitter, self).forward
        augmented_images = [color_jitter(img).unsqueeze(0) for img in images]
        augmented_images = torch.cat(augmented_images)
        assert augmented_images.shape == torch.Size([B, C, H, W])
        return augmented_images


class DeviceAgnosticRandomResizedCrop(T.RandomResizedCrop):
    def __init__(self, size: Union[int, Tuple[int, int]], scale: float):
        """This is the same as T.RandomResizedCrop but it only accepts batches of images and works on GPU"""
        super().__init__(size=size, scale=scale)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        assert len(images.shape) == 4, f"images should be a batch of images, but it has shape {images.shape}"
        B, C, H, W = images.shape
        # Applies a different color jitter to each image
        random_resized_crop = super(DeviceAgnosticRandomResizedCrop, self).forward
        augmented_images = [random_resized_crop(img).unsqueeze(0) for img in images]
        augmented_images = torch.cat(augmented_images)
        return augmented_images

class ToDayTimeShift(torch.nn.Module):
    def __init__(self, domain: int):
        """This shift images either from day to night or from night to day (0 or 1) domains but it only accepts batches of images and works on GPU"""
        super().__init__()
        self.domain = domain
        self.opt = TestOptions().parse(["--phase", "test", "--serial_test", "--name", "Training", "--dataroot", "", "--n_domains", "2", "--which_epoch", "150"], verbose=False)
        self.opt.nThreads = 1   # test code only supports nThreads = 1
        self.opt.batchSize = 1  # test code only supports batchSize = 1
        self.transform = get_transform(self.opt, True)
        self.model = ComboGANModel(self.opt, False)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        assert len(images.shape) == 4, f"images should be a batch of images, but it has shape {images.shape}"

        augmented_images = [self.__apply_domain_shift(self.transform(img)).unsqueeze(0) for img in images]
        augmented_images = torch.cat(augmented_images)
        return augmented_images
        
    def __apply_domain_shift(self, img: torch.Tensor) -> torch.Tensor:
        self.model.set_input({'A': img, 'DA': [self.domain], 'path': ''})
        self.model.test()
        return self.model.get_generated_tensors(self.domain)[0]

class RandomAugmentation(torch.nn.Module):
    def __init__(self, augmentations: List[torch.nn.Module], p: float):
        """This augmentation combines two augmentations choosing the first over the second based on a threshold probability p, but it only accepts batches of images and works on GPU"""
        assert len(augmentations) == 2, f"Only two possible augmentations but {len(augmentations)} were passed"

        super().__init__()
        self.augmentations = augmentations
        self.p = p
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        assert len(images.shape) == 4, f"images should be a batch of images, but it has shape {images.shape}"

        augmentation_index = 1 if torch.rand(1).item() < self.p else 0
        return self.augmentations[augmentation_index].forward(images)


if __name__ == "__main__":
    """
    You can run this script to visualize the transformations, and verify that
    the augmentations are applied individually on each image of the batch.
    """
    from PIL import Image
    # Import skimage in here, so it is not necessary to install it unless you run this script
    from skimage import data
    
    # Initialize DeviceAgnosticRandomResizedCrop
    random_crop = DeviceAgnosticRandomResizedCrop(size=[256, 256], scale=[0.5, 1])
    # Create a batch with 2 astronaut images
    pil_image = Image.fromarray(data.astronaut())
    tensor_image = T.functional.to_tensor(pil_image).unsqueeze(0)
    images_batch = torch.cat([tensor_image, tensor_image])
    # Apply augmentation (individually on each of the 2 images)
    augmented_batch = random_crop(images_batch)
    # Convert to PIL images
    augmented_image_0 = T.functional.to_pil_image(augmented_batch[0])
    augmented_image_1 = T.functional.to_pil_image(augmented_batch[1])
    # Visualize the original image, as well as the two augmented ones
    pil_image.show()
    augmented_image_0.show()
    augmented_image_1.show()
