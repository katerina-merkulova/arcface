from pathlib import Path

import numpy as np
import torchvision
from PIL import Image
from torch.utils import data
from torchvision import transforms as T


class Dataset(data.Dataset):

    def __init__(self, root, phase='train', input_shape=(1, 128, 128)):
        self.phase = phase
        self.input_shape = input_shape

        self.imgs = list(root.iterdir())
        # self.imgs = np.random.permutation(imgs)

        normalize = T.Normalize(mean=[0.5, 0.5, 0.5],
                                std=[0.5, 0.5, 0.5])

        # normalize = T.Normalize(mean=[0.5], std=[0.5])

        if self.phase == 'train':
            self.transforms = T.Compose([
                T.RandomCrop(self.input_shape[1:]),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                normalize
            ])
        else:
            self.transforms = T.Compose([
                T.CenterCrop(self.input_shape[1:]),
                T.ToTensor(),
                normalize
            ])

    def __getitem__(self, index):
        img_path = self.imgs[index]
        data = Image.open(img_path)
        data = self.transforms(data)
        label = np.int32(img_path.stem.split('_')[0])
        return data.float(), label

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    dataset = Dataset(Path.cwd().parent / 'Market-1501-v15.09.15' / 'bounding_box_train',
                      phase='test',
                      input_shape=(3, 64, 128))

    trainloader = data.DataLoader(dataset, batch_size=64)
    for i, (imgs, labels) in enumerate(trainloader):
        # print(imgs.numpy().shape)
        # print(imgs.numpy())
        img = torchvision.utils.make_grid(imgs).numpy()
        # chw -> hwc
        img = np.transpose(img, (1, 2, 0))
        # img *= np.array([0.229, 0.224, 0.225])
        # img += np.array([0.485, 0.456, 0.406])
        img += np.array([1, 1, 1])
        img *= 127.5
        img = img.astype(np.uint8)
        img = img[:, :, [2, 1, 0]]

        # Image.fromarray(img).show()

        # break
        # dst.decode_segmap(labels.numpy()[0], plot=True)
