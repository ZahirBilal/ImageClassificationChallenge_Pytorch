import glob
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, models


class ChristmasImages(Dataset):

    def __init__(self, path, training=True):
        super().__init__()
        self.training = training
        # If training == True, path contains subfolders
        # containing images of the corresponding classes

        if training == True:
            self.path = path

            # Get all image paths from a directory

            self.image_paths = glob.glob(self.path + "*")
            self.data = []
            for image_path in self.image_paths:
                for path in glob.glob(image_path + "**\\*.png", recursive=True):
                    self.data.append(path)

            # Get the labels from the image paths

            self.labels = [x.split("/")[-2] for x in self.data]

            self.labels = ["0" if x ==
                           "christmas_cookies" else x for x in self.labels]
            self.labels = ["1" if x ==
                           "christmas_presents" else x for x in self.labels]
            self.labels = ["2" if x ==
                           "christmas_tree" else x for x in self.labels]
            self.labels = ["3" if x ==
                           "fireworks" else x for x in self.labels]
            self.labels = ["4" if x ==
                           "penguin" else x for x in self.labels]
            self.labels = ["5" if x ==
                           "reindeer" else x for x in self.labels]
            self.labels = ["6" if x ==
                           "santa" else x for x in self.labels]
            self.labels = ["7" if x ==
                           "snowman" else x for x in self.labels]

            # Create a dictionary mapping each label to a index.
            self.label_to_idx = {'0': 0, '1': 1, '2': 2,
                                 '3': 3, '4': 4, '5': 5, '6': 6, '7': 7}

            # Set self.transform to transform

            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5],  # 0-1 to [-1,1] , formula (x-mean)/std
                                     [0.5, 0.5, 0.5])
            ])

        # If training == False, path directly contains
        # the test images for testing the classifier

        elif training == False:

            self.path = path

            # Get all image paths from a directory

            self.image_paths = glob.glob(self.path + "\\*.png")
            # For Kaggle
            # self.image_paths.sort()
            # self.image_paths.sort(key=len)

            # Set self.transform to transform

            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5],  # 0-1 to [-1,1] , formula (x-mean)/std
                                     [0.5, 0.5, 0.5])
            ])

    def __len__(self):
        # return length of dataset

        if self.training == True:
            return len(self.data)

        elif self.training == False:
            return len(self.image_paths)

    def __getitem__(self, index):
        # If self.training == False, output (image, )
        # where image will be used as input for your model
        # raise NotImplementedError

        if self.training == True:
            # open and send one image and label

            img_name = self.data[index]
            label = self.labels[index]
            image = Image.open(img_name).convert('RGB')

            if self.transform:
                image = self.transform(image)

            return image, self.label_to_idx[label]

        elif self.training == False:
            # open and send one image

            img_name = self.image_paths[index]
            image = Image.open(img_name).convert('RGB')

            if self.transform:
                image = self.transform(image)

            image.unsqueeze_(0)

            return image
