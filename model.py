import csv
import glob
import pathlib
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.autograd import Variable
from torchvision import transforms, models
from PIL import Image
from data import ChristmasImages


class Network(nn.Module):

    def __init__(self):
        super().__init__()

        # #############################
        # # Initialize your network
        # #############################

        # Checking for device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        self.model = models.resnet50(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False

        self.model.fc = nn.Sequential(nn.Linear(2048, 8),
                                      nn.LogSoftmax(dim=1))

        self.model.eval().to(self.device)

    def forward(self, x):

        #############################
        # Implement the forward pass
        #############################

        x = x
        # print('Shape of x in forward:', x.shape)

        x = self.model(x)
        # print('Shape of x after self.model(x) in forward: ', x.shape)

        return x

    def train_model(self):

        # #############################
        # # Load the Data
        # #############################

        data_path = "DL\\uebung8_challenge\\data"

        # training_data

        training_dataset = ChristmasImages(
            path=data_path + '\\train\\',
            training=True
        )

        # Split the training_dataset into a train and test datset

        training_data_size = 0.8

        train_size = int(training_data_size * len(training_dataset))
        test_size = len(training_dataset) - train_size

        train_dataset, test_dataset = torch.utils.data.random_split(
            training_dataset, [train_size, test_size])

        self.train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=128,
                                                        num_workers=0, shuffle=True)  # batch size 64
        self.test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=128,
                                                       num_workers=0, shuffle=False)  # batch size 32

        # #############################
        # # Train the Model
        # #############################

        # Optimizer
        self.optimizer = Adam(self.model.parameters(), lr=0.0001)

        # Loss function
        self.loss_function = nn.CrossEntropyLoss()

        # Number of epochs
        self.num_epochs = 10

        # calculating the size of training and testing images
        self.train_count = len(self.train_loader.dataset)
        self.test_count = len(self.test_loader.dataset)

        # Model training and saving best model

        self.best_accuracy = 0.0
        for epoch in range(self.num_epochs):

            # Evaluation and training on training dataset
            self.model.train()
            self.train_accuracy = 0.0
            self.train_loss = 0.0

            for data in self.train_loader:
                images, labels = data
                if torch.cuda.is_available():
                    images = images.cuda()
                    labels = labels.cuda()
#                     images = Variable(images.cuda())
#                     labels = Variable(labels.cuda())

#                 print('training images shape: ', images.shape) ###########################################
                self.optimizer.zero_grad()

                self.outputs = self.model(images)

#                 print('training outputs shape: ', self.outputs.shape) ######################################
                self.loss = self.loss_function(self.outputs, labels)
                self.loss.backward()
                self.optimizer.step()

                self.train_loss += self.loss.cpu().data*images.size(0)
                self._, self.prediction = torch.max(self.outputs.data, 1)

                self.train_accuracy += int(
                    torch.sum(self.prediction == labels.data))

            self.train_accuracy = self.train_accuracy/self.train_count
            self.train_loss = self.train_loss/self.train_count

        # Evaluation on testing dataset
        self.model.eval()

        self.test_accuracy = 0.0
        for data in self.test_loader:
            images, labels = data
            if torch.cuda.is_available():
                images = images.cuda()  # Vairable()
                labels = labels.cuda()  # Variable()

#             print('testing images shape: ', images.shape)###########################################
            self.outputs = self.model(images)
#             print('testing output shape: ', self.outputs.shape)#################################
            self._, self.prediction = torch.max(self.outputs.data, 1)
            self.test_accuracy += int(
                torch.sum(self.prediction == labels.data))

        self.test_accuracy = self.test_accuracy/self.test_count

        print('Epoch: '+str(epoch)+' Train Loss: '+str(self.train_loss) +
              ' Train Accuracy: '+str(self.train_accuracy)+' Test Accuracy: '+str(self.test_accuracy))

        # Save the best model
        if self.test_accuracy > self.best_accuracy:
            Network.save_model(self)
            self.best_accuracy = self.test_accuracy

    def save_model(self):
        #############################
        # Saving the model's weitghts
        # Upload 'model' as part of
        # your submission
        # Do not modify this function
        #############################
        torch.save(self.state_dict(), 'model')


net = Network()

net.eval()

net = net.train_model()

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

checkpoint = torch.load('model', map_location=device)
net.load_state_dict(checkpoint)
net.eval().to(device)

train_path = "DL\\uebung8_challenge\\data\\train"
pred_path = "DL\\uebung8_challenge\\data\\val"

# categories
root = pathlib.Path(train_path)
classes = sorted([j.name.split('\\')[-1] for j in root.iterdir()])
res = ['0', '1', '2', '3', '4', '5', '6', '7']
classes = res


# Transforms

transformer = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),  # 0-255 to 0-1, numpy to tensors
    transforms.Normalize([0.5, 0.5, 0.5],  # 0-1 to [-1,1] , formula (x-mean)/std
                         [0.5, 0.5, 0.5])
])

# prediction function


def prediction(img_path, transformer):

    image = Image.open(img_path).convert('RGB')  # convert input imgs to RGB

    image_tensor = transformer(image).float()

    image_tensor = image_tensor.unsqueeze_(0)

    image_tensor = image_tensor.to('cpu')

    image_tensor = image_tensor.to(device)

    input = image_tensor

    output = net.model(input)

    index = output.cpu().data.numpy().argmax()

    pred = classes[index]

    return pred


images_path = glob.glob(pred_path+'\\*.png')
images_path.sort()
images_path.sort(key=len)

pred_dict = {}

for i in images_path:
    pred_dict[i.replace('.png', '')[i.rfind('\\')+1:]
              ] = prediction(i, transformer)

pred_dict

with open('output.csv', 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['Id', 'Category'])
    for key, value in pred_dict.items():
        writer.writerow([key, value])
