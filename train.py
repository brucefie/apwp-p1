#IMPORTS
import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', help='Data directory',default="./flowers/")
parser.add_argument ('--save_dir', help = 'Save directory', default=".")
parser.add_argument ('--arch', help = 'Vgg19', default='vgg19')
parser.add_argument ('--lrn', help = 'Learning rate', default=0.0008)
parser.add_argument ('--hidden_units', help = 'Hidden units',default=512)
parser.add_argument ('--epochs', help = 'Number of epochs', type = int, default=8)
parser.add_argument ('--gpu', help = "Option to use GPU", type = str,default=False)
args = parser.parse_args ()

device = ("cuda" if args.gpu else "cpu")

#LOAD THE DATA
data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
data_transforms = { 'train': transforms.Compose([transforms.RandomResizedCrop(224),
                   transforms.ToTensor(),
                   transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),
'valid': transforms.Compose([transforms.RandomResizedCrop(224),
                   transforms.ToTensor(),
                   transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])]),
'test': transforms.Compose([transforms.RandomResizedCrop(224),
                   transforms.ToTensor(),
                   transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])}

image_datasets = {
    'train': datasets.ImageFolder(train_dir, transform=data_transforms['train']),
    'test': datasets.ImageFolder(test_dir, transform=data_transforms['test']),
    'valid': datasets.ImageFolder(valid_dir, transform=data_transforms['valid'])
}

dataloaders = {
    'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=64, shuffle=True),
    'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=64),
    'valid': torch.utils.data.DataLoader(image_datasets['valid'], batch_size=64, shuffle=True)
}


#LABEL MAPPING
import json

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
    
#Model
def load_model(hidden_units, learning_rate, device):
    model=models.vgg19(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    
    classifier = nn.Sequential(OrderedDict([
                                ('fc1',nn.Linear(25088, 1024)),
                                ('relu',nn.ReLU()),
                                ('dropout',nn.Dropout(p=0.5)),
                                ('fc2',nn.Linear(1024,hidden_units)),
                                ('relu2',nn.ReLU()),
                                ('dropout2',nn.Dropout(p=0.5)),
                                ('fc3', nn.Linear(hidden_units, 102)),
                                ('output', nn.LogSoftmax(dim=1))
                                ]))
  
    model.classifier = classifier
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)

    if device == "cuda" and torch.cuda.is_available(): 
        model.to("cuda")
    else: 
        model.to("cpu")
    
    return model, criterion, optimizer
            
model, criterion, optimizer = load_model(hidden_units=args.hidden_units, learning_rate=args.lrn, device=device)

#TRAIN
epochs= args.epochs
print("epochs: {}".format(epochs))
steps = 0
print_every = 10
                           
for epoch in range(epochs):
    running_loss=0
    for inputs, labels in dataloaders['train']:
        
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs) #error here
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
                           
        running_loss += loss.item()
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            
            with torch.no_grad():
                model.to(device)
                for inputs, labels in dataloaders['valid']:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels).item()
                    
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                
                training_loss = batch_loss/len(dataloaders['train'])
                batch_loss= batch_loss/len(dataloaders['valid'])
                accuracy = accuracy/len(dataloaders['valid'])*100
            
                print("Epoch: {}/{}.. ".format(epoch+1, epochs),
                      "Training Loss: {:.3f}.. ".format(training_loss),
                      "Batch Loss: {:.3f}.. ".format(batch_loss),
                      "Valid Accuracy: {:.3f}%".format(accuracy))
            
           
                running_loss = 0
                model.train() 

#TEST
correct=0
total=0
model.to(device)
with torch.no_grad():
    for inputs, labels in dataloaders['test']:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    print('Accuracy of the network on test images: %d %%' % (100 * correct / total))
    
#CHECKPOINT
model.to('cpu')
model.class_to_idx = image_datasets['train'].class_to_idx
checkpoint={'classifier': model.classifier,
            'epochs': epochs,
            'arch': 'vgg19',
            'class_to_idx': model.class_to_idx,
            'state_dict': model.state_dict (),
            'optimizer' : optimizer.state_dict() 
           }
torch.save(checkpoint,args.save_dir + 'checkpoint.pth')
