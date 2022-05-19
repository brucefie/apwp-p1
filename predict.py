#Imports
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
import json

parser = argparse.ArgumentParser()
parser.add_argument ('--image_dir', help = 'Path to image', type = str, default='flowers/train/10/image_07086.jpg')
parser.add_argument ('--load_dir', help = 'Path to checkpoint, use . first', type = str, default='.checkpoint.pth')
parser.add_argument('--top_k', action='store', help='Top k', default=int(5))
parser.add_argument ('--category_names', help = 'Mapping of categories names', type = str, default='cat_to_name.json')
parser.add_argument ('--gpu', help = "Option to use GPU. Optional", type = str, default=False)

args = parser.parse_args ()
device = ("cuda" if args.gpu else "cpu")
image_path = args.image_dir
checkpoint_path= args.load_dir

with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)

#LOAD CHECKPOINT
def load_model(checkpoint_path):
    model = models.vgg19(pretrained=True)
    checkpoint = torch.load(checkpoint_path)   
       
    for param in model.parameters(): 
            param.requires_grad = False
   
    model.classifier = checkpoint['classifier']
    model.epochs = checkpoint['epochs']
    model.class_to_idx = checkpoint['class_to_idx']
    model.optimizer = checkpoint ['optimizer']
   
    return model

model=load_model(checkpoint_path)

#PROCESS IMAGE
def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    
    pil_image = Image.open(image) 
    
    process_image = transforms.Compose([transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    np_image = process_image(pil_image)
    return np_image

#PREDICT
def predict(image_path, model, topk, device):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    
    model.to(device)
    image = process_image(image_path).to(device)
    np_image = image.unsqueeze_(0)
    model.eval()
    
    with torch.no_grad():
        logps = model.forward(np_image)
    
    ps = torch.exp(logps)
    top_k, top_classes_idx = ps.topk(topk, dim=1)
    top_k, top_classes_idx = np.array(top_k.to('cpu')[0]), np.array(top_classes_idx.to('cpu')[0])
    idx_to_class = {x: y for y, x in model.class_to_idx.items()}
    top_classes = []
    
    for index in top_classes_idx:
        top_classes.append(idx_to_class[index])
    
    return list(top_k), list(top_classes)

probabilities, classes=predict(args.image_dir,model,int(args.top_k),device=device)
class_names = [cat_to_name [item] for item in classes]

for l in range (int(args.top_k)):
     print("Number: {}/{}.. ".format(l+1, int(args.top_k)),
            "Flower name: {}.. ".format(class_names [l]),
            "Probability of : {:.3f}..% ".format(probabilities [l]*100),
            )
        