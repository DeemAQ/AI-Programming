import argparse
from torchvision import datasets, transforms, models
from torch import nn, optim
import os
import torch


def get_args():
    parser = argparse.ArgumentParser(description="102 Flowers Classification--Training")
    parser.add_argument("data_dir", type=str, help="images data directory -- String -- required*")
    parser.add_argument("--save_dir", default="checkpoints/", type=str, help="directory path to save checkpoint -- String -- default=checkpoints/")
    parser.add_argument("--arch", default="vgg16", type=str, help="model architecture."+
                        " Options: vgg13, vgg16, vgg19 -- String -- defaul=vgg16")
    parser.add_argument("--learning_rate", default=0.01, type=float, help="learning rate -- Float -- defaul=0.01")
    parser.add_argument("--hidden_units", default=512, type=int, help="number of units in the hidden layer -- Integer -- defaul=512")
    parser.add_argument("--epochs", default=20, type=int, help="number of epochs during training -- Integer -- default=20")
    parser.add_argument("--gpu", default=False, action="store_true", help="Use GPU while training -- Boolean -- default=False")
    return parser.parse_args()


# global values:
means = [0.485, 0.456, 0.406]
standard_devs = [0.229, 0.224, 0.225]


# this function transforms and loads the dataset, returns dataloaders
def get_loaders(data_dir):
    train_dir = data_dir + "/train"
    valid_dir = data_dir + "/valid"

    train_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.RandomResizedCrop((224, 224)),
                                      transforms.RandomVerticalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(means, 
                                                           standard_devs)
                                      ])
    valid_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop((224, 224)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(means, 
                                                            standard_devs)
                                        ])

    # Loading the datasets with ImageFolder
    train_set = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_set = datasets.ImageFolder(valid_dir, transform=valid_transforms)

    # Defining the dataloaders
    train_loader = torch.utils.data.DataLoader(train_set, 
                                            batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_set,
                                            batch_size=32, shuffle=True)
    
    return train_loader, valid_loader


# builds the model and creates custom classifier, returns model
def network(arch, h_dim, out_dim, drop_prob):
    if arch == "vgg13":
        print("using vgg13, see the full architecture:\n")
        model = models.vgg13(weights=models.VGG13_Weights.DEFAULT)
    elif arch == "vgg19":
        print("using vgg19, see the full architecture:\n")
        model == models.vgg19(weights=models.VGG19_Weights.DEFAULT)
    else:
        print("using default vgg16, see the full architecture:\n")
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    
    for param in model.parameters():
        param.requires_grad = False

    # Build the classifier
    # output from prev layer was 25088
    input_dim = model.classifier[0].in_features
    model.classifier = nn.Sequential(nn.Linear(input_dim, h_dim),
                                    nn.ReLU(),
                                    nn.Dropout(drop_prob),
                                    nn.Linear(h_dim, out_dim),
                                    nn.LogSoftmax(dim=1)
                                    )

    print(model, "\n")

    return model



# this function trains the model
def train(model, train_loader, valid_loader, criterion, optimizer, epochs, use_gpu):
    if use_gpu and torch.cuda.is_available():
        model.cuda()

    print_every = 5
    steps = 0
    train_loss = 0
    
    for epoch in range(epochs):
        for images, labels in train_loader:
            steps += 1
            
            if use_gpu and torch.cuda.is_available():
                images, labels = images.to("cuda"), labels.to("cuda")
            
            images.requires_grad = True

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                
                valid_loss, accuracy = validate(model, valid_loader, criterion, use_gpu)

                print(f"Epoch {epoch+1}/{epochs}.. "
                      f"Training Loss: {train_loss/print_every:.3f}.. "
                      f"Validation Loss: {valid_loss/len(valid_loader):.3f}.. "
                      f"Accuracy: {accuracy/len(valid_loader):.3f}")

                train_loss = 0
                model.train()


# this function validates the model during training, returns validation loss and accuracy
def validate(model, data_loader, criterion, use_gpu):
    if use_gpu and torch.cuda.is_available():
        model.cuda()

    valid_loss = 0
    accuracy = 0
    with torch.no_grad():
        for images, labels in data_loader:
            if use_gpu and torch.cuda.is_available():
                images, labels = images.to("cuda"), labels.to("cuda")

            images.requires_grad = True

            logits = model(images)
            batch_loss = criterion(logits, labels)
            valid_loss += batch_loss.item()

            probs = torch.exp(logits)
            _, top_class = probs.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    return valid_loss, accuracy


def save_model(model, save_dir, arch, optimizer, epochs, lr, h_units):
    checkpoint = {
        "structure": arch,
        "learning_rate": lr,
        "hidden_units": h_units,
        "classifier": model.classifier,
        "epochs": epochs,
        "optimizer": optimizer.state_dict(),
        "state_dict": model.state_dict()
    }

    if save_dir[-1] != "/":
        save_dir += "/"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(checkpoint, f"{save_dir}{arch}_checkpoint.pth")


def main():
    args = get_args()

    train_loader, valid_loader = get_loaders(args.data_dir)

    model = network(args.arch.lower(), args.hidden_units, 102, 0.2)
    print("-"*50, "\n")
    criterion = nn.NLLLoss()
    optimizer = optim.SGD(model.classifier.parameters(), lr=args.learning_rate)
    
    train(model, train_loader, valid_loader, criterion, optimizer, args.epochs, args.gpu)

    # save the model after training
    save_model(
        model,
        args.save_dir,
        args.arch,
        optimizer,
        args.epochs,
        args.learning_rate, 
        args.hidden_units
    )


if __name__ == "__main__":
    main()