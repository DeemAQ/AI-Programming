import argparse
import torch
from torch import nn
import json
from torchvision import transforms, models
from PIL import Image


def get_args():
    parser = argparse.ArgumentParser(description="102 Flowers Classification--Prediction")
    parser.add_argument("input_image", type=str, help="input image path -- String -- required*")
    parser.add_argument("checkpoint", type=str, help="pre-trained model path -- String -- required*")
    parser.add_argument("--top_k", default=5, type=int, help="top_k results -- Integer -- dafault=5")
    parser.add_argument("--category_names", default="", type=str, help="default category names file path -- String -- default=empty")
    parser.add_argument("--gpu", default=False, action="store_true", help="Use GPU while predicting -- Boolean -- default=False")
    return parser.parse_args()


def process_image(image):
    """ Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    """
    means = [0.485, 0.456, 0.406]
    standard_devs = [0.229, 0.224, 0.225]

    # Process a PIL image for use in a PyTorch model
    img_pil = Image.open(image)
    img_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(means, standard_devs)
    ])
    
    image = img_transforms(img_pil)
    
    return image


def predict(image, model, topk, category_names, use_gpu):
    """ Predict the class (or classes) of an image using a trained deep learning model.
    """
    model.eval()

    image = torch.from_numpy(image.numpy()).float()
    image = torch.unsqueeze(image, dim=0)

    if use_gpu and torch.cuda.is_available():
        image, model = image.to("cuda"), model.to("cuda")

    with torch.no_grad():
        output = model(image)
        
    probs, classes = torch.exp(output).data.topk(topk)

    probs = probs.cpu().data.numpy().tolist()[0]
    classes = classes.cpu().data.numpy().tolist()[0]

    if category_names:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
        
        classes = [cat_to_name[str(c)] for c in classes]

    
    return probs, classes


# this function loads the model from pth file, returns model
def load_model(model_path):
    checkpoint = torch.load(model_path)
    
    model = eval("models."+checkpoint["structure"]+"(weights=models."+checkpoint["structure"].upper()+"_Weights.DEFAULT)")
    
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = checkpoint["classifier"]
    model.load_state_dict(checkpoint["state_dict"])

    print("model loaded successfully, see the architecture:\n")
    print(model, "\n\n")

    return model


def main():
    args = get_args()
    processed_img = process_image(args.input_image)
    model = load_model(args.checkpoint)
    probs, classes = predict(processed_img, model, args.top_k, args.category_names, args.gpu)

    print(f"Top {args.top_k} predictions: {list(zip(classes, probs))}")


if __name__ == "__main__":
    main()