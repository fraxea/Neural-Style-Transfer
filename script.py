import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.optim import LBFGS
import torch.nn.functional as F
from torchvision.models import vgg19
from torchvision.models.vgg import VGG19_Weights

from PIL import Image
import copy
import matplotlib.pyplot as plt

NAME = "mamad"


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
torch.set_default_device(device)

vgg_model = vgg19(weights=VGG19_Weights.DEFAULT).to(device).eval()

vgg_model_features = vgg_model.features

def imshow(tensor, title=None):
    image = transforms.ToPILImage()(tensor.cpu().clone().squeeze(0))
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.axis("off")
    plt.show()


img_size = 512 if not device == "cpu" else 256

cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.to(device).eval()

#TODO: Implement some transformations for the images
transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor()
])

cnn_normalization_mean = torch.tensor(
    [0.485, 0.456, 0.406]
)  # Normalization mean for the VGG19 model
cnn_normalization_std = torch.tensor(
    [0.229, 0.224, 0.225]
)  # Normalization std for the VGG19 model

def gram_calculator(input):
    """
    Calculate the normalized Gram Matrix of a given input
    """
    N, C, H, W = input.size()
    input = input.view(N * C, H * W)
    gram = torch.mm(input, input.t())
    return gram.div(N * C * H * W)

# Normalize the images in order to feed it to the VGG19 model
class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, img):
        return (img - self.mean) / self.std

class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        self.target = gram_calculator(target_feature).detach()

    def forward(self, input):
        G = gram_calculator(input)
        self.loss = F.mse_loss(G, self.target)
        return input

content_layers_default = ["conv_4"]
style_layers_default = ["conv_1", "conv_2", "conv_3", "conv_4", "conv_5"]

def get_style_model_and_losses(cnn, normalization_mean, normalization_std, style_img, content_img, content_layers=content_layers_default, style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)
    normalization = Normalization(normalization_mean, normalization_std).to(device)
    content_losses = []
    style_losses = []
    model = nn.Sequential(normalization)
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f"conv_{i}"
        elif isinstance(layer, nn.ReLU):
            name = f"relu_{i}"
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f"pool_{i}"
        elif isinstance(layer, nn.BatchNorm2d):
            name = f"bn_{i}"
        else:
            raise RuntimeError("Unrecognized layer")
        model.add_module(name, layer)
        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)
        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break
    model = model[:(i + 1)]
    return model, style_losses, content_losses


NUM_ITER = 300
alpha = 1
beta = 1e6

totals = []
styles = []
contents = []

content_image = Image.open(f"./Images/Neural_Transfer/{NAME}.jpg")
style_image = Image.open("./Images/Neural_Transfer/Style.jpeg")

content_image = transform(content_image).to(device).unsqueeze(0)
style_image = transform(style_image).to(device).unsqueeze(0)

model, style_losses, content_losses = get_style_model_and_losses(
    vgg_model_features, cnn_normalization_mean, cnn_normalization_std, style_image, content_image
)

content_image = content_image.requires_grad_(True)

model.eval()
model.requires_grad_(False)

optimizer = LBFGS([content_image.requires_grad_(True)])

print("Start Training")
i = [0]
while i[0] <= NUM_ITER:
    def closure():
        with torch.no_grad():
            content_image.clamp_(0, 1)

        optimizer.zero_grad()
        model(content_image)
        style_score = 0
        content_score = 0

        for style_loss in style_losses:
            style_score += style_loss.loss
        for content_loss in content_losses:
            content_score += content_loss.loss

        content_score *= alpha
        style_score *= beta

        loss = style_score + content_score
        loss.backward()

        styles.append(style_score.item())
        contents.append(content_score.item())
        totals.append(loss.item())

        i[0] += 1
        if i[0] % 50 == 0:
            # imshow(content_image, f"Iteration: {i}: {loss.item():.4f}={style_score.item():.4f}+{content_score.item():.4f}")
            print(f"Iteration: {i}: {loss.item():.4f}={style_score.item():.4f}+{content_score.item():.4f}")

        return loss

    optimizer.step(closure)

with torch.no_grad():
    content_image.clamp_(0, 1)

# save content image to disk
content_image = content_image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
content_image = Image.fromarray((content_image * 255).astype("uint8"))
content_image.save(f"./Images/Neural_Transfer/{NAME}_result.jpg")
# plt.imshow(content_image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy())
# plt.axis("off")
# plt.title("Final Image")
# plt.show()