import torch
import torch.nn as nn

VGG19_STATE_DICT_PATH = "adain/models/vgg19-dcbb9e9d.pth"

# Although I could just import the whole model,
# the architecture is defined here for sanity-checks
# Modelled after: https://docs.pytorch.org/vision/main/models/vgg.html
# Layers whose activations are have comments next to them
class VGG19FeatureExtractor(nn.Module):

    def __init__(self):
        super(VGG19FeatureExtractor, self).__init__()

        self.features = nn.Sequential(
            nn.ReflectionPad2d(1), # To avoid border artifacts, as in the AdaIN paper
            nn.Conv2d(3, 64, kernel_size=3, padding=0),
            nn.ReLU(), #relu1_1 => 2
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 128, kernel_size=3, padding=0),
            nn.ReLU(), #relu2_1 => 9
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 256, kernel_size=3, padding=0),
            nn.ReLU(), #relu3_1 => 16
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 512, kernel_size=3, padding=0),
            nn.ReLU() #relu4_1 => 29
        )

    def forward(self, x):
        relu1_1 = None
        relu2_1 = None
        relu3_1 = None
        relu4_1 = None
        for module_number, module in self.features.named_modules():
            if isinstance(module, nn.Sequential):
                continue
            x = module(x)
            if module_number == '2':
                relu1_1 = x.clone() # Not sure if .clone() is really necessary, but I'm just making sure
            elif module_number == '9':
                relu2_1 = x.clone()
            elif module_number == '16':
                relu3_1 = x.clone()
            elif module_number == '29':
                relu4_1 = x.clone()

        return x, relu1_1, relu2_1, relu3_1, relu4_1
    
class DecoderForVGG19Encoder(nn.Module):

    def __init__(self):
        super(DecoderForVGG19Encoder, self).__init__()

        self.features = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, kernel_size=3, padding=0),
            nn.ReLU(),

            nn.UpsamplingNearest2d(scale_factor=2), # From the AdaIN paper
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 256, kernel_size=3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256, 128, kernel_size=3),
            nn.ReLU(),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 128, kernel_size=3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(128, 64, kernel_size=3),
            nn.ReLU(),

            nn.UpsamplingNearest2d(scale_factor=2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 3, kernel_size=3)
        )

    def forward(self, x):
        return self.features(x)
    
class AdaIN(nn.Module):

    def __init__(self):
        super(AdaIN, self).__init__()

    # x => content, y => style
    def forward(self, x, y):
        content_mean = x.reshape(x.shape[0], x.shape[1], -1).mean(dim=2).unsqueeze(2).unsqueeze(3)
        content_std = x.reshape(x.shape[0], x.shape[1], -1).std(dim=2).unsqueeze(2).unsqueeze(3)

        style_mean = y.reshape(y.shape[0], y.shape[1], -1).mean(dim=2).unsqueeze(2).unsqueeze(3)
        style_std = y.reshape(y.shape[0], y.shape[1], -1).std(dim=2).unsqueeze(2).unsqueeze(3)

        return style_std*(x - content_mean) / (content_std + 1e-5) + style_mean # 1e-5 is to avoid divison by 0

def get_vgg19_extractor():

    key_mapping = {
        "features.0.weight" : "features.1.weight",
        "features.0.bias" : "features.1.bias",
        "features.2.weight" : "features.4.weight",
        "features.2.bias" : "features.4.bias",
        "features.5.weight" : "features.8.weight",
        "features.5.bias" : "features.8.bias",
        "features.7.weight" : "features.11.weight",
        "features.7.bias" : "features.11.bias",
        "features.10.weight" : "features.15.weight",
        "features.10.bias" : "features.15.bias",
        "features.12.weight" : "features.18.weight",
        "features.12.bias" : "features.18.bias",
        "features.14.weight" : "features.21.weight",
        "features.14.bias" : "features.21.bias",
        "features.16.weight" : "features.24.weight",
        "features.16.bias" : "features.24.bias",
        "features.19.weight" : "features.28.weight",
        "features.19.bias" : "features.28.bias"
    }

    extractor = VGG19FeatureExtractor()
    state_dict = torch.load(VGG19_STATE_DICT_PATH)
    mapped_state_dict = {}
    for key in key_mapping.keys():
        mapped_state_dict[key_mapping[key]] = state_dict[key]
    extractor.load_state_dict(mapped_state_dict)
    return extractor