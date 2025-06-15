import torch
import torch.nn as nn
import models

class ContentLoss():

    def __init__(self, encoder: nn.Module):
        self.adain = models.AdaIN()
        self.encoder = encoder
        self.loss = nn.MSELoss()
        pass

    def __call__(self, generated_image: torch.Tensor, adain_output: torch.Tensor, **kwds):
        return self.loss(self.encoder(generated_image)[0], adain_output)
    
class StyleLoss():

    def __init__(self, encoder: nn.Module):
        self.encoder = encoder
        self.loss = nn.MSELoss()

    def __call__(self, generated_image: torch.Tensor, style_image: torch.Tensor):
        _, *generated_feature_maps = self.encoder(generated_image)
        _, *style_feature_maps = self.encoder(style_image)
        assert len(generated_feature_maps) == len(style_feature_maps) # Just in case
        loss = 0
        for i in range(len(generated_feature_maps)): # Weights are all 1, as in the paper
            fm_gen = generated_feature_maps[i]
            fm_style = style_feature_maps[i]
            mean_gen = fm_gen.reshape(fm_gen.shape[0], fm_gen.shape[1], -1).mean(2)
            std_gen = fm_gen.reshape(fm_gen.shape[0], fm_gen.shape[1], -1).std(2)
            mean_style = fm_style.reshape(fm_style.shape[0], fm_style.shape[1], -1).mean(2)
            std_style = fm_style.reshape(fm_style.shape[0], fm_style.shape[1], -1).std(2)
            loss += (self.loss(mean_gen, mean_style) + self.loss(std_gen,std_style))
        return loss