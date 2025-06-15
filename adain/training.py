import torch
import torch.nn as nn
import models
import losses
import torch.optim as optim
import numpy as np
import os

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

LR = 1e-4 # 1e-4 in https://github.com/naoto0804/pytorch-AdaIN/blob/master/train.py
PROCESSED_DATA_CONTENT_TRAIN_PATH = "adain/processed_data/content/train"
PROCESSED_DATA_CONTENT_VAL_PATH = "adain/processed_data/content/val"
PROCESSED_DATA_STYLE_TRAIN_PATH = "adain/processed_data/style/train"
PROCESSED_DATA_STYLE_VAL_PATH = "adain/processed_data/style/val"
BATCH_SIZE = 8 # Batch size to use during training
DISK_BATCH_SIZE = 10000 # How many images per .npy file on disk
NUM_EPOCHS = 50
MODEL_WEIGHTS_DEST_PATH = "adain/models"
STYLE_LOSS_WEIGHT = 5 # 10 in https://github.com/naoto0804/pytorch-AdaIN

# Didn't have enough RAM to preprocess the entire dataset at once,
# so the preprocessed images are divided into many .npy files,
# each containing 10 000 images
# TODO: Maybe rework so it follows standard Python iterator interface
class BatchedDataLoader():

    def __init__(self, processed_dataset_path: str, batch_size=BATCH_SIZE, disk_batch_size=DISK_BATCH_SIZE):
        self.batch_size = batch_size
        self.disk_batch_size = disk_batch_size
        self.paths = os.listdir(processed_dataset_path) # Assuming they're all in one directory, and nothing else is in there
                                                              # TODO: Maybe make more flexible with os.walk
        self.paths = iter([path for path in self.paths if path.endswith(".npy")])

        self.images_loaded_in_this_disk_batch = 0
        self.current_disk_batch = None
        self.dirpath = processed_dataset_path + "/"

    def get_next_batch(self) -> torch.Tensor:
        if (self.images_loaded_in_this_disk_batch >= self.disk_batch_size):
            self.images_loaded_in_this_disk_batch = 0
        if self.images_loaded_in_this_disk_batch == 0:
            try:
                self.current_disk_batch = torch.Tensor(np.load(self.dirpath + next(self.paths))).to(DEVICE)
            except: # No more batches in the dataset
                return None
        left_idx = self.images_loaded_in_this_disk_batch
        right_idx = min(left_idx + self.batch_size, self.current_disk_batch.shape[0]) # min(left_idx + batch_size, number_of_elements)
        self.images_loaded_in_this_disk_batch += (right_idx - left_idx)
        return self.current_disk_batch[left_idx : right_idx]

class AdaINTrainer():
    
    def __init__(self, encoder_model: nn.Module, decoder_model: nn.Module, content_dirpath: str, style_dirpath: str,
                 content_val_dirpath: str, style_val_dirpath: str, model_weights_dest_path: str, lr=LR,
                 style_loss_weight=STYLE_LOSS_WEIGHT):
        self.content_dirpath = content_dirpath
        self.style_dirpath = style_dirpath
        self.content_val_dirpath = content_val_dirpath
        self.style_val_dirpath = style_val_dirpath
        self.model_weights_dest_path = model_weights_dest_path
        self.encoder_model = encoder_model.to(DEVICE)
        self.decoder_model = decoder_model.to(DEVICE)
        self.optimizer = optim.Adam(self.decoder_model.parameters(), lr)
        self.style_loss_weight = style_loss_weight
        self.adain = models.AdaIN().to(DEVICE)
        # Freeze the parameters of encoder since it's pre-trained and fixed
        for param in self.encoder_model.parameters():
            param.requires_grad = False

        self.content_loss = losses.ContentLoss(self.encoder_model)
        self.style_loss = losses.StyleLoss(self.encoder_model)

        self.epoch_train_losses = []

    def train_epoch(self):
        content_loader = BatchedDataLoader(self.content_dirpath)
        style_loader = BatchedDataLoader(self.style_dirpath)

        epoch_loss = 0

        while True:
            content_batch = content_loader.get_next_batch()
            style_batch = style_loader.get_next_batch()
            if content_batch is None or style_batch is None:
                break
            content_batch = content_batch.to(DEVICE)
            style_batch = style_batch.to(DEVICE)
            
            content_encoded = self.encoder_model(content_batch)[0]
            style_encoded = self.encoder_model(style_batch)[0]
            adain_output = self.adain(content_encoded, style_encoded)
            generated_batch = self.decoder_model(adain_output)

            loss = self.content_loss(generated_batch, adain_output) + self.style_loss_weight * self.style_loss(generated_batch, style_batch)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss.item()
            
        return epoch_loss
    
    def evaluate_model(self):
        content_loader = BatchedDataLoader(self.content_val_dirpath)
        style_loader = BatchedDataLoader(self.style_val_dirpath)

        total_loss = 0

        with torch.no_grad():
            while True:
                content_batch = content_loader.get_next_batch()
                style_batch = style_loader.get_next_batch()
                if content_batch is None or style_batch is None:
                    break
                content_batch = content_batch.to(DEVICE)
                style_batch = style_batch.to(DEVICE)

                content_encoded = self.encoder_model(content_batch)[0]
                style_encoded = self.encoder_model(style_batch)[0]
                adain_output = self.adain(content_encoded, style_encoded)
                generated_batch = self.decoder_model(adain_output)

                loss = self.content_loss(generated_batch, adain_output) + self.style_loss(generated_batch, style_batch)
                total_loss += loss.item()

        return total_loss
    
    def train_n_epochs(self, n=NUM_EPOCHS):
        for i in range(n):
            epoch_loss = self.train_epoch()
            self.epoch_train_losses.append(epoch_loss)
            print("Epoch {} training loss: {}".format(i+1, epoch_loss))
            val_loss = self.evaluate_model()
            print("Epoch {} validation loss: {}".format(i+1, val_loss))
            torch.save(self.decoder_model.state_dict(), self.model_weights_dest_path + "/epoch_{}.pt".format(i+1))


if __name__ == '__main__':
    trainer = AdaINTrainer(models.get_vgg19_extractor(), models.DecoderForVGG19Encoder(), PROCESSED_DATA_CONTENT_TRAIN_PATH,
                           PROCESSED_DATA_STYLE_TRAIN_PATH, PROCESSED_DATA_CONTENT_VAL_PATH, PROCESSED_DATA_STYLE_VAL_PATH,
                           MODEL_WEIGHTS_DEST_PATH)
    trainer.train_n_epochs()