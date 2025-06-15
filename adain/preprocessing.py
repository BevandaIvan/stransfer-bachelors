from abc import ABC, abstractmethod
import os
from PIL import Image
import numpy as np
import random
import string
from tqdm import tqdm
import torchvision.models as models

# Preprocessing constants and parameters
DATASET_CONTENT_TRAIN_PATH = "dataset/mscoco2014_train"
DATASET_CONTENT_VAL_PATH = "dataset/mscoco2014_val"
DATASET_CONTENT_TRAIN_PROCESSED_DEST = "adain/processed_data/content/train"
DATASET_CONTENT_VAL_PROCESSED_DEST = "adain/processed_data/content/val"
DATASET_STYLE_TRAIN_PATH = "dataset/wikiart_train"
DATASET_STYLE_VAL_PATH = "dataset/wikiart_val"
DATASET_STYLE_TRAIN_PROCESSED_DEST = "adain/processed_data/style/train"
DATASET_STYLE_VAL_PROCESSED_DEST = "adain/processed_data/style/val"
DATASET_FRACTION_USED = 1.0
BATCH_SIZE = 10000

class AbstractHandler(ABC):

    @abstractmethod
    def process(data_batch):
        # Do something to data_batch
        # Call next_handler.process(modified_data_batch)
        pass

class DefaultHandler(AbstractHandler):

    def __init__(self):
        pass

    def process(self, data_batch):
        return data_batch
    
class ImagePathBatchLoaderHandler(AbstractHandler):

    def __init__(self, dataset_path: str, batch_size: int, randomize: bool, next_handler: AbstractHandler):
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.left_index = 0
        self.right_index = batch_size
        self.next_handler = next_handler
        self.paths = self.get_image_path_list()
        if (randomize):
            random.shuffle(self.paths)

    def get_image_path_list(self):
        paths = []
        for dirpath, _, filenames in os.walk(self.dataset_path):
            for filename in filenames:
                if filename.endswith(".jpg"):
                    paths.append(os.path.join(dirpath, filename))
        return paths

    def process(self, data_batch=None):
        if self.left_index >= self.right_index:
            return [] # No more chunks
        ret = self.paths[self.left_index : self.right_index]
        self.left_index = self.left_index + self.batch_size
        self.right_index = min(self.right_index + self.batch_size, len(self.paths))
        return self.next_handler.process(ret)
    
class PathToImageHandler(AbstractHandler):

    def __init__(self, next_handler: AbstractHandler):
        self.next_handler = next_handler

    def process(self, data_batch):
        return self.next_handler.process([Image.open(path).convert('RGB') for path in data_batch])
    
class ImageToVGGReadyTensorHandler(AbstractHandler):

    def __init__(self, next_handler: AbstractHandler):
        self.transform = models.VGG19_Weights.IMAGENET1K_V1.transforms()
        self.next_handler = next_handler

    def process(self, data_batch):
        return self.next_handler.process([self.transform(img) for img in data_batch])
    
class TensorToNumpyHandler(AbstractHandler):

    def __init__(self, next_handler: AbstractHandler):
        self.next_handler = next_handler

    def process(self, data_batch):
        return self.next_handler.process([x.detach().numpy() for x in data_batch])
    
class NumpySaveImageBatchHandler(AbstractHandler):

    def __init__(self, save_dir: str, next_handler: AbstractHandler):
        self.save_dir = save_dir
        self.next_handler = next_handler

    def process(self, data_batch):
        characters = string.ascii_letters + string.digits
        random_string = ''.join(random.choices(characters, k=15))
        np.save(self.save_dir + "/" + random_string + ".npy", np.array(data_batch))
        self.next_handler.process(data_batch)
    
def process_dataset(fraction_used: float):

    pipeline = ImagePathBatchLoaderHandler(DATASET_CONTENT_TRAIN_PATH, BATCH_SIZE, True,
                                           PathToImageHandler(
                                           ImageToVGGReadyTensorHandler(
                                           TensorToNumpyHandler(
                                           NumpySaveImageBatchHandler(DATASET_CONTENT_TRAIN_PROCESSED_DEST,
                                           DefaultHandler()
                                           )))))
    
    for _ in tqdm(range(int(fraction_used * len(pipeline.paths) / BATCH_SIZE))):
        pipeline.process()

    pipeline = ImagePathBatchLoaderHandler(DATASET_CONTENT_VAL_PATH, BATCH_SIZE, True,
                                           PathToImageHandler(
                                           ImageToVGGReadyTensorHandler(
                                           TensorToNumpyHandler(
                                           NumpySaveImageBatchHandler(DATASET_CONTENT_VAL_PROCESSED_DEST,
                                           DefaultHandler()
                                           )))))
    
    for _ in tqdm(range(int(fraction_used * len(pipeline.paths) / BATCH_SIZE))):
        pipeline.process()

    pipeline = ImagePathBatchLoaderHandler(DATASET_STYLE_TRAIN_PATH, BATCH_SIZE, True,
                                           PathToImageHandler(
                                           ImageToVGGReadyTensorHandler(
                                           TensorToNumpyHandler(
                                           NumpySaveImageBatchHandler(DATASET_STYLE_TRAIN_PROCESSED_DEST,
                                           DefaultHandler()
                                           )))))
    
    for _ in tqdm(range(int(fraction_used * len(pipeline.paths) / BATCH_SIZE))):
        pipeline.process()

    pipeline = ImagePathBatchLoaderHandler(DATASET_STYLE_VAL_PATH, BATCH_SIZE, True,
                                           PathToImageHandler(
                                           ImageToVGGReadyTensorHandler(
                                           TensorToNumpyHandler(
                                           NumpySaveImageBatchHandler(DATASET_STYLE_VAL_PROCESSED_DEST,
                                           DefaultHandler()
                                           )))))
    
    for _ in tqdm(range(int(fraction_used * len(pipeline.paths) / BATCH_SIZE))):
        pipeline.process()

if __name__ == '__main__':
    process_dataset(DATASET_FRACTION_USED)