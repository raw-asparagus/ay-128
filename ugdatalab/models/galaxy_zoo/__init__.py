"""Galaxy Zoo 2 classification labels, image preprocessing, and PyTorch dataset."""

from ugdatalab.models.galaxy_zoo.galaxy_zoo import GalaxyZooData
from ugdatalab.models.galaxy_zoo.images import GalaxyImages, GalaxyZooImages
from ugdatalab.models.galaxy_zoo.images_pipeline import CropFraction, Resize
from ugdatalab.models.galaxy_zoo.dataset import GalaxyZooDataset
