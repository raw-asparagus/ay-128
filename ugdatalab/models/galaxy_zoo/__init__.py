from ugdatalab.models.galaxy_zoo.galaxy_zoo import (
    GalaxyZooData,
    GalaxyZooSplit,
)
from ugdatalab.models.galaxy_zoo.images import (
    GalaxyZooImages,
)

try:
    from ugdatalab.models.galaxy_zoo.images import GalaxyZooDataset
except ImportError:
    pass
