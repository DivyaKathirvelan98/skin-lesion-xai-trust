# Data

This folder is intentionally empty in version control (see `.gitignore`).

Download HAM10000 yourself from the official source and accept its non-commercial terms of
use, then arrange it as:

```
data/raw/images/                    # all HAM10000 .jpg images (parts 1 and 2 merged)
data/raw/HAM10000_metadata.csv      # official metadata (image_id, lesion_id, dx, ...)
data/raw/segmentations/             # ISIC ground-truth lesion segmentation masks, named <image_id>_segmentation.png
```

Official source: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

Segmentation masks (for the XAI faithfulness metrics) can be retrieved from the ISIC Archive
gallery/API for the matching `image_id`s: https://www.isic-archive.com/
