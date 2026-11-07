"""Read the interop eye photos and reframe them like the training inputs (see core.py)."""

from .core import (
    DATASETS,
    DEFAULT_ROOT,
    DEFAULT_SIZE,
    TARGET_CENTER,
    TARGET_IRIS_W,
    Prepared,
    Record,
    list_images,
    load_overrides,
    make_dataset,
    prepare,
    preview_sheet,
    to_source,
    to_tensor,
    write_template,
)
