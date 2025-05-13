# %% [markdown]
# # Cremi example
# This example shows how to use volara to predict LSDs and affinities on the cremi dataset, and then run mutex watershed on the predicted affinities.

# %%
from pathlib import Path

import wget
from funlib.geometry import Coordinate

Path("_static/cremi").mkdir(parents=True, exist_ok=True)

# Download some cremi data
# immediately convert it to zarr for convenience
if not Path("sample_A+_20160601.zarr").exists():
    wget.download(
        "https://cremi.org/static/data/sample_A+_20160601.hdf", "sample_A+_20160601.hdf"
    )
if not Path("sample_A+_20160601.zarr/raw").exists():
    import h5py
    import zarr

    raw_ds = zarr.open("sample_A+_20160601.zarr", "w").create_dataset(
        "raw", data=h5py.File("sample_A+_20160601.hdf", "r")["volumes/raw"][:]
    )
    raw_ds.attrs["voxel_size"] = (40, 4, 4)
    raw_ds.attrs["axis_names"] = ["z", "y", "x"]
    raw_ds.attrs["unit"] = ["nm", "nm", "nm"]

# %% [markdown]
# Now we can predict the LSDs and affinities for this dataset. We have provided a very simple
# pretrained model for this dataset. We went for speed and efficiency over accuracy for this
# model so that it can run in a github action. You can train a significantly better model
# with access to a GPU and more Memmory.

# %%
# Here are some important details about the model:

# The number of output channels of our model. 10 lsds, 7 affinities
out_channels = [10, 7]

# The input shape of our model (not including channels)
min_input_shape = Coordinate(36, 252, 252)

# The output shape of our model (not including channels)
min_output_shape = Coordinate(32, 160, 160)

# The minimum increment for adjusting the input shape
min_step_shape = Coordinate(1, 1, 1)

# The range of predicted values. We have a sigmoid activation on our model
out_range = (0, 1)

# How much to grow the input shape for prediction. This is usually adjusted to maximize GPU memory,
# but depends on how you saved your model. The model we provided does not support different
# input shapes.
pred_size_growth = Coordinate(0, 0, 0)


# %%
from volara.datasets import Affs, Raw
from volara_torch.blockwise import Predict
from volara_torch.models import TorchModel

# %% [markdown]

# First we define the datasets that we are using along with some basic information about them

# %%
# our raw data is stored in uint8, but our model expects floats in range (0, 1) so we scale it
raw_dataset = Raw(store="sample_A+_20160601.zarr/raw", scale_shift=(1 / 255, 0))
# The affinities neighborhood depends on the model that was trained. Here we learned long range xy affinities
affs_dataset = Affs(
    store="sample_A+_20160601.zarr/affs",
    neighborhood=[
        Coordinate(1, 0, 0),
        Coordinate(0, 1, 0),
        Coordinate(0, 0, 1),
        Coordinate(0, 6, 0),
        Coordinate(0, 0, 6),
        Coordinate(0, 18, 0),
        Coordinate(0, 0, 18),
    ],
)
# We are just storing the lsds in a simple zarr dataset using the same format as the raw data
lsds_dataset = Raw(store="sample_A+_20160601.zarr/lsds")

# %% [markdown]
# Now we can define our model with the parameters we defined above. We will use the
# `TorchModel` class to load the model from a checkpoint and pass it to the `Predict` class.

# %%
torch_model = TorchModel(
    save_path="checkpoint_data/model.pt",
    checkpoint_file="checkpoint_data/model_checkpoint_15000",
    in_channels=1,
    out_channels=out_channels,
    min_input_shape=min_input_shape,
    min_output_shape=min_output_shape,
    min_step_shape=min_step_shape,
    out_range=out_range,
    pred_size_growth=pred_size_growth,
)
predict_cremi = Predict(
    checkpoint=torch_model,
    in_data=raw_dataset,
    out_data=[lsds_dataset, affs_dataset],
)

predict_cremi.run_blockwise(multiprocessing=False)

# %% [markdown]
# Let's visualize the results

# %%
import matplotlib.pyplot as plt
import matplotlib.animation as animation


fig, axes = plt.subplots(1, 3, figsize=(14, 8))

ims = []
for i, (raw_slice, affs_slice, lsd_slice) in enumerate(
    zip(
        raw_dataset.array("r")[:],
        affs_dataset.array("r")[:].transpose([1, 0, 2, 3]),
        lsds_dataset.array("r")[:].transpose([1, 0, 2, 3]),
    )
):
    # Show the raw data
    if i == 0:
        im_raw = axes[0].imshow(raw_slice, cmap="gray")
        axes[0].set_title("Raw")
        im_affs_long = axes[1].imshow(
            affs_slice[[0, 5, 6]].transpose([1, 2, 0]),
            vmin=0,
            vmax=255,
            interpolation="none",
        )
        axes[1].set_title("Affs (0, 5, 6)")
        im_lsd = axes[2].imshow(
            lsd_slice[:3].transpose([1, 2, 0]),
            vmin=0,
            vmax=255,
            interpolation="none",
        )
        axes[2].set_title("LSDs (0, 1, 2)")
    else:
        im_raw = axes[0].imshow(raw_slice, cmap="gray", animated=True)
        axes[0].set_title("Raw")
        im_affs_long = axes[1].imshow(
            affs_slice[[0, 5, 6]].transpose([1, 2, 0]),
            vmin=0,
            vmax=255,
            interpolation="none",
            animated=True,
        )
        axes[1].set_title("Affs (0, 5, 6)")
        im_lsd = axes[2].imshow(
            lsd_slice[:3].transpose([1, 2, 0]),
            vmin=0,
            vmax=255,
            interpolation="none",
            animated=True,
        )
        axes[2].set_title("LSDs (0, 1, 2)")
    ims.append([im_raw, im_affs_long, im_lsd])

ims = ims + ims[::-1]
ani = animation.ArtistAnimation(fig, ims, blit=True)
ani.save("_static/cremi/outputs.gif", writer="pillow", fps=10)
plt.close()

# %% [markdown]
# ![segmentation](_static/cremi/outputs.gif)
