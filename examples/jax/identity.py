# %% [markdown]
# # JAX Identity Example
# This example shows how to use volara with a JAX/Flax model.
# We create a trivial identity model, run blockwise prediction on synthetic data,
# and verify the output matches the input.

# %%
import cloudpickle
import shutil
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import zarr
from flax import linen as nn
from funlib.geometry import Coordinate

from volara.datasets import Raw
from volara_ml.blockwise import Predict
from volara_ml.models import JaxModel

# %% [markdown]
# ## 1. Create synthetic input data

# %%
work_dir = Path("_jax_identity_workdir")
work_dir.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)
data = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)

store = zarr.open(str(work_dir / "data.zarr"), mode="w")
raw_ds = store.create_array("raw", shape=data.shape, dtype=data.dtype)
raw_ds[:] = data
raw_ds.attrs["voxel_size"] = (1, 1)
raw_ds.attrs["axis_names"] = ["y", "x"]
raw_ds.attrs["unit"] = ["px", "px"]

# %% [markdown]
# ## 2. Define and save a trivial Flax identity model

# %%
class Identity(nn.Module):
    @nn.compact
    def __call__(self, x):
        # Flax Conv expects channels-last (N, H, W, C), but input is (N, C, H, W).
        x = jnp.transpose(x, (0, 2, 3, 1))
        x = nn.Conv(
            features=1,
            kernel_size=(1, 1),
            use_bias=False,
            kernel_init=nn.initializers.ones,
        )(x)
        x = jnp.transpose(x, (0, 3, 1, 2))
        return x


model = Identity()

model_path = work_dir / "model.pkl"
with open(model_path, "wb") as f:
    cloudpickle.dump(model, f)

rng = jax.random.PRNGKey(0)
dummy = jnp.ones((1, 1, 64, 64))
params = model.init(rng, dummy)

ckpt_path = work_dir / "ckpt"
checkpointer = ocp.StandardCheckpointer()
checkpointer.save(str(ckpt_path.resolve()), params)
checkpointer.close()

# %% [markdown]
# ## 3. Configure JaxModel and Predict

# %%
jax_model = JaxModel(
    model_path=model_path,
    params_path=ckpt_path.resolve(),
    in_channels=1,
    out_channels=1,
    min_input_shape=Coordinate(64, 64),
    min_output_shape=Coordinate(64, 64),
    min_step_shape=Coordinate(1, 1),
    out_range=(0, 1),
    pred_size_growth=Coordinate(0, 0),
)

raw_in = Raw(
    store=work_dir / "data.zarr/raw",
    scale_shift=(1 / 255, 0),
)
raw_out = Raw(store=work_dir / "data.zarr/output")

predict_task = Predict(
    checkpoint=jax_model,
    in_data=raw_in,
    out_data=[raw_out],
)

# %% [markdown]
# ## 4. Run blockwise prediction

# %%
predict_task.drop()
predict_task.run_blockwise(multiprocessing=False)

# %% [markdown]
# ## 5. Verify output matches input

# %%
input_data = zarr.open(str(work_dir / "data.zarr/raw"), mode="r")[:]
output_data = zarr.open(str(work_dir / "data.zarr/output"), mode="r")[:]

# The identity model receives float data in [0, 1] (via scale_shift) and outputs
# it unchanged. The output is then quantized back to uint8. The output has a
# channel dimension prepended (1, H, W), so we squeeze it for comparison.
output_data = output_data.squeeze(axis=0)
assert output_data.shape == input_data.shape, (
    f"Shape mismatch: input {input_data.shape} vs output {output_data.shape}"
)
assert np.allclose(output_data, input_data, atol=1), (
    f"Output does not match input. Max diff: {np.abs(output_data.astype(int) - input_data.astype(int)).max()}"
)

print("Success! Output matches input (identity model verified).")

# Clean up
shutil.rmtree(work_dir)
shutil.rmtree("volara_logs", ignore_errors=True)
