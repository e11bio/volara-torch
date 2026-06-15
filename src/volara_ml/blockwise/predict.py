import logging
from contextlib import contextmanager
from typing import Annotated, Callable, Literal

import daisy
import gunpowder as gp
import numpy as np
from funlib.geometry import Coordinate, Roi
from funlib.persistence import Array
from gunpowder import ArrayKey, Batch, BatchProvider
from gunpowder.nodes.generic_predict import GenericPredict
from pydantic import Field
from volara.blockwise import BlockwiseTask
from volara.datasets import LSD, Affs, Dataset, Raw

from ..models import JaxModel, Model, TorchModel

logger = logging.getLogger(__file__)


class ArraySource(BatchProvider):
    def __init__(self, key: ArrayKey, array: Array):
        self.key = key
        self.array = array

    def setup(self):
        spec = gp.ArraySpec(self.array.roi, self.array.voxel_size)
        self.provides(self.key, spec)

    def provide(self, request):
        outputs = Batch()
        data = self.array[request[self.key].roi]
        outputs[self.key] = gp.Array(
            data=data,
            spec=gp.ArraySpec(
                roi=request[self.key].roi,
                voxel_size=self.array.voxel_size,
            ),
        )
        return outputs


class ArrayWrite(gp.BatchFilter):
    def __init__(self, key: ArrayKey, array: Array, to_out_dtype: Callable):
        self.key = key
        self.array = array
        self.to_out_dtype = to_out_dtype

    def setup(self):
        self.updates(self.key, self.spec[self.key].copy())

    def process(self, batch, request):
        write_roi = request[self.key].roi.intersect(self.array.roi)
        data = batch[self.key].crop(write_roi).data
        self.array[write_roi] = self.to_out_dtype(data)


class CallablePredict(GenericPredict):
    """A framework-agnostic predict node that wraps a predict callable."""

    def __init__(
        self,
        predict_fn: Callable[[np.ndarray], list[np.ndarray]],
        input_key: ArrayKey,
        output_keys: dict[int, ArrayKey],
        spawn_subprocess: bool = False,
    ):
        self._predict_fn = predict_fn
        inputs = {0: input_key}
        super().__init__(inputs, output_keys, spawn_subprocess=spawn_subprocess)

    def predict(self, batch, request):
        input_data = batch[self.inputs[0]].data
        outputs = self._predict_fn(input_data)
        for idx, key in self.outputs.items():
            if key in request:
                spec = self.spec[key].copy()
                spec.roi = request[key].roi
                batch.arrays[key] = gp.Array(outputs[idx], spec)


OutDataType = Annotated[
    Raw | Affs | LSD,
    Field(discriminator="dataset_type"),
]


class Predict(BlockwiseTask):
    task_type: Literal["predict"] = "predict"
    checkpoint: Annotated[
        TorchModel | JaxModel,
        Field(discriminator="model_type"),
    ]
    in_data: Raw
    out_data: list[OutDataType | None]

    fit: Literal["overhang"] = "overhang"
    read_write_conflict: Literal[False] = False

    @property
    def out_array_dtype(self) -> np.dtype:
        return np.dtype(np.uint8)

    @property
    def checkpoint_config(self) -> Model:
        return self.checkpoint

    @property
    def write_roi(self) -> Roi:
        in_data_roi = self.in_data.array("r").roi
        if self.roi is not None:
            return in_data_roi.intersect(self.roi)
        else:
            return in_data_roi

    @property
    def voxel_size(self) -> Coordinate:
        return self.in_data.array("r").voxel_size

    @property
    def write_size(self) -> Coordinate:
        return self.checkpoint_config.eval_output_shape * self.voxel_size

    @property
    def context_size(self) -> Coordinate | tuple[Coordinate, Coordinate]:
        context = self.checkpoint_config.context
        if isinstance(context, Coordinate):
            return context * self.voxel_size
        elif isinstance(context[0], Coordinate) and isinstance(context[1], Coordinate):
            return (context[0] * self.voxel_size, context[1] * self.voxel_size)
        else:
            raise NotImplementedError(
                f"Unsupported context {context} type: {type(context)}. Expected Coordinate or tuple of Coordinates."
            )

    @property
    def task_name(self) -> str:
        return f"{'-'.join(dataset.name for dataset in self.output_datasets)}-{self.task_type}"

    @property
    def output_datasets(self) -> list[Dataset]:
        return [out_data for out_data in self.out_data if out_data is not None]

    def drop_artifacts(self):
        for out_data in self.out_data:
            if out_data is not None:
                out_data.drop()

    def init(self):
        self.init_out_array()

    def init_out_array(self):
        # TODO: hardcoding assuming channels first :/
        in_data = self.in_data.array("r")
        units = in_data.units
        axis_names = in_data.axis_names[-in_data.voxel_size.dims :]
        types = in_data.types[-in_data.voxel_size.dims :]
        for out_data, num_channels in zip(
            self.out_data, self.checkpoint_config.num_out_channels
        ):
            if num_channels is None:
                num_channels = self.in_data.array("r").shape[0]
            if out_data is not None:
                shape = (num_channels, *(self.write_roi.shape / self.voxel_size))
                chunk_shape = (num_channels, *self.write_size / self.voxel_size)
                out_data.prepare(
                    shape=shape,
                    chunk_shape=chunk_shape,
                    offset=self.write_roi.offset,
                    voxel_size=self.voxel_size,
                    units=units,
                    axis_names=[f"{out_data.name}^"] + axis_names,
                    types=[out_data.name] + types,
                    dtype=self.out_array_dtype,
                )

    @contextmanager
    def process_block_func(self):
        try:
            client = daisy.Client()
            device = self.checkpoint.select_device(client.worker_id)
        except KeyError:
            device = self.checkpoint.select_device(0)

        logging.info(f"using device {device}")

        input_key = gp.ArrayKey("INPUT_KEY")
        output_keys = [
            gp.ArrayKey(f"OUTPUT_KEY_{i}") for i in range(len(self.out_data))
        ]

        in_array = self.in_data.array("r")

        with self.checkpoint.predict(device) as predict_fn:
            pipeline = ArraySource(input_key, in_array)

            pipeline += gp.Pad(input_key, size=None)

            if in_array.channel_dims < 2:
                pipeline += gp.Stack(1)
            if in_array.channel_dims < 1:
                pipeline += gp.Stack(1)

            pipeline += CallablePredict(
                predict_fn=predict_fn,
                input_key=input_key,
                output_keys={
                    i: output_key
                    for i, output_key in enumerate(output_keys)
                    if self.out_data[i] is not None
                },
                spawn_subprocess=self.num_cache_workers is not None
                and self.num_cache_workers > 1,
            )

            pipeline += gp.Squeeze(
                [
                    output_key
                    for i, output_key in enumerate(output_keys)
                    if self.out_data[i] is not None
                ]
            )

            for output_key, out_data in zip(output_keys, self.out_data):
                if out_data is not None:
                    pipeline += ArrayWrite(
                        output_key, Dataset.array(out_data, "a"), self.checkpoint.to_out_dtype
                    )

            print("Starting prediction...")

            with gp.build(pipeline):

                def process_block(block):
                    if np.count_nonzero(in_array.to_ndarray(block.write_roi)) == 0:
                        return

                    request = gp.BatchRequest()
                    request[input_key] = gp.ArraySpec(roi=block.read_roi)
                    for i, output_key in enumerate(output_keys):
                        if self.out_data[i] is not None:
                            request[output_key] = gp.ArraySpec(roi=block.write_roi)
                    pipeline.request_batch(request)

                yield process_block
