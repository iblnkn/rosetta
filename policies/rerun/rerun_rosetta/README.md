# rerun_rosetta

Rerun dataset backend for Rosetta: train directly from `.rrd` recordings
registered to a [Rerun](https://rerun.io) catalog server, with no export step
and no second copy of the data. It imports no robot middleware. The
contract's robot interface says how its recordings are laid out, and the
`ros2` interface does so for bags recorded with
[`rosbag2_storage_rrd`](https://github.com/rerun-io/rosbag2_storage_rrd):

```bash
ros2 launch rosetta_ros episode_recorder_launch.py contract_path:=robot.yaml storage_id:=rrd
```

## How it fits

The contract already has two projections: eager frames for the offline porter
(`iter_bag_frames`) and for the live bridge (`TopicBridge`). This package adds
a lazy one on [Rerun's dataloader](https://rerun.io/docs/howto/train/dataloader):

```text
contract ──► compile_contract ──► one Field per stream ──► Manifest ──► RerunIterableDataset
                                       │                                      │
                                       └─ ContractDecoder:                    └─ FrameLayout.assemble
                                          Arrow row ─► layout.message ─► decode_value(msg, spec)
```

The layout belongs to the robot interface. Each interface registers a
`RecordingLayout` under the `rerun_rosetta.layouts` entry-point group, named
like the interface. It names the column holding each channel, the Rerun
timeline holding receive times, and how one recorded message becomes the input
the interface's decoders take. `ros2`'s, `Rosbag2RrdLayout` in `rosetta_ros`,
serves each reflected message (a struct mirroring the message definition) as a
read-only attribute view.

Decoding is Rosetta's. Each message goes through `decode_value` with the
codecs the contract's robot interface registered, so select, apply and width
rules are shared with the porter and the live bridge. Each recorded row is
decoded once per fetch block, however many action windows include it.

Sampling is Rerun's. The server's latest-at fill is the contract's `hold`
alignment. The fixed-rate grid, windows, shuffling and worker/rank sharding
are the dataloader's.

Rerun's grid steps by a whole number of nanoseconds, `round(1e9 / fps)`,
while the porter places tick `k` at `floor(k * 1e9 / fps)`. The grids match
when `fps` divides 1e9 (10, 20, 25 or 50 fps, say). At other rates they
drift apart by about one nanosecond every three ticks, 36 µs after an hour
at 30 fps, and a message stamped inside that gap lands one frame apart from
where the porter puts it. Rerun 0.41's `FixedRateSampling` takes only a rate,
so the exact grid can't be requested.

Warmup is a manifest rule. The manifest is built with every observation field
required, so ticks before all observation streams have a sample are dropped,
as the porter skips them.

## Usage

```python
from rerun.catalog import CatalogClient
import torch
from rerun_rosetta import contract_dataset

client = CatalogClient("rerun+http://127.0.0.1:51234")
client.create_dataset("my_task", exist_ok=True).register_prefix("file:///abs/path/bags").wait()

frames = contract_dataset("rerun+http://127.0.0.1:51234", "my_task", "contract.yaml",
                          windows={"action": (0, 49)})
loader = torch.utils.data.DataLoader(frames, batch_size=8)
for epoch in range(10):
    frames.set_epoch(epoch)
    for batch in loader:
        ...
```

Start a server with `rerun server` (ships with `rerun-sdk`). `file://` needs the
server on the same filesystem as the bags. With `num_workers > 0`, pass the
same `num_workers` to `contract_dataset` and `multiprocessing_context="spawn"`
to the `DataLoader`. `persistent_workers=True` works: call `set_epoch` before
each epoch's loop, as the example does, and the workers pick up the new order.

Samples are shuffled with Rerun's `SampleShuffle` by default, so every batch
mixes ticks from across the dataset. `shuffle_strategy=BlockShuffle(buffer_size=N)`
fetches contiguous blocks instead, which is faster when fetching is the
bottleneck, and mixes decoded samples through an `N`-sample buffer before
emitting them. Each `DataLoader` worker holds `N` decoded samples, and batches
stay correlated by about 128 / `N` (128 samples per fetch block). A
`BlockShuffle` without `buffer_size` emits runs of consecutive ticks, so it
raises. `NoShuffle()` gives natural order.

Building the manifest scans the whole dataset on every run. Pass
`manifest="manifest.parquet"` to write it on the first run and load it after
that. A saved manifest built for another dataset, contract, `windows` or
`num_workers` raises. It lists the samples that existed when it was built, so
delete it after registering new recordings.

[`examples/train_act.py`](examples/train_act.py) trains LeRobot's ACT end to end.

## Normalization stats

`write_stats_layer(dataset, "contract.yaml", "stats_dir")` computes exact
normalization statistics and stores them as a Rerun layer beside the
recordings, one layer per contract, named `rosetta_stats_<hash of the contract
file>`. Each episode's layer holds the frame values of its numeric keys and
per-channel pixel histograms of its image keys, after the contract's
transforms. `load_stats(dataset, "contract.yaml")` merges them and returns, per
key, `count` (frames), `mean`, `std`, `min`, `max` and the quantiles `q01`,
`q10`, `q50`, `q90` and `q99`, computed over every frame and every pixel. Image
stats are per channel, in pixel values 0 to 255.

`write_stats_layer` computes only the episodes that don't have the layer yet,
so after registering new recordings, call it again. It writes the layer files
under the directory you pass and registers them from there, so a `file://`
catalog needs that directory on the server's filesystem, and a hub needs the
files uploaded. Any edit to the contract file, even to a comment, makes a new
layer.

## Limits

- `align.strategy: hold` only, and every stream aligned on the layout's
  receive timeline (`receive` for `ros2`), checked when the contract compiles.
  The porter anchors its tick grid at the bag's first receive time. The
  dataloader anchors on the index timeline, so any other alignment would
  sample different ticks.
- A windowed key never runs past the episode: the dataloader trims the last
  `window - 1` ticks of each episode as sample starts, where LeRobot pads them
  and masks them with `<key>_is_pad`.
- With `ros2` recordings, topics must be recorded `reflected` (the plugin's
  default). A topic recorded only `raw` raises when the contract compiles. Add
  the reflected form with `ros2 bag convert` (see the plugin's README). A
  topic recorded only as `lenses` reads as not recorded: an observation
  raises, and an action zero-fills with a warning.
- With `ros2` recordings, record without `--max-bag-size`: each file of a
  split bag gets its own recording id and registers as its own episode.
- No episode markers or `task` strings in samples.
- The sample grid adds a fixed whole-nanosecond step, so at a frame rate whose
  period isn't a whole number of nanoseconds it drifts from the porter's exact
  grid by up to 0.5 ns per tick: 18 µs per hour at 15 fps, 72 µs at 60 fps.
- One training process only. The manifest is built for a single rank, so
  iterating under `torch.distributed` with more than one process raises
  `Manifest was built for 1x<workers> (ranks x workers_per_rank)`.
- Raw `sensor_msgs/Image` topics fetch uncompressed pixels, including during
  the manifest scan, which reads whole chunks. For the SO-ARM101 Gazebo demo's
  two 640x480 cameras the scan fetches about 2 GB per 35-second episode.
  Prefer `CompressedImage` topics for camera streams, and save the manifest
  (see Usage) so the scan runs once.
