# Record Raw, Decide Late

Rosetta records demonstrations to [rosbag2](https://github.com/ros2/rosbag2) files first, then converts them to training datasets in a separate step. This page explains why.

Recording is an expensive step. Robot time, operator time, etc. Every decision baked into a recording is permanent, so Rosetta bakes in nothing: bags store raw messages, and the contract assigns meaning at port time. Recorders bound to a training format fix fps, keys, and image size the moment you press record. Deferring buys you:

- **Preserves raw data.** Bags store every message at original rate and timestamp. No alignment, no downsampling, no loss. Change the contract and reprocess without re-recording. Format churn stops mattering: LeRobot's dataset format has moved from v1 to v2 to v3, and bags re-port to each. The dataset is disposable. The bags are the asset.
- **Familiar to ROS2 users.** Bag files are the standard data format in the ROS2 ecosystem, with mature tooling for [recording, playback, inspection](https://docs.ros.org/en/jazzy/Tutorials/Beginner-CLI-Tools/Recording-And-Playing-Back-Data/Recording-And-Playing-Back-Data.html), and analysis. Any bag-file tool works with your recorded data.
- **Stores data beyond what LeRobot needs.** Bags include topics with no LeRobot feature mapping: diagnostics, TF trees, debug streams, extra sensors. This data stays available for analysis, debugging, or future use even outside the training dataset.
- **Leverages MCAP.** Rosetta defaults to [MCAP](https://mcap.dev/) storage, which provides [high-performance](https://mcap.dev/guides/benchmarks/rosbag2-storage-plugins) random-access reads, efficient compression, and broad ecosystem support beyond ROS2.
- **Write-optimized for live recording.** Bag files (especially MCAP) are designed for high-throughput sequential writes with minimal overhead, well-suited for capturing live sensor data. LeRobot datasets (Parquet + MP4) are read-optimized for training but involve more overhead when writing live, including in-memory buffering and post-episode video encoding.

The recorder reinforces this stance by recording **every topic** on the graph by default, not only contract topics (see [Nodes](../reference/nodes.md#topic-recording)). Data you did not know you needed is still in the bag. Add a contract mapping later and re-port.
