# Train-Serve Skew

Rosetta matches training input to live input. Offline conversion (`bag_frames`) and online inference (`topic_bridge`) run the same `StreamBuffer` resampling, `aggregate_frame`, and operator pipeline from the same contract, so decode, `select`, `apply` operators, alignment, and key aggregation match across all supported frameworks.
