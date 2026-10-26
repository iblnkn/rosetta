# How a recording ended, published once on the recorder's ~/result topic
# (transient-local), however it was started. Same fields as RecordEpisode's
# result; termination_reason takes the values RecordEpisode declares.
string termination_reason        # Why recording ended
string message                   # Detail when something went wrong; empty otherwise
string bag_path                  # Path to the recorded rosbag
int32 messages_written           # Total messages captured
