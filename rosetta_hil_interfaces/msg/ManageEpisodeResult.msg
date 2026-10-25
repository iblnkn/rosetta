# How an episode ended, published once on the manager's ~/result topic
# (transient-local), however it was started. Same fields as ManageEpisode's
# result; termination_reason and outcome take the values ManageEpisode declares.
string termination_reason           # Why the episode ended
string outcome                      # Task verdict
string message                      # Detail when something went wrong; empty otherwise
string bag_path                     # Path to the recorded rosbag
float64 final_reward                # Last reward value at episode end
int32 messages_written              # Total messages recorded
