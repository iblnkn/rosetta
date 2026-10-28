# Progress of an episode, published on the manager's ~/status topic while it
# runs, however it was started. Same fields as ManageEpisode feedback.
float64 elapsed_s                   # Seconds since episode start
float64 current_reward              # Latest reward value
string control_source               # "policy" | "teleop"
string outcome                      # Task verdict so far
int32 messages_written              # Messages recorded so far
