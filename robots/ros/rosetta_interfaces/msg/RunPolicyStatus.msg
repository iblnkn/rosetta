# Progress of a policy run, published on the runner's ~/status topic while it
# runs, however it was started. Same fields as RunPolicy feedback.
uint32 published_actions         # Total actions published so far
uint32 queue_depth               # Current action queue depth
