# How a policy run ended, published once on the runner's ~/result topic
# (transient-local), however it was started. Same fields as RunPolicy's result;
# termination_reason takes the values RunPolicy declares.
string termination_reason        # Why the run ended
string message                   # Detail when something went wrong; empty otherwise
