mkdir -p ~/rosetta_ws/src
cd ~/rosetta_ws/src
if [ ! -d "rosetta_interfaces" ]; then
  git clone https://github.com/iblnkn/rosetta_interfaces.git
fi
cd ~/rosetta_ws
rsync -av \
  --exclude='*.mcap' \
  --exclude='.git' \
  /workspace/ src/rosetta/

source /opt/ros/humble/setup.bash
colcon build --packages-ignore turtlebot3_gazebo turtlebot3_fake_node
source install/setup.bash