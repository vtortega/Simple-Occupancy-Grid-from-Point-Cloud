# Point Cloud Plane Level Detector
This program's goal is to identigy the levels inside a point cloud and extract them as new point clouds so that we can process each level individually, for making 2D occupancy maps, for example.

## How it Works

It works by using the histogram method.

The allignement of the point cloud is important for the algorithm. A point cloud tilted 30 degrees will yield different results from the same cloud gravity or plane alligned, that's why this algorithm provides a plane fit step(If you feel like you have a better alligned method, feel free to use it and not pass the plane `--fit_plane` parameter).

## Attention Points

* Usually, with lidar like the MID360, the level with the most points in a scan of a single level will be the ceiling of the level, so the `--num_levels 1` will end up selecting the ceiling instead of the floor. For this case it's recommended to use the `2` instead of `1` for the parameter value, so that the `L2` will be the floor you actually want.