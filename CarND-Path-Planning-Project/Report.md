# Autonomous Highway Driving Project

## Result

<img src="./media/highway-driving.gif" alt="highway-driving" style="zoom:200%;" />

## Project Description

### 1. Lane keeping

#### 1.1 Project path points

5 waypoints are projected in front of the car, first two points are aligned with car heading, can be get from last two points of previous path points. The other 3 points are projected in front of the car with 50, 80, 110 accordingly along the s axis of Frenet coordinate.

#### 1.2 Create spline to smooth the path

Based on the 5 points from the previous step, a spline is created to follow the 5 points. This spline ensures the path to be smooth to avoid motion jerk of the car.

#### 1.3 Interpolate points

To translate the spline function into a path (a vector of waypoints), 50 points are interpolate with fixed interval along the spline.

### 2. Collision avoidance

#### 2.1 Surrounding awareness

Surrounding cars information is read from output of sensor fusion. If there is a car in front and its projected position when our car reach the last waypoint, is within a safety distance of 40 m, a `too_close` flag is raised.

#### 2.2 Velocity keeping

If a `too_close` flag is raised, a target velocity is calculated based on the car velocity, to make the car to maintain the same speed as the car in front.

If the car velocity is much slower than our car, target velocity will be reduced significantly to allow emergency brake.

### 3. Lane shifting

#### 3.1 Passing front vehicle

If the `too_close` flag is raised, our car will check if the neighbor lane is free. If the neighbor lane is free, a target lane will be set, and a smooth lane changing path will be generated automatically. 

#### 3.2 Priorities middle lane

Due to the fact that the middle lane gives our car more lane choices (left and right), the middle lane is always being priorities, meaning if our car is at the left or right lane, it will always try to shift to the middle lane if the middle lane is totally free.

#### 3.3 Priorities left lane

Due to the fact that the left lane is a faster lane (at least US road is), the cars on left lane are always faster. Therefore, if the car meet a slow car in the middle lane, left shifting is always priorities over right shifting.

### 4. State machine

A state of `keeping:0` and `changing:1` are keep tracked. It is useful to avoid our car to make multiple lane shifting back to back, which causes a big jerk/acceleration of the motion. Therefore, a lane shifting will only be initiated when the car is in `keeping:0` state.

 