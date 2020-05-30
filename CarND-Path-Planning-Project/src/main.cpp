#include <uWS/uWS.h>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "helpers.h"
#include "json.hpp"
#include "spline.h"

// for convenience
using nlohmann::json;
using std::string;
using std::vector;

int main() {
  uWS::Hub h;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0
  double max_s = 6945.554;
  double ref_vel = 0.0;
  int lane = 1;
  int state = 0; // 0 = keeping lane, 1 = changing lane

  std::ifstream in_map_(map_file_.c_str(), std::ifstream::in);

  string line;
  while (getline(in_map_, line)) {
    std::istringstream iss(line);
    double x;
    double y;
    float s;
    float d_x;
    float d_y;
    iss >> x;
    iss >> y;
    iss >> s;
    iss >> d_x;
    iss >> d_y;
    map_waypoints_x.push_back(x);
    map_waypoints_y.push_back(y);
    map_waypoints_s.push_back(s);
    map_waypoints_dx.push_back(d_x);
    map_waypoints_dy.push_back(d_y);
  }

  h.onMessage([&map_waypoints_x,&map_waypoints_y,&map_waypoints_s,
               &map_waypoints_dx,&map_waypoints_dy,&ref_vel,&lane,&state]
              (uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
               uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {

      auto s = hasData(data);

      if (s != "") {
        auto j = json::parse(s);
        
        string event = j[0].get<string>();
        
        if (event == "telemetry") {
          // j[1] is the data JSON object
          
          // Main car's localization Data
          double car_x = j[1]["x"];
          double car_y = j[1]["y"];
          double car_s = j[1]["s"];
          double car_d = j[1]["d"];
          double car_yaw = j[1]["yaw"];
          double car_speed = j[1]["speed"];
          
          // Previous path data given to the Planner
          auto previous_path_x = j[1]["previous_path_x"];
          auto previous_path_y = j[1]["previous_path_y"];
          // Previous path's end s and d values 
          double end_path_s = j[1]["end_path_s"];
          double end_path_d = j[1]["end_path_d"];

          // Sensor Fusion Data, a list of all other cars on the same side 
          //   of the road.
          auto sensor_fusion = j[1]["sensor_fusion"];

          json msgJson;

          /**
           * TODO: define a path made up of (x,y) points that the car will visit
           *   sequentially every .02 seconds
           */
          
          int prev_path_size = previous_path_x.size();
          bool too_close = false;
          double target_vel = 50.0;
          
          // Read cars information [ id, x, y, vx, vy, s, d] into a map data
          for (int i=0; i<sensor_fusion.size(); i++) {
            auto car = sensor_fusion[i];
            double d = car[6];
            
            if (fabs(d-(2+4*lane)) < 2) { // car is in the same lane as us
              double vx = car[3];
              double vy = car[4];
              double s = car[5];
              
              double speed = sqrt(vx*vx + vy*vy);
              double future_s = s + 0.02*speed*(double)prev_path_size;
              double safety_dist = 40;
              double different_s = future_s - car_s;
              
              if (future_s > car_s && different_s < safety_dist) {
                too_close = true;
                if (different_s < safety_dist/2.0) target_vel = std::min(speed, ref_vel);
                else target_vel = std::min(speed*2.2, ref_vel);
              }
            }
          }
          
          // check if car reach the target lane
          if (state == 1 && fabs(car_d-(2+4*lane)) < 0.2) {
            state = 0;
          }
          
          if (too_close) {
            if (ref_vel > target_vel) ref_vel -= (ref_vel - target_vel)/15.0;
            
            // Change lane logic
            if (lane == 0 || lane == 2) {
              if (checkLaneFree(1, sensor_fusion, prev_path_size, car_s, 50)) {
                lane = 1;
                state = 1;
              }
            }
            else if (lane == 1) {
              if (checkLaneFree(0, sensor_fusion, prev_path_size, car_s, 50)) {
                lane = 0;
                state = 1;
              }
              else if (checkLaneFree(2, sensor_fusion, prev_path_size, car_s, 60)) {
                lane = 2;
                state = 1;
              }
            }
          }
          else {
            if (lane != 1) {
              if (checkLaneFree(1, sensor_fusion, prev_path_size, car_s, 100) && state == 0) {
                lane = 1;
                state = 1;
              }
            }
            if (ref_vel < target_vel - 0.5) {
              double diff_vel = target_vel - 0.5 - ref_vel;
              if (diff_vel > 20.0) ref_vel += (diff_vel)/40.0;
              else ref_vel += (diff_vel)/30.0;
            }
          }
          
          vector<double> ptsx;
          vector<double> ptsy;
          
          double ref_x = car_x;
          double ref_y = car_y;
          double ref_yaw = deg2rad(car_yaw);
          
          if (prev_path_size < 2) {
            
            double prev_car_x = car_x - cos(car_yaw);
            double prev_car_y = car_y - sin(car_yaw);
            
            ptsx.push_back(prev_car_x);
            ptsx.push_back(car_x);
            
            ptsy.push_back(prev_car_y);
            ptsy.push_back(car_y);
            
          } else {
            ref_x = previous_path_x[prev_path_size-1];
            ref_y = previous_path_y[prev_path_size-1];
            
            double prev_ref_x = previous_path_x[prev_path_size-5];
            double prev_ref_y = previous_path_y[prev_path_size-5];
            ref_yaw = atan2(ref_y-prev_ref_y, ref_x-prev_ref_x);
            
            ptsx.push_back(prev_ref_x);
            ptsx.push_back(ref_x);
            
            ptsy.push_back(prev_ref_y);
            ptsy.push_back(ref_y);
          }
            
          // In frenet add evenly 30m spaced points ahead of starting reference
          vector<double> next_wp0 = getXY(car_s+50, 4/2 + (4 * lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
          vector<double> next_wp1 = getXY(car_s+80, 4/2 + (4 * lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);
          vector<double> next_wp2 = getXY(car_s+110, 4/2 + (4 * lane), map_waypoints_s, map_waypoints_x, map_waypoints_y);

          ptsx.push_back(next_wp0[0]);
          ptsx.push_back(next_wp1[0]);
          ptsx.push_back(next_wp2[0]);

          ptsy.push_back(next_wp0[1]);
          ptsy.push_back(next_wp1[1]);
          ptsy.push_back(next_wp2[1]);

          // Making coordinates to local car coordinates.
          for ( int i = 0; i < ptsx.size(); i++ ) {
            double shift_x = ptsx[i] - ref_x;
            double shift_y = ptsy[i] - ref_y;

            ptsx[i] = shift_x * cos(0.0 - ref_yaw) - shift_y * sin(0.0 - ref_yaw);
            ptsy[i] = shift_x * sin(0.0 - ref_yaw) + shift_y * cos(0.0 - ref_yaw);
          }

          // Create the spline.
          tk::spline s;
          s.set_points(ptsx, ptsy);

          //
          vector<double> next_x_vals;
          vector<double> next_y_vals;

          for ( int i = 0; i < prev_path_size; i++ ) {
            next_x_vals.push_back(previous_path_x[i]);
            next_y_vals.push_back(previous_path_y[i]);
          }

          // Calculate distance y position on 30 m ahead.
          double target_x = 30.0;
          double target_y = s(target_x);
          double target_dist = sqrt(target_x*target_x + target_y*target_y);

          double x_add_on = 0;

          // fill up the rest of our path planner after filling it with previous points, here we will always output 50 points
          for( int i = 1; i <= 50 - prev_path_size; i++ ) {

            // 2.24 constant is to convert miles per hours to meter per seconds
            double N = target_dist/(0.02*ref_vel/2.24);
            double x_point = x_add_on + target_x/N;
            double y_point = s(x_point);
            x_add_on = x_point;

            double x_ref = x_point;
            double y_ref = y_point;

            // rotate back to normal after rotating it earlier
            x_point = x_ref * cos(ref_yaw) - y_ref * sin(ref_yaw);
            y_point = x_ref * sin(ref_yaw) + y_ref * cos(ref_yaw);

            x_point += ref_x;
            y_point += ref_y;

            next_x_vals.push_back(x_point);
            next_y_vals.push_back(y_point);
          }


          msgJson["next_x"] = next_x_vals;
          msgJson["next_y"] = next_y_vals;

          auto msg = "42[\"control\","+ msgJson.dump()+"]";

          ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
        }  // end "telemetry" if
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }  // end websocket if
  }); // end h.onMessage

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  
  h.run();
}