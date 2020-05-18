/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <string>
#include <vector>
#include <cassert>

#include "helper_functions.h"

using std::string;
using std::vector;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * Set the number of particles. Initialize all particles to 
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1. 
   * Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method 
   *   (and others in this file).
   */
  num_particles = 150;  // Set the number of particles
  
  std::normal_distribution<double> dist_x(x, std[0]);
  std::normal_distribution<double> dist_y(y, std[1]);
  std::normal_distribution<double> dist_theta(theta, std[2]);
  
  for (int i=0; i<num_particles; i++) {
    Particle p;
    p.x = dist_x(gen);
    p.y = dist_y(gen);
    p.theta = dist_theta(gen);
    p.weight = 1;
    
    particles.push_back(p);
    weights.push_back(1);
  }
  assert((int)particles.size() == num_particles);
  assert((int)weights.size() == num_particles);
  
  is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], 
                                double velocity, double yaw_rate) {
  /**
   * Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution 
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  
  std::normal_distribution<double> dist_x(0.0, std_pos[0]);
  std::normal_distribution<double> dist_y(0.0, std_pos[1]);
  std::normal_distribution<double> dist_theta(0.0, std_pos[2]);
  
  for (auto &p : particles) {
    double noise_x = dist_x(gen);
    double noise_y = dist_y(gen);
    double noise_theta = dist_theta(gen);
    
    if (fabs(yaw_rate) < 0.0001) {
      p.x += velocity*cos(p.theta)*delta_t + noise_x;
      p.y += velocity*sin(p.theta)*delta_t + noise_y;
      p.theta += noise_theta;
    }
    else {
      double new_theta = p.theta+yaw_rate*delta_t;
      p.x += (velocity/yaw_rate)*(sin(new_theta) - sin(p.theta)) + noise_x;
      p.y += (velocity/yaw_rate)*(cos(p.theta) - cos(new_theta)) + noise_y;
      p.theta = new_theta + noise_theta;
    }
  }
  
//   for (int n=0; n < 10; ++n) {
//     std::cout<<particles[n].x<<" "<<particles[n].y<<std::endl;
//   }
  
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted, 
                                     vector<LandmarkObs>& observations) {
  /**
   * Find the predicted measurement that is closest to each 
   *   observed measurement and assign the observed measurement to this 
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will 
   *   probably find it useful to implement this method and use it as a helper 
   *   duriauto ng the updateWeights phase.
   */
  for (unsigned int i=0; i<observations.size(); ++i) {
    double min_dist = std::numeric_limits<double>::max();
    for (unsigned int j=0; j<predicted.size(); ++j) {
      double curr_dist = dist(observations[i].x, observations[i].y, predicted[j].x, predicted[j].y);
      if (curr_dist < min_dist) {
        min_dist = curr_dist;
        observations[i].id = j;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
                                   const vector<LandmarkObs> &observations, 
                                   const Map &map_landmarks) {
  /**
   * Update the weights of each particle using a mult-variate Gaussian 
   *   distribution. You can read more about this distribution here: 
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system. 
   *   Your particles are located according to the MAP'S coordinate system. 
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  
  double std_x = std_landmark[0];
  double std_y = std_landmark[1];
  
  for (int i=0; i<num_particles; ++i) {
    auto &p = particles[i];
    vector<LandmarkObs> predicted_landmarks;
    
    for (auto &l : map_landmarks.landmark_list) {
      if (dist(p.x, p.y, l.x_f, l.y_f) <= sensor_range) predicted_landmarks.push_back(LandmarkObs{l.id_i, l.x_f, l.y_f});
    }
    
    // Transform observation to map's coordinate
    vector<LandmarkObs> observations_map;
    for (auto &obs : observations) {
      LandmarkObs obs_map;
      obs_map.x = p.x + cos(p.theta)*obs.x - sin(p.theta)*obs.y;
      obs_map.y = p.y + sin(p.theta)*obs.x + cos(p.theta)*obs.y;
      obs_map.id = obs.id;
      observations_map.push_back(obs_map);
    }
    
    // Data association
    dataAssociation(predicted_landmarks, observations_map);
    
    double weight = 1.0;
    for (unsigned int j=0; j<observations_map.size(); ++j) {
      auto match_landmark = predicted_landmarks[observations_map[j].id];
      weight *= multiv_prob(std_x, std_y, observations_map[j].x, observations_map[j].y, match_landmark.x, match_landmark.y);
    }
    p.weight = weight;
    weights[i] = p.weight;
    
  }
}

void ParticleFilter::resample() {
  /**
   * Resample particles with replacement with probability proportional 
   *   to their weight. 
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  std::vector<Particle> particles_resampled;
  std::discrete_distribution<> distribution(weights.begin(), weights.end());
  for (int i=0; i<num_particles; ++i) {
    int n = distribution(gen);
    particles_resampled.push_back(particles[n]);
  }
  particles = particles_resampled;
}

void ParticleFilter::SetAssociations(Particle& particle, 
                                     const vector<int>& associations, 
                                     const vector<double>& sense_x, 
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association, 
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}