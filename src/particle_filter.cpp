/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

/**
 * Initialize particle filter with a set number of particles with positions
 * that are randomly sampled from a Gaussian distribution with mean at the
 * GPS location with a specified uncertainty.  All particle weights are also
 * initialized to 1.0 to be updated later.
 */
void ParticleFilter::init(double x, double y, double theta, double std[]) {
	
	// Set a fixed number of particles
	num_particles = 100;
	
	default_random_engine gen;
	double std_x = std[0], std_y = std[1], std_theta = std[2]; // Standard deviations for x, y, and theta
	
	// This line creates a normal (Gaussian) distribution for x
	normal_distribution<double> dist_x(x, std_x);
	
	// This line creates a normal (Gaussian) distribution for y
	normal_distribution<double> dist_y(y, std_y);
	
	// This line creates a normal (Gaussian) distribution for theta
	normal_distribution<double> dist_theta(theta, std_theta);
	
	for ( int i = 0; i < num_particles; i++) {
		
		//create a particle object
		Particle particle;
		
		particle.x = dist_x(gen);
		particle.y = dist_y(gen);
		particle.theta = dist_theta(gen);	
		particle.id = i;		
		// Normalize theta to [0, 2*pi] 
		while (particle.theta > 2*M_PI) { particle.theta -= 2*M_PI; }
		while (particle.theta < 0) { particle.theta += 2*M_PI; }
		particle.weight = 1.0;
		
		//push to particles vector
		particles.push_back(particle);

	}
	
	is_initialized = true;
}

/**
 * Predict where each particle would move to after time delta_t from the previous 
 * time step using CTRV motion equations, and the controlled velocity
 * and yaw_rate.  
 * Next, randomly sample from Gaussian distribution around this
 * predicted mean position with the given standard deviations for x, y, and
 * theta.
 */
void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	
	default_random_engine gen;
	double std_x = std_pos[0], std_y = std_pos[1], std_theta = std_pos[2]; // Standard deviations for x, y, and theta
	// This line creates a normal (Gaussian) distribution for x
	normal_distribution<double> dist_x(0, std_x);
	// This line creates a normal (Gaussian) distribution for y
	normal_distribution<double> dist_y(0, std_y);	
	// This line creates a normal (Gaussian) distribution for theta
	normal_distribution<double> dist_theta(0, std_theta);
	
	//pre-compute variable to save execution time
	const double vel_dt = velocity * delta_t;
	const double yaw_dt = yaw_rate * delta_t;
	const double vel_over_yawrate = velocity/yaw_rate;
	
	for (int i = 0; i < num_particles; i++) {
		//chose the motion model based on threshold for yaw_rate 
		if (fabs(yaw_rate) > 0.0001) {
			double theta_new = particles[i].theta + yaw_dt;
            particles[i].x += vel_over_yawrate * (sin(theta_new) - sin(particles[i].theta));
            particles[i].y += vel_over_yawrate * (-cos(theta_new) + cos(particles[i].theta));
            particles[i].theta = theta_new;			
		}
		else {
			particles[i].x += vel_dt * cos(particles[i].theta);
            particles[i].y += vel_dt * sin(particles[i].theta);
            // particles[i].theta remains same if yaw_rate is too small
		}
		 // Add random Gaussian noise
		particles[i].x += dist_x(gen);
        particles[i].y += dist_y(gen);
        particles[i].theta += dist_theta(gen);
		// Normalize theta to [0, 2*pi] 
		while (particles[i].theta > 2*M_PI) { particles[i].theta -= 2*M_PI; }
		while (particles[i].theta < 0) { particles[i].theta += 2*M_PI; }
	}

}

/**
* Find the predicted measurement that is closest to each observed measurement and assign the 
*	    observed measurement to this particular landmark.
*/
void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations, std::vector<int>& associations, std::vector<double>& sense_x, std::vector<double>& sense_y ) {
	
	unsigned int nObs = observations.size();
	unsigned int nLandmarks = predicted.size();
	
	for (unsigned int i = 0; i < nObs; i++) { // For each observation
	
		// Initialize minDistance to a large num
		double minDistance = numeric_limits<double>::max();
		
		// Initialize the landmark ID
		int landmarkId = -1;
	
		for (unsigned int j = 0; j < nLandmarks; j++) { // For each landmarks {
			double xDistance = observations[i].x - predicted[j].x;
			double yDistance = observations[i].y - predicted[j].y;
			double distance = xDistance * xDistance + yDistance * yDistance;
			
			 // If the distance is the least, this landmark is the closest landmark that needs to be associated with the observation
			if (distance < minDistance) {
				minDistance = distance;
				landmarkId = predicted[j].id; 
			}
		}
		
		// assign the observation identifier.
		observations[i].id = landmarkId;
		
		// Add associated observation info to visualization vectors
      associations.push_back(observations[i].id);
      sense_x.push_back(observations[i].x);
      sense_y.push_back(observations[i].y);
	
	}
	

}

/**
 * Update the weight of each particle based on the multivariate Gaussian
 * probability of the particle's predicted landmark measurement positions vs
 * the actual measured observations.  
 *   1. Filter known map landmarks to find which ones are within the
 *      particle's sensor range 
 *   2. Convert vehicle's measured observations to map coordinates and associate
 *      each measured observation to the nearest (filtered) landmark 
 *   3. For each observation, calculate the multivariate Gaussian probability between each observed
 *      measurement and the matched landmark x,y position and set the particle's
 *      weight by multiplying them all together
 *   4. Set the particle's association information for visualization
 *  
 *
 * References:
 * https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
 * http://planning.cs.uiuc.edu/node99.html 
 */

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {

	
	double stdLandmark_x = std_landmark[0];
	double stdLandmark_y = std_landmark[1];
	double sensor_range2 = sensor_range * sensor_range;
	
	// Reset all particle weights before updating
	weights.clear();
	weights.resize(num_particles);
	
	for (int i = 0; i < num_particles; i++) {
		
		double x = particles[i].x;
		double y = particles[i].y;
		double theta = particles[i].theta; 
		
		vector<LandmarkObs> withinRangeLandmarks;
		
		for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++) {
			
			LandmarkObs withinRangeLandmark;
			withinRangeLandmark.x = map_landmarks.landmark_list[j].x_f;
			withinRangeLandmark.y = map_landmarks.landmark_list[j].y_f;
			withinRangeLandmark.id = map_landmarks.landmark_list[j].id_i;
			
			double dX = x - withinRangeLandmark.x;
			double dY = y - withinRangeLandmark.y;
			double particle_range = dX * dX + dY * dY;
			
			//check if landmark location is within the range of the given particle
			if(particle_range <= sensor_range2) {
				withinRangeLandmarks.push_back(withinRangeLandmark);
			}
		}
		
		 // Transform observation coordinates.
		vector<LandmarkObs> mappedObservations;
		for(unsigned int j = 0; j < observations.size(); j++) {
			double x_m = cos(theta)*observations[j].x - sin(theta)*observations[j].y + x;
			double y_m = sin(theta)*observations[j].x + cos(theta)*observations[j].y + y;
			mappedObservations.push_back(LandmarkObs{ observations[j].id, x_m, y_m });
		}
		
		// Vectors to store the particles association info for visualization
		vector<int> associations;
		vector<double> sense_x;
		vector<double> sense_y;
		
		// Nearest neighbour association of observations to landmark.
		dataAssociation(withinRangeLandmarks, mappedObservations, associations, sense_x, sense_y);
		//dataAssociation(withinRangeLandmarks, mappedObservations);
		
		particles[i].weight = 1.0;
		
		// Calculate weights
		double weight_normalizer = 2 * M_PI * stdLandmark_x * stdLandmark_y;
		double normalizer_x = 2 * stdLandmark_x * stdLandmark_x;
		double normalizer_y = 2 * stdLandmark_y * stdLandmark_y;
		
		for(unsigned int j = 0; j < mappedObservations.size(); j++) {
			
			double obs_X = mappedObservations[j].x;
			double obs_Y = mappedObservations[j].y;
			int landmark_Id = mappedObservations[j].id;
			bool matched = false;
			unsigned int k = 0;
			double landmarkX, landmarkY;
			
			//matching the landmark and observation IDs
			while(!matched) {
				if(landmark_Id == withinRangeLandmarks[k].id) {
					matched = true;
					landmarkX = withinRangeLandmarks[k].x;
					landmarkY = withinRangeLandmarks[k].y;
				}
				k++;
			}
			
			double dX = obs_X - landmarkX;
			double dY = obs_Y - landmarkY;
			
			double weight_temp = (1/(weight_normalizer)) * exp(-( dX*dX/(normalizer_x) + dY*dY/(normalizer_y)));
			particles[i].weight *= weight_temp;			
		}
		weights[i] = particles[i].weight;
		
		// Set particle's associations for visualization
		SetAssociations(particles[i], associations, sense_x, sense_y);
	}
}

void ParticleFilter::resample() {
	
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	
	double maxWeight = *max_element(weights.begin(), weights.end());
		
	// Create uniform distributions to pick random_weights
	uniform_real_distribution<double> distDouble(0.0, maxWeight);
	uniform_int_distribution<int> distInt(0, num_particles - 1);
	
	default_random_engine gen;
	
	// Generate a random index.
	int index = distInt(gen);	
	double beta = 0.0;
	//sebastian thrun's lecture - random wheel
	vector<Particle> resampledParticles;
	for( int i = 0; i < num_particles; i++) {
		
		beta += distDouble(gen) * 2.0;
		
		while( beta > weights[index]) {
			beta -= weights[index];
			index = (index + 1) % num_particles;
		}
		resampledParticles.push_back(particles[index]);
	}
	
	//assign resampled particles to particles
	particles = resampledParticles;

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
	
	// Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();
	
    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
	
	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
