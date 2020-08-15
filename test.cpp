#include <fdeep/fdeep.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <fstream>
#include <string>

#include "simulation/before_simulation.hpp"

extern "C"{
    #include "asv-swarm/include/asv.h"
    #include "asv-swarm/include/io.h"
}

int main(int argc, char** argv){

    std::ofstream f;
    std::string record_file = "/lyceum/rz2u19/DeepRL/record/record.csv";

    f.open(record_file, std::ios::out | std::ios::in | std::ofstream::trunc);
    f << "pre_state" << "," << "action" << "," << "reward" << "," << "state\n";
    struct Asv asv;

    double wave_ht = 0.0;
    double wave_heading = 0.0;

    init(asv, wave_ht, wave_heading);

    struct Dimensions waypoint;

    get_random_waypoint(waypoint);
    std::cout << "waypoint: " << waypoint.x << ", " << waypoint.y << std::endl;

    int t = 0;
    int max_t = 10000;
    double time = 0.0;

    std::default_random_engine generator;
    std::normal_distribution<double> dist(0.0, 0.2);

    std::string model_file = "/lyceum/rz2u19/DeepRL/model/action_model.json";

    const auto action_model = fdeep::load_model(model_file);

    std::vector<float> n = {0, 0, 0, 0};

    float prev_angle, prev_distance, prev_attitude_x, prev_attitude_y, prev_attitude_z, prev_velocity, prev_accelerate;
    get_state(asv, waypoint, prev_angle, prev_distance, prev_attitude_x, prev_attitude_y, prev_attitude_z, prev_velocity, prev_accelerate);
    float prev_normalized_distance = 2.0 * atan(prev_distance/50.0) / M_PI;


    for(; t<max_t; t++){

        const auto result = action_model.predict(
        {fdeep::tensor(fdeep::tensor_shape(static_cast<std::size_t>(4)),
        std::vector<float>{prev_angle,prev_normalized_distance,prev_attitude_z, prev_velocity})});
        get_nosie(n);
        std::vector<float> force = {0,0,0,0};
        for(int p=0; p<4; p++){
             force[p] = result[0].get(fdeep::tensor_pos(p)) + 1;
            if(force[p] > 2)
                force[p] = 2;
            else if(force[p] < 0)
                force[p] = 0;
            asv.propellers[p].thrust = force[p]; //N
            asv.propellers[p].orientation = (struct Dimensions){0.0, 0.0, 0.0};
        }
        time = t * asv.dynamics.time_step_size;
        asv_compute_dynamics(&asv, time);

        float angle, distance, attitude_x, attitude_y, attitude_z, velocity, accelerate;
        get_state(asv, waypoint, angle, distance, attitude_x, attitude_y, attitude_z, velocity, accelerate);
        float normalized_distance = 2.0 * atan(distance/50.0) / M_PI;

        //reward
        float reward = -distance;

        if(compute_distance(asv.cog_position.x, asv.cog_position.y, waypoint.x, waypoint.y) < 1){
            f << prev_angle << " " << prev_normalized_distance << " " << prev_attitude_z <<" " << prev_velocity <<",";
            f << force[0] << " " << force[1] << " " << force[2] << " " << force[3] <<",";
            f << 0 << ",";
            f << angle << " " << normalized_distance << " " << attitude_z << " " << velocity << "\n";
            break;
        }

        f << prev_angle << " " << prev_normalized_distance << " " << prev_attitude_z <<" " << prev_velocity <<",";
        f << force[0] << " " << force[1] << " " << force[2] << " " << force[3] <<",";
        f << reward << ",";
        f << angle << " " << normalized_distance << " " << attitude_z <<" " << velocity << "\n";


        prev_angle = angle;
        prev_distance = distance;
        prev_attitude_x = attitude_x;
        prev_attitude_y = attitude_y;
        prev_attitude_z = attitude_z;
        prev_normalized_distance = normalized_distance;
        prev_velocity = velocity;
        prev_accelerate = accelerate;

    }

    f.close();


//    const auto action_model = fdeep::load_model("../model/action_model.json");
//    const auto result1 = action_model.predict(
//        {fdeep::tensor(fdeep::tensor_shape(static_cast<std::size_t>(5)),
//        std::vector<float>{0.4,0.2,0.3,0.8,0.2})});
//    std::cout << fdeep::show_tensors(result1) << std::endl;
//
//
//    std::cout << result1[0].get(fdeep::tensor_pos(0)) << std::endl;
    return 0;

}