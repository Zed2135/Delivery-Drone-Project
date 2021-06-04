1. Data source 
We built a drone with a F450 frame, and four XX2216 motors. There is a power module v1.0 equipped with the drone to monitor current and voltage of the battery. 
The log data were collected from the flight control board of the drone, Pixhawk 2.4.8, via MAVLINK using the Mission Planner platform. 
There are also log files (with the same information, but a different data structure) on the flight control board sd card, 
but those files cannot be distinguished by timestamps easily, and thus are hard to analyze. 
Therefore, we only utilize the log data transmitted to PC, and due to transmission interference, a few data points are missing, but most are fine.

2. Data format 
To utilize the log data, we first transformed those '.tlog' files into '.mat' files. 
Then, we used MATLAB to analyze the data, because Mission Planner provides the transformation of log into '.mat' files directly.  
You may transform the log files into other types as well.

3. Experiment settings
1) Power exploration experiments
The 'Power V.S. Weight' folder contains log data we used in exploring the relationship between delivery weight and drone power. 
Due to safety concern, we only operated the drone in a playground without other people. 
We did the hovering and straight flight experiments separatively to differentiate the two statuses. 
In the paper, we plotted the two statuses in one figure to show their similarities and differences.

a) hovering
In each weight setting (from 0g to 600g), we let the drone hover at a location for 120s. 
In data analysis, we filtered the data from the drone's takeoffs, landings.

b) straight flight
In each weight setting (from 0g to 600g), we let the drone fly at 5m/s for 100m back and forth 3 times using flight planning of Mission Planner, which means 600m in total.
Therefore, in data analysis, we filtered the data from the drone's takeoffs, landings, and turns. 
The drone was very shaky when it was flying with over 500g weight, so you may find those log files strange. 
We also filtered the unreliable data eventually. 

2) Case study
We set 6 routes and several weight settings, which are indicated by the folder names.
Route 4 was repeated in the same weight setting twice, for two baselines.
Be aware the experiment figure in the paper is a 180 degree rotation of the google map, since it was taken by the drone.

4. Data content
The flight log provides comprehensive information of the drone after it is turned on. The data includes:
1) Movement information, such as accelaration, velocity, altitude, the motor status.
2) Power information, such as remaining battery, current, voltage.
3) Environment information, such as airspeed.
In this paper, we used the following information to calculate accurate power (there is a variable storing the power information, but the accuracy is concerned):
'current_battery_mavlink_sys_status_t': the real-time current of the whole system
'voltage_battery_mavlink_sys_status_t': the real-time voltage of the whole system
Since the sampling rate is not constant, all the data are labeled with timestamps. 
