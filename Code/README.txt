This folder contains RT-Drone and RTW-Drone algorithms.
I deleted the other baselines since they are not proposed by us.

1. Input requirement
The 'Beta.csv', 'Distance.csv', 'eta.csv', 'Indicator.csv', 'RouteSet.csv', 'Size.csv', 'Utility.csv' files are information of entire route set, generated from the 'Beijing Service point.csv', and 'Beijing Storage point.csv' in the 'Service station & warehouse data' folder. 

'Beta.csv' and 'eta.csv' are parameters. They are corresponding with the parameters in the paper.
'RouteSet.csv' is the whole route set, with task indices.
'Indicator.csv' contains the indicators that records which route each task in on.
'Distance.csv' records the original length and length after adding sensing tasks of each route. It is for GD, GE baselines. You may delete it, since they are in the file anymore.
'Size.csv' is the total DG number and route number of the route set.

The csv files are required to run the algorithms. You may generate new files according to your own need, but the data strutures have to remain the same, or you have to adjust the algorithms slightly based on the data struture changes.

If you need help in running the programs or any other issues, please contact me via my email 570486155@qq.com.

2. Operation guide
1) Use the 'datapreprocessing.py' to generate the above csv files.(With the given csv files, you can skip this setp)
2) Then run the 'RT-Drone.py' or 'RTW-Drone.py' files for RT-Drone algorithm or RTW-Drone algorithm. (The algorithms will filter a smaller route set based on your settings first. In this way, we can conduct repetitive experiments.)
3) The algorithms will generate csv files to store the output utility and selected routes.

3. Algorithm settings
The setting code is in the end of both '.py' files after ########main########.
The given algorithm files are for exploring the utility increase with varied budget.
You can easily change the settings to get the similar result in the paper. 
