1. Service stations
We collected the GPS data of 625 service station of 7 major express company in Beijing City.
In the paper, we only use the points inside the 4th ring of Beijing.

The data is stored as one csv file named 'Beijing Service point.csv', with 9 columns of information:
Type: 1 for warehouse, 2 for service station.
Number: label of the point in the express company, (The labels do not mean importance or anything else, but help us distinguish those GPS points.)
Company: corresponding express company of the service station.
Location ID: label of the point in all the data. (Similar to Number, the labels help us distinguish those points)
City: which city the service station is located at. 
Province: which city the service station is located at.
Country: which country the service station is located at.
Latitude & Longitude: GPS coordinates. (The data is from Baidu Map after encryption, whose coordinate system is BD09. 
If you want to show it in Google map, you need to transform the data into other coordinate system first.) 

We also separate the service station data into several csv files according to the express company name.

2. Warehouses
During data collection, we found out not all express companies own a warehouse in Beijing, or some companies own one warehouse there. 
Therefore, we only collected the warehouse information of JingDong, the largest express company in China.  
There are 15 JD warehouses in Beijing, which is a hugh surprise. The data is stored as  one csv file named 'Beijing Storage point.csv'. 
The data format is the similar to that of service stations.
