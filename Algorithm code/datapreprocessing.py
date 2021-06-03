#import json
import csv
import numpy as np
import math


N=100 #areas: N*N        
# create N*N blocks 
Long_boundary=np.linspace(116.262,116.492,N+1)  #boundary for the blocks
Lati_boundary=np.linspace(38.8233,39.9906,N+1)


def CoordinateRead(folder):#file1,file2):
    # read cooridinates of storage points
    filename2=r'\Beijing Service point.csv' #service station
    file2=folder+filename2
    # read cooridinates of service points
    Long2=[]
    Lati2=[]
    with open(file2) as csv_file: 
        reader=csv.reader(csv_file)
        for row in reader:
            Long2.append(row[8])
            Lati2.append(row[7])
    Long2=Long2[1:len(Long2)]
    Lati2=Lati2[1:len(Lati2)]
    Long2=list(map(float,Long2))
    Lati2=list(map(float,Lati2))
    
    # filer off the points out of the range
    Long2_flr=[]
    Lati2_flr=[]
    for m in range(len(Long2)):
        if Long2[m]<=Long_boundary[N] and Long2[m]>=Long_boundary[0]\
           and Lati2[m]<=Lati_boundary[N] and Lati2[m]>=Lati_boundary[0]:
            Long2_flr.append(Long2[m])
            Lati2_flr.append(Lati2[m])
            
    return [Long2_flr,Lati2_flr]




def GenerateWarehouse(Coordinates):
    Long2_flr=Coordinates[0]
    Lati2_flr=Coordinates[1]
    #randomly select 1/10 of the service stations as warehouses
    rdm_idx=np.random.randint(0,len(Long2_flr),round(len(Long2_flr)/10))
    WH_long=[]
    WH_lati=[]
    SP_long=[]
    SP_lati=[]
    for index in range(len(Long2_flr)):
        if index in rdm_idx:
            WH_long.append(Long2_flr[index])
            WH_lati.append(Lati2_flr[index])
        else:
            SP_long.append(Long2_flr[index])
            SP_lati.append(Lati2_flr[index])

    return [[WH_long,WH_lati],[SP_long,SP_lati]]   

def GenerateDG(Coordinates):
# detect trajectories
    Trajectories=[]
    #warehouses
    WH=Coordinates[0]
    WH_long=WH[0]
    WH_lati=WH[1]
    #service points
    SP=Coordinates[1]
    Long2_flr=SP[0]
    Lati2_flr=SP[1]
    

    for i in range(len(WH_long)): # warehouse
        Long_1=WH_long[i]
        Lati_1=WH_lati[i]
        '''LongNum_1=WH_LongNum[i]
        LatiNum_1=WH_LatiNum[i]'''
        
        for j in range(len(Long2_flr)): # service station
            path=[] # path of each trajectory. record passing coordinates
            
            Long_2=Long2_flr[j]
            Lati_2=Lati2_flr[j]
            '''LongNum_2=LongNum2[j]
            LatiNum_2=LatiNum2[j]'''
            Lati1=Lati_1*math.pi/180
            Lati2=Lati_2*math.pi/180
            Long1=Long_1*math.pi/180
            Long2=Long_2*math.pi/180
            d_Lati=Lati1-Lati2
            d_Long=Long1-Long2
            
            # haversine formula
            a = np.array(math.sin(d_Lati/2))*np.array(math.sin(d_Lati/2))\
                   + np.array(math.cos(Lati1))*np.array(math.cos(Lati2))\
                   * np.array(math.sin(d_Long/2))*np.array(math.sin(d_Long/2))
            c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
            distance=6371e3*c # in meter

            # filter trajectories with threshold 3000 meters
            if distance<=3000:

                path=[Long_1,Lati_1,Long_2,Lati_2,distance]
                # generate trajectories that records passing coordinates of each path
                Trajectories.append(path)
    Trajectories=np.array(Trajectories)
    return Trajectories


  
def RoutesWithTasks(Trajectories):
    Long_1=Trajectories[:,0]
    Lati_1=Trajectories[:,1]
    Long_2=Trajectories[:,2]
    Lati_2=Trajectories[:,3]
    distance1=Trajectories[:,4]
    
    Num=10000
    # generate tasks
    utility_per_time=2+(15-2)*np.random.rand(Num,1) #[4,10] profit/s
    utility_bound=1000+(3000-1000)*np.random.rand(Num,1) # [1,10] min
    task_distance=500+(1500-500)*np.random.rand(Num,1)
    
    # generate 10 routes for each DG
    RouteSet=[]
    task_idx=[]
    distance2=[]
    added=[]
    
    for i in range(len(Long_1)):
        DG=[]
        d0=GeoDistance(Long_1[i], Lati_1[i], Long_2[i], Lati_2[i])
        flag_stop=1
        distance1=[]
        for j in range(10):
            d1=d0
        # assign the tasks to trajectories
            tasks=[]
            d_threshold=np.random.randint(3000,6000,1)[0]
            for k in range(Num):
                if k not in added:
                    flag_stop=0
                    if d1+task_distance[k]<=d_threshold:
                        d1=d1+task_distance[k]
                        added.append(k)
                        tasks.append(k)
                        task_idx.append([i,j,k])
            if flag_stop==1:
                break
            if tasks!=[]:
                DG.append([[i,j],tasks,[d0,float(d1)]])   
                distance1.append([d0,float(d1)])
        if flag_stop==1:
            break
        RouteSet.append(DG)
        distance2.append(distance1)    

    # record trajectories and corresponding task utility
    return [RouteSet, distance2 , utility_per_time, utility_bound, task_idx]                
   
# calculte geographic distance of two GPS points by hersive formula       
def GeoDistance(Long1,Lati1,Long2,Lati2):
    Lati1=Lati1*math.pi/180
    Lati2=Lati2*math.pi/180
    Long1=Long1*math.pi/180
    Long2=Long2*math.pi/180
    d_Lati=Lati1-Lati2
    d_Long=Long1-Long2
    a = np.array(math.sin(d_Lati/2))*np.array(math.sin(d_Lati/2))\
       + np.array(math.cos(Lati1))*np.array(math.cos(Lati2))\
       * np.array(math.sin(d_Long/2))*np.array(math.sin(d_Long/2))
    c = 2*math.atan2(math.sqrt(a), math.sqrt(1-a))
    d_distance=6371e3*c # in meter
    return d_distance

def SaveTrajectoryData(Candidate_Trajectories,folder):
    Trajectory_task=Candidate_Trajectories[0]
    distance=Candidate_Trajectories[1]
    utility=[Candidate_Trajectories[2],Candidate_Trajectories[3]]
    idx=Candidate_Trajectories[4]
    Size=[len(Trajectory_task),10]
    
    for DG in Trajectory_task:
        for route in DG:
            while DG.count(route)>1:
                DG.remove(route)
    for DG in distance:
        for route in DG:
            while DG.count(route)>1:
                DG.remove(route)
    for task in idx:
        while idx.count(task)>1:
            idx.remove(task)
    
    beta= 4e4+ (7.5e4-4e4)*np.random.rand(Size[0],Size[1]) # [60,100] energy constraint
#print(beta)
    eta= np.round(3+(7-3)*np.random.rand(Size[0]))
    #beta1=[]
    '''for line in beta:
        line = list(line)
        beta1.append(line)'''
    beta=[list(line) for line in beta]    
    eta=list(eta)
    with open(folder+r'\RouteSet.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        for DG in Trajectory_task:
            wr.writerow(DG)
    with open(folder+r'\Distance.csv', 'w', newline='') as myfile:
        wr2 = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        for d in distance:
            wr2.writerow(d)
    with open(folder+r'\Utility.csv', 'w', newline='') as myfile:
        wr3 = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        for k in utility:
            wr3.writerow(k)
    with open(folder+r'\Indicator.csv', 'w', newline='') as myfile:
        wr4 = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        for row in idx:
            wr4.writerow(row)
    with open(folder+r'\Size.csv', 'w', newline='') as myfile:
        wr5 = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr5.writerow(Size)
    with open(folder+r'\Beta.csv', 'w', newline='') as myfile:
        wr6 = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        for row in beta:
            wr6.writerow(row)
    with open(folder+r'\eta.csv', 'w', newline='') as myfile:
        wr7 = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        #for row in beta:
        wr7.writerow(eta)


# read the file 'Beijing Service point.csv' from folder1
folder= r'D:\UAV-aided MCS\实验数据\仿真实验\1'
Coordinates = CoordinateRead(folder)     # read the file and save the coordinates
WH_and_SP  = GenerateWarehouse(Coordinates)  # divide warehouses and service points      
Trajectories = GenerateDG(WH_and_SP) # generate trajetories from one warehouse to on service point
CandidateRoutes = RoutesWithTasks(Trajectories) # add sensing tasks to the trajectories'''

# save info in folder2
folder2=r'D:\UAV-aided MCS\实验数据\仿真实验\载重'
SaveTrajectoryData(CandidateRoutes,folder2)


