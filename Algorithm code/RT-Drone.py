# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 15:33:55 2020

@author: 57048
"""
import csv
import numpy as np
import copy
import itertools
import random


def ReadData(folder):
    # read data
    RouteSet=[] # row: DG;     DG:multiple route with same start and end
    count=0
    with open(folder+r'\RouteSet.csv', newline='') as csvfile:
         spamreader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_ALL)
         for row in spamreader:
             row1=[]
             for line in row:             
                 line=eval(line)
                 row1.append(line)
             RouteSet.append(row1) 
    
    
    Distance=[] # row: [original distance, distance with tasks]; 
    with open(folder+r'\Distance.csv', newline='') as csvfile:
         spamreader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_ALL)
         for row in spamreader:
             row2=[]
             for line in row:
                 line=eval(line)
                 row2.append(line)
             Distance.append(row2) 
    
    Utility=[] # row: [original distance, distance with tasks]; 
    with open(folder+r'\Utility.csv', newline='') as csvfile:
         spamreader = csv.reader(csvfile,  delimiter=',',quoting=csv.QUOTE_ALL)
         for row in spamreader:
             row3=[]
             for line in row:
                 line=eval(line)[0]
                 row3.append(line)
             Utility.append(row3) 
     
    Indicator=[] # row: [original distance, distance with tasks]; 
    with open(folder+r'\Indicator.csv', newline='') as csvfile:
         spamreader = csv.reader(csvfile,  delimiter=',',quoting=csv.QUOTE_ALL)
         for row in spamreader:
             row4=[]
             for line in row:
                 line=eval(line)
                 row4.append(line)
             Indicator.append(row4)
    
    Size=[]         # size of the route set
    with open(folder+r'\Size.csv', newline='') as csvfile:
         spamreader = csv.reader(csvfile,  delimiter=',',quoting=csv.QUOTE_ALL)
         for row in spamreader:
             row5=[]
             for line in row:
                 line=eval(line)
                 row5.append(line)
             Size.append(row5) 
             
    beta=[]       # parameter
    with open(folder+r'\beta.csv', newline='') as csvfile:
         spamreader = csv.reader(csvfile,  delimiter=',',quoting=csv.QUOTE_ALL)
         for row in spamreader:
             row6=[]
             for line in row:
                 #print(line)
                 line=eval(line)
                 row6.append(line)
                 
             beta.append(row6)
             
    eta=[]      # parameter
    with open(folder+r'\eta.csv', newline='') as csvfile:
         spamreader = csv.reader(csvfile,  delimiter=',',quoting=csv.QUOTE_ALL)
         for row in spamreader:
             row7=[]
             for line in row:
                 line=eval(line)
                 row7.append(line)
             eta.append(row7)         
    return [RouteSet,Distance,Utility,Indicator,Size,beta,eta]

# filter a certain size of route set for the algorithm based on the given DG number and route number
def ExtractCandidateRoute_route(AllData,DG_num,route_num):
    # list can be changed inside functions.
    # copy the input lists to avoid changes.
    RouteSet=AllData[0]
    distance=AllData[1]
    utility_per_time=AllData[2][0]
    utility_bound=AllData[2][1]
    
    idx=AllData[3]
    # randomly select route of a certain size from a originally large route set 
    # to conduct the controlled experiments
    Selected_DG=list(np.random.randint(0,len(RouteSet),DG_num)) #i
    RouteSet_candidate=[]
    Indicator=[]
    Distance=[]
    for DG_idx in Selected_DG:
        Candidate_DG=RouteSet[DG_idx]
        DG_candidate=[]
        Distance1=[]
        # randomly select route of a certain size from a originally large route set 
        # to conduct the controlled experiments
        Selected_route=list(np.random.randint(0,len(Candidate_DG),route_num)) #j
        for route_idx in Selected_route: 
            DG_candidate.append(Candidate_DG[route_idx])
            Distance1.append(distance[DG_idx][route_idx])
            for index in idx:
                if (index[0]==DG_idx) & (index[1]==route_idx):
                    Indicator.append(index)
        if DG_candidate!=[]:
            RouteSet_candidate.append(DG_candidate)
            Distance.append(Distance1)
    uk_selected=[]
    Uk_selected=[]
    task_no=[]
    for task in Indicator:
        task_no.append(task[2])  # indicate task serial number
        uk_selected.append(utility_per_time[task[2]])
        Uk_selected.append(utility_bound[task[2]])
    
    return [RouteSet_candidate,Distance,[uk_selected,Uk_selected,task_no],\
            Indicator,AllData[4],AllData[5],AllData[6]]

# filter a certain size of route set for the algorithm based on the given task number
# this one is not shown in the result of the paper due to page limit
def ExtractCandidateRoute_task(AllData, task_num):
    RouteSet = AllData[0]
    idx = AllData[3]
    distance = AllData[1]
    utility_per_time = AllData[2][0]
    utility_bound = AllData[2][1]
    count_task = []
    for DG in RouteSet:
        count_task_DG = []
        for route in DG:
            count_task_DG.append(len(route[1]))  # task number on a route
        count_task.append(count_task_DG)  # task number on each route of a DG

    task_count = 0
    # if the tasks on candidate routes is less than the task number threshold
    RouteSet_candidate = []
    added = []
    Indicator = []
    while task_count <= task_num:
        i = np.random.randint(0, len(RouteSet), 1)[0]
        j = np.random.randint(0, 10, 1)[0]
        # if the tasks of a route are selected, add the route into candidate set
        if [i, j] not in added:
            task_count = task_count + count_task[i][j]
            RouteSet_candidate.append(RouteSet[i][j])  # candidate set
            added.append([i, j])
    for route in RouteSet_candidate:
        for index in idx:

            if index[0:-1] == route[0]:
                Indicator.append(index)
    # sort the candidate set
    DG_sorted = []
    Distance = []
    for i in range(len(RouteSet)):
        route_sorted = []
        Distance1 = []
        for j in range(10):
            for route in RouteSet_candidate:
                if route[0] == [i, j]:
                    route_sorted.append(route)
                    Distance1.append(distance[i][j])

        if Distance1 != []:
            Distance.append(Distance1)
            DG_sorted.append(route_sorted)
    uk_selected = []
    Uk_selected = []
    task_no = []
    for task in Indicator:
        task_no.append(task[2])  # indicate task serial number
        uk_selected.append(utility_per_time[task[2]])
        Uk_selected.append(utility_bound[task[2]])

    return [DG_sorted, Distance, [uk_selected, Uk_selected, task_no], Indicator, AllData[4], AllData[5], AllData[6]]

# the greedy time allocation strategy (Algorithm 1)
def TimeAllocation(Selected_routes):
    ################## calculate time constraints ##########################
    # find the index of selected routes
    if Selected_routes == None:
        return [[], 0, 0]
    else:
        Tij = np.zeros([Size[0][0], Size[0][1]])
        d = 0
        cost_flight = 0
        total_weight = 0
        task_on_routes = []
        for route in Selected_routes:  # selected i j pair
            i = route[0]
            j = route[1]
            
            # if it is a feasible set
            # update constraints
            for m in range(len(RouteSet)):
                if RouteSet[m][0][0][0] == i:
                    for n in range(len(RouteSet[m])):
                        if RouteSet[m][n][0][1] == j:
                            delta_d = Distance[m][n][1] - Distance[m][n][0]
                            T1 = beta[i][j] - Pf[i][j] * Distance[m][n][1] / v
                            Tij[i][j] = T1
                            d = d + delta_d

                            cost_flight = cost_flight + Pf[i][j] * delta_d / v
                            total_weight = total_weight + Weight[i][j]
                            # record task index on the selected routes
                            task_on_routes.extend(RouteSet[m][n][1])

        T = B / lamb - cost_flight

        if T < 0:
            # print('T<0')
            return [[], 0, 0]
        for i in range(Size[0][0]):
            for j in range(Size[0][1]):
                if Tij[i][j] < 0:
                    # print('Tij<0')
                    return [[], 0, 0]
        ################### the greedy time allocation #####################
        # sort tasks based on utility
        utility_per_time = Utility[0]
        utility_bound = Utility[1]
        task_no = Utility[2]

        uk = []
        Uk = []
        task_no_selected = []
        # filter the tasks not on selected routes to significantly reduce computation complexity
        for n in range(len(task_no)):
            if task_no[n] in task_on_routes:
                task_no_selected.append(task_no[n])
                uk.append(utility_per_time[n])
                Uk.append(utility_bound[n])
        # sort the tasks according to utility rate        
        sort_index = np.argsort(uk)  # sort in increasing order
        sort_index = sort_index[::-1]  # sort in decreasing order

        I = []
        J = []
        u1 = []
        U1 = []
        # record the sort order of each task
        for index in sort_index:  # index = k th task
            for task in Indicator:  # rijk
                k = task[2]
                if k == task_no_selected[index]:
                    i = task[0]
                    j = task[1]
                    I.append(i)
                    J.append(j)
                    u1.append([uk[index], k])
                    U1.append([Uk[index], k])
        # allocate time in the decreasing order of utility rate
        t = []
        for n in range(len(u1)):
            k = u1[n][1]  # k
            i = I[n]
            j = J[n]

            tk = 0

            if T > 0:
                if Tij[i][j] > 0:
                    tk = min(T / Ph[i][j], Tij[i][j] / Ph[i][j], U1[n][0] / u1[n][0])
                else:

                    tk = 0
                t.append([tk, i, j, k])
                T = T - Ph[i][j] * tk
                Tij[i][j] = Tij[i][j] - Ph[i][j] * tk
            else:
                break

        gain = 0
        # cost=0
        for n in range(len(t)):
            gain = gain + u1[n][0] * t[n][0]
            # cost=cost+Ph*t[n][0]
        gain = gain + total_weight
        # print(t)
        # print(gain)
        return [t, gain, cost_flight, d]




def RouteSchedule(RouteSet,Distance,Utility,Indicator):
    
    added=[] # local optimal solutions
    Local_optimal_gain=np.zeros(q+1)
    X=[]
    for path in Indicator:
        #for path in DG:
        X.append(path[0:2])# candidate routes
    for path in X:
        while X.count(path)>1:
            X.remove(path)  # remove repeated 
    for s in range(q+1): # find 3 local optimal solutions q+1
        # initial feasible solution
        # find the shortest route
        gain_flag=0
        '''
        for i in range(len(RouteSet)):
            path=RouteSet[i]
            for j in range(len(path)):
                S0=[[i,path[0][-1]]] # test gain of a route set
                gain0=TimeAllocation(S0)[1]
                               
                if gain0>gain_flag:
                    gain_flag=gain0  # initial solution set
                    Snew=[[i,j]]  # route with largest gain
        '''
        flag_replace=0
        for path in X:
            S0=[path]
            gain0=TimeAllocation(S0)[1]
            if gain0>gain_flag:
                Snew=[path] # route with largest gain
                gain_flag=gain0  # initial solution set
                flag_replace=1
                
        # start iteration   
        if  flag_replace ==1:
            S0=Snew
            gain0=gain_flag 
        else:
            break
        flag=1
        while flag==1:
            flag_neighbor=1
            flag=0
            flag_exchange=1
            
            # delete operation
            for e in S0:
                Snew=copy.deepcopy(S0)
                Snew.remove(e)
                gain1=TimeAllocation(Snew)[1] # gain of the neighbor
                if gain1>(1+epsilon/(size**4))*gain0:
                    S0=copy.deepcopy(Snew)
                    flag_exchange=0# if find a solution by deletion, do not proceed exchange
                    gain0=gain1
                    flag=1   # if find a solution by deletion, go to next iteration
                    break

            if flag_exchange==0:
                s0 = copy.deepcopy(S0)
            
            elif flag_exchange==1:
                # exchange operation:
                S1=copy.deepcopy(S0)
                for m in range(q): #q -> k=2
                    S1.append([])   
                E=list(itertools.combinations(S1,q))        
                S2=Diff(X,S0)
                D=list(itertools.combinations(S2,1))
                neighborhood=[] # neighborhood
                for e in E:
                    e=list(e)
                    for d in D:
                        d=list(d)
                        if Diff(S0,e)==[]:
                            S_e=[Diff(S0,e)]
                        else:
                            S_e=Diff(S0,e)
                        d.extend(S_e) # potential solution
                        Snew=d
        
                        while [] in Snew:
                            Snew.remove([])
                            
                        if Snew==[]:    
                            continue
                        Snew.sort()
                        neighborhood.append(Snew) # find the neighborhood
                for solution in neighborhood:
                    if neighborhood.count(solution)>1:
                        neighborhood.remove(solution) # remove repeated solutions

                #print(neighborhood)
                
                for Snew in neighborhood: 
                    if flag_neighbor==0:
                        break
                    elif flag_neighbor==1:
                        count_i=np.zeros(Size[0][0])
                        #count_j=np.zeros(Size[0][1])
                        flag21=1
                        #flag22=1
                        for path in Snew:
                            i=path[0]
                            #j=path[1]
                            count_i[i]=count_i[i]+1 
                           #count_j[j]=count_j[j]+1
                        flag11=eta[0]-count_i   # only eta (i)  paths for ith DG
                        for k in flag11: # if it fits constraint (14)
                            if k <0:
                                flag21=0 # 0 for not
                        #for k in count_j:
                           # if k>1:
                               # flag22=0 # 0 for not
                        if flag21==1:# if it is feasible
                            gain1=TimeAllocation(Snew)[1] # gain of the neighbor
                            if gain1>=(1+epsilon/(size**4))*gain0: # if the neighbor is better
                                S0=copy.deepcopy(Snew)
                                S0.sort()
                                gain0=gain1
                                flag=1          # continue searching for next neighbor after replacing solution
                                #print('New Solution',S0)
                                #print('New Gain',gain0)
                                s0 = copy.deepcopy(S0)
                                flag_neighbor=0
                                break
                if flag_neighbor==1:  # if no solution in neighborhood fits the condition
                    s0=copy.deepcopy(S0)
                    break
            
        #try:
        s0.sort()
        added.append(s0)
        print('Solution',s,':',s0)
        print('Gain',s,':',gain0)
        
        Local_optimal_gain[s]=gain0 
        for path in s0:
            X.remove(path) # remove the local optimal
    #print(added)    
    optimal=max(Local_optimal_gain) 
    idx=list(np.where(Local_optimal_gain==optimal))[0][0]
    Result=added[idx]
    cost0=TimeAllocation(Result)[2]
    print('1: Final Solution:',Result)
    print('1: Final Gain:',optimal)
    print('1: Final Cost:',cost0)     
    return [Result,optimal,cost0]

def Diff(li1, li2):
    li_dif = [i for i in li1 + li2 if i in li1 and i not in li2] 
    return li_dif

def AllSubset(list1):
    s = list(list1)
    subsets=[]
    for n in range(len(s)+1):
        subset=list(itertools.combinations(s, n))
        for element in subset:
            element1=list(element)
            subsets.append(element1)                 
    return subsets

################################### main ############################################
# parameters
q=2 # matroid constraints k
p=1 # p
v=5 # m/s
lamb= 0.5/3.6  # cost per Joul
#B=50000 # budget
epsilon=1         #epsilon

# folder where the data of original route set are stored
# and where the result of the algorithm will be saved
folder=r'D:\ZhouYanlin\仿真实验\模拟退火'
# budge range from 150000 to 200000
Energycost=np.linspace(150000,200000,6)
Budget=Energycost*lamb
# experiment
for budget_idx in range(len(Budget)):
    B=int(Budget[budget_idx])
    Result1=[]
    Result2=[]
    Result3=[]
    Result4=[]
    Result5=[]
    Result6=[]
    
    file=folder+'\B='+str(B)+'.csv'
    for time in range(5):
        # DG_num controls the DG number 
        # Route_num controls the Route number
        # They cannot exceed the size of original route set
        DG_num=50
        Route_num=4
        
        # task_num = 500
        # task_num controls the task number of selected route set
        # it can not exceed the total number of tasks in the original route set
        
        AllData=ReadData(folder)
        #TestData=ExtractCandidateRoute_task(AllData,task_num)
        TestData=ExtractCandidateRoute_route(AllData,DG_num,Route_num)
        RouteSet=TestData[0]
        Distance=TestData[1]
        Utility=TestData[2]
        Indicator=TestData[3]
        Size=TestData[4]
        beta=TestData[5]
        eta=TestData[6]
        
        Weight=50+(600-50)*np.random.rand(Size[0][0],Size[0][1])  # g
        Ph=0.0256*Weight+26.6172  # hovering power 
        Pf=0.0256*Weight+25.7519  # flight power at 5m/s
        route_num=Size[0][0]*Size[0][1] 
        size=route_num         #number
        
        Solution1=RouteSchedule(RouteSet,Distance,Utility,Indicator)
        #Solution2=Greedy_utility_rate(RouteSet,Distance,Utility,Indicator)
        #Solution3=Greedy_utility(RouteSet,Distance,Utility,Indicator) 
        #Solution4=Greedy_distance(RouteSet,Distance,Utility,Indicator) 
        #Solution5=SimulatedAnnealing(RouteSet, Distance, Utility, Indicator)
        #Solution5=EvenTimeSchedule(RouteSet,Distance,Utility,Indicator)   
        #Solution6=RandomRoute(RouteSet,Distance,Utility,Indicator) 
            #Solution5=BrutalForce(RouteSet,Distance,Utility,Indicator)   
        
        Result1.append(Solution1[1])
        #Result2.append(Solution2[1])
        #Result3.append(Solution3[1])
        #Result4.append(Solution4[1])
        #Result5.append(Solution5[1])
        #Result6.append(Solution6[1])

    title = 'B='+str(B)
    with open(file, 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    
        #wr.writerow(title)
        wr.writerow(['Proposed',B,Result1])
        #wr.writerow(['Greedy cost efficiency',B,Result2])
        #wr.writerow(['Greedy utility',B,Result3])
        #wr.writerow(['Greedy distance',B,Result4])
        #wr.writerow(['Simulated Annealing',B,Result5])
        #wr.writerow(['Random route',B,Result6])
        

        
