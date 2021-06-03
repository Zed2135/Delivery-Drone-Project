# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 15:33:55 2020

@author: 57048
"""
import csv
import numpy as np
import copy
import math
import itertools
import random


def ReadData(folder):
    # read data
    RouteSet = []  # row: DG;     DG:multiple route with same start and end
    count = 0
    with open(folder + r'\RouteSet.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_ALL)
        for row in spamreader:
            row1 = []
            for line in row:
                line = eval(line)
                row1.append(line)
            RouteSet.append(row1)

    Distance = []  # row: [original distance, distance with tasks];
    with open(folder + r'\Distance.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_ALL)
        for row in spamreader:
            row2 = []
            for line in row:
                line = eval(line)
                row2.append(line)
            Distance.append(row2)

    Utility = []  # row: [original distance, distance with tasks];
    with open(folder + r'\Utility.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_ALL)
        for row in spamreader:
            row3 = []
            for line in row:
                line = eval(line)[0]
                row3.append(line)
            Utility.append(row3)

    Indicator = []  # row: [original distance, distance with tasks]
    with open(folder + r'\Indicator.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_ALL)
        for row in spamreader:
            row4 = []
            for line in row:
                line = eval(line)
                row4.append(line)
            Indicator.append(row4)

    Size = []
    with open(folder + r'\Size.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_ALL)
        for row in spamreader:
            row5 = []
            for line in row:
                line = eval(line)
                row5.append(line)
            Size.append(row5)
            
    beta = []
    with open(folder + r'\beta.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_ALL)
        for row in spamreader:
            row6 = []
            for line in row:
                # print(line)
                line = eval(line)
                row6.append(line)

            beta.append(row6)
    eta = []
    with open(folder + r'\eta.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',', quoting=csv.QUOTE_ALL)
        for row in spamreader:
            row7 = []
            for line in row:
                line = eval(line)
                row7.append(line)
            eta.append(row7)
    return [RouteSet, Distance, Utility, Indicator, Size, beta, eta]


def ExtractCandidateRoute_route(AllData, DG_num, route_num):
    RouteSet = AllData[0]
    distance = AllData[1]
    utility_per_time = AllData[2][0]
    utility_bound = AllData[2][1]

    idx = AllData[3]
    Selected_DG = list(random.sample(range(len(RouteSet)), DG_num))  # i
    Selected_DG.sort()

    RouteSet_candidate = []
    Indicator = []
    Distance = []

    for DG_idx in Selected_DG:
        Candidate_DG = RouteSet[DG_idx]
        DG_candidate = []
        Distance1 = []
        Selected_route = list(random.sample(range(len(Candidate_DG)), route_num))  # j
        Selected_route.sort()
        for route_idx in Selected_route:
            route_candidate = Candidate_DG[route_idx]
            DG_candidate.append(route_candidate)
            Distance1.append(distance[DG_idx][route_idx])
            for index in idx:
                # for index in DG:
                if index[0:2] == route_candidate[0]:
                    Indicator.append(index)
        if DG_candidate != []:
            RouteSet_candidate.append(DG_candidate)
            Distance.append(Distance1)

    for DG in RouteSet_candidate:
        for path in DG:
            while DG.count(path) > 1:
                DG.remove(path)
    for DG in RouteSet_candidate:
        if DG == []:
            RouteSet_candidate.remove(DG)
    for DG in Distance:
        for path in DG:
            while DG.count(path) > 1:
                DG.remove(path)
    for DG in Distance:
        if DG == []:
            Distance.remove(DG)

    uk_selected = []
    Uk_selected = []
    task_no = []
    for task in Indicator:
        task_no.append(task[2])  # indicate task serial number
        uk_selected.append(utility_per_time[task[2]])
        Uk_selected.append(utility_bound[task[2]])

    return [RouteSet_candidate, Distance, [uk_selected, Uk_selected, task_no], \
            Indicator, AllData[4], AllData[5], AllData[6]]


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
                    route_sorted.append(route[0])
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

    return [RouteSet_candidate, Distance, [uk_selected, Uk_selected, task_no], Indicator, AllData[4], AllData[5],
            AllData[6]]


def TimeAllocation(Selected_routes, weight):
    Ph = 0.0256 * weight + 26.6172  # hovering power
    Pf = 0.0256 * weight + 25.7519  # flight power at 5m/s
    # calculate time constraints
    # find the index of selected routes
    if Selected_routes == None:
        return [[], 0, 0, 0, 0, []]
    else:
        Tij = np.zeros([Size[0][0], Size[0][1]])
        d = 0
        cost_flight = 0
        total_weight = 0
        for route in Selected_routes:  # selected i j pair
            i = route[0]
            j = route[1]
            # try:
            # if it is a feasible set
            for m in range(len(RouteSet)):
                if RouteSet[m][0][0][0] == i:
                    for n in range(len(RouteSet[m])):
                        if RouteSet[m][n][0][1] == j:
                            delta_d = Distance[m][n][1] - Distance[m][n][0]
                            T1 = beta[i][j] - Pf[i][j] * Distance[m][n][1] / v
                            Tij[i][j] = T1
                            d = d + delta_d
                            cost_flight = cost_flight + Pf[i][j] * delta_d / v
                            total_weight = total_weight + weight[i][j]
        T = B / lamb - cost_flight

        if T < 0:
            return [[], 0, 0, 0, 0, []]
        for i in range(Size[0][0]):
            for j in range(Size[0][1]):
                if Tij[i][j] < 0:
                    return [[], 0, 0, 0, 0, []]

        # sort tasks based on utility
        utility_per_time = Utility[0]
        utility_bound = Utility[1]
        task_no = Utility[2]
        sort_index = np.argsort(utility_per_time)  # sort in increasing order
        sort_index = sort_index[::-1]  # sort in decreasing order

        I = []
        J = []
        u1 = []
        U1 = []
        # sort tasks
        for index in sort_index:  # index = k th task
            for task in Indicator:  # rijk
                i = task[0]
                j = task[1]
                k = task[2]
                if k == task_no[index]:
                    I.append(i)
                    J.append(j)
                    u1.append([utility_per_time[index], k])
                    U1.append([utility_bound[index], k])
        # print(u1)
        t = []
        for n in range(len(u1)):
            k = u1[n][1]  # k
            i = I[n]
            j = J[n]
            # for pair in Indicator:
            tk = 0
            # sum_t=0
            if T > 0:
                if Tij[i][j] > 0:
                    tk = min(T / Ph[i][j], Tij[i][j] / Ph[i][j], U1[n][0] / u1[n][0])
                else:
                    tk = 0
                t.append([tk, i, j, k])
                T = T - Ph[i][j] * tk
                Tij[i][j] = Tij[i][j] - Ph[i][j] * tk
                # sum_t=sum_t+tk
            else:
                break
        t_sum = []
        for time in t:
            while t.count(time) > 1:
                t.remove(time)

        for i in range(Size[0][0]):
            for j in range(Size[0][1]):
                sum_t = 0
                for time in t:
                    if [i, j] == time[1:3]:
                        sum_t = sum_t + time[0]
                if sum_t != 0:
                    t_sum.append([sum_t, time[1], time[2]])

        gain = 0
        # cost=0
        for n in range(len(t)):
            gain = gain + u1[n][0] * t[n][0]
            # cost=cost+Ph*t[n][0]
        gain1 = gain + total_weight * 0.5
        # print(t)
        # print(gain)
        return [t, gain1, cost_flight, d, total_weight, t_sum]


def Diff(li1, li2):
    li_dif = [i for i in li1 + li2 if i in li1 and i not in li2]
    return li_dif


def RouteSchedule(RouteSet, Distance, Utility, Indicator, S0, weight):
    S_origin = copy.deepcopy(S0)
    #gain_origin = copy.deepcopy(gain0)
    added = []  # local optimal solutions
    Local_optimal_gain = np.zeros(q + 1)
    X = []
    # print('Indicator',Indicator)
    for path in Indicator:
        # for path in DG:
        X.append(path[0:2])  # candidate routes
    for path in X:
        # print(path)
        while X.count(path) > 1:
            X.remove(path)  # remove repeated
    # print('X',X)
    gain0 = TimeAllocation(S0, weight)[1]
    if gain0 <= 0:
        return [[], 0, 0, [], 0]

    for s in range(q + 1):  # find 3 local optimal solutions q+1
        S0 = copy.deepcopy(S_origin)
        gain0 = TimeAllocation(S0,weight)[1]
        # initial feasible solution
        # find the shortest route
        # gain_flag=0
        '''
        for i in range(len(RouteSet)):
            path=RouteSet[i]
            for j in range(len(path)):
                S0=[[i,path[0][-1]]] # test gain of a route set
                gain0=TimeAllocation(S0)[1]

                if gain0>gain_flag:
                    gain_flag=gain0  # initial solution set
                    Snew=[[i,j]]  # route with largest gain

        #print('X',X)
        flag_replace=0
        for path in X:
            S0=[path]

            #print(gain0)
            if gain0>gain_flag:
                Snew=[path] # route with largest gain
                gain_flag=gain0  # initial solution set
                flag_replace=1



        #print('Initial',gain_flag)
        # start iteration   
        if  flag_replace ==1:
            S0=Snew
            gain0=gain_flag 
            #print(S0)
        else:
            break
        '''
        flag = 1
        while flag == 1:
            # print('New Solution',S0)
            flag_neighbor = 1
            flag = 0
            flag_exchange = 1

            # delete operation
            for e in S0:
                Snew = copy.deepcopy(S0)
                Snew.remove(e)
                gain1 = TimeAllocation(Snew, weight)[1]  # gain of the neighbor
                if gain1 > (1 + epsilon / (size ** 4)) * gain0:
                    S0 = copy.deepcopy(Snew)
                    flag_exchange = 0  # if find a solution by deletion, do not proceed exchange
                    gain0 = copy.deepcopy(gain1)
                    flag = 1  # if find a solution by deletion, go to next iteration
                    break

            if flag_exchange == 0:
                s0 = copy.deepcopy(S0)

            elif flag_exchange == 1:
                # exchange operation:
                S1 = copy.deepcopy(S0)
                for m in range(q):  # q -> k=2
                    S1.append([])
                E = list(itertools.combinations(S1, q))
                S2 = Diff(X, S0)
                D = list(itertools.combinations(S2, 1))
                neighborhood = []  # neighborhood
                for e in E:
                    e = list(e)
                    for d in D:
                        d = list(d)
                        if Diff(S0, e) == []:
                            S_e = [Diff(S0, e)]
                        else:
                            S_e = Diff(S0, e)
                        d.extend(S_e)  # potential solution
                        Snew = d

                        while [] in Snew:
                            Snew.remove([])
                        if Snew == []:
                            continue
                        Snew.sort()
                        neighborhood.append(Snew)  # find the neighborhood
                for solution in neighborhood:
                    if neighborhood.count(solution) > 1:
                        neighborhood.remove(solution)  # remove repeated solutions

                for Snew in neighborhood:
                    if flag_neighbor == 0:
                        break
                    elif flag_neighbor == 1:
                        count_i = np.zeros(Size[0][0])
                        count_j = np.zeros(Size[0][1])

                        # if Snew not in added:
                        flag21 = 1
                        flag22 = 1
                        for path in Snew:
                            i = path[0]
                            j = path[1]
                            count_i[i] = count_i[i] + 1
                            count_j[j] = count_j[j] + 1
                        flag11 = eta[0] - count_i  # only eta (i)  paths for ith DG
                        for k in flag11:  # if it fits constraint (14)
                            if k < 0:
                                flag21 = 0  # 0 for not
                        for k in count_j:
                            if k > 1:
                                flag22 = 0  # 0 for not
                        if (flag21 == 1) & (flag22 == 1):  # if it is feasible
                            gain1 = TimeAllocation(Snew, weight)[1]  # gain of the neighbor
                            if gain1 >= (1 + epsilon / (size ** 4)) * gain0:  # if the neighbor is better
                                S0 = copy.deepcopy(Snew)
                                S0.sort()
                                gain0 = copy.deepcopy(gain1)
                                # flag_neighbor=0 # stop search other neighbor
                                # print('Neighbor flag',flag_neighbor)
                                flag = 1  # continue searching for next neighbor after replacing solution
                                print('New Solution', S0)
                                print('New Gain', gain0)
                                s0 = copy.deepcopy(S0)
                                flag_neighbor = 0
                                break
                if flag_neighbor == 1:  # if no solution in neighborhood fits the condition
                    s0 = copy.deepcopy(S0)
                    break

        # try:
        s0.sort()
        flag_add = 1
        for path in s0:
            if path not in X:
                flag_add = 0
            if flag_add == 1:
                added.append(s0)
                Local_optimal_gain[s] = gain0
                X.remove(path)  # remove the local optimal
    optimal = max(Local_optimal_gain)
    idx = list(np.where(Local_optimal_gain == optimal))[0][0]
    Result = added[idx]
    cost0 = TimeAllocation(Result, weight)[2]
    t0 = TimeAllocation(Result, weight)[5]
    weight0 = TimeAllocation(Result, weight)[4]
    print('1: Final Solution:', Result)
    print('1: Final Gain:', optimal)
    return [Result, optimal, cost0, t0, weight0]


def WeightAssign(RouteSet, Distance, Utility, Indicator, Weight):
    Weight_solution = []  # record solution in each iteration
    weight = copy.deepcopy(Weight)  # initialize weight
    weight_bound = 600 * np.zeros([Size[0][0], Size[0][1]])  # initialize weight bound
    # Ph=0.0256*weight+26.6172  # hovering power
    # Pf=0.0256*weight+25.7519  # flight power at 5m/s
    flag_weight = 1
    X = []
    for path in Indicator:
        # for path in DG:
        X.append(path[0:2])  # candidate routes
    for path in X:
        while X.count(path) > 1:
            X.remove(path)  # remove repeated
    gain_flag = 0
    flag_replace = 0
    for path in X:
        S0 = [path]
        gain0 = TimeAllocation(S0, weight)[1]
        if gain0 > gain_flag:
            Snew = [path]  # route with largest gain
            gain_flag = gain0  # initial solution set
            flag_replace = 1
    if flag_replace == 0:
        return [[], 0, 0, 0]  # if find no initial feasible set, return nothing
    else:
        S0 = copy.deepcopy(Snew)

    Solution0 = TimeAllocation(S0, weight)
    gain0 = Solution0[1]
    print('Original:', S0, gain0)
    cost0 = Solution0[2]
    t0 = Solution0[5]
    for time in t0:
        while t0.count(time) > 1:
            t0.remove(time)
    Weight_solution.append(['Original', 0, [], t0, S0, gain0, cost0])

    Solution1 = RouteSchedule(RouteSet, Distance, Utility, Indicator, S0, weight)  # initial solution
    S0 = copy.deepcopy(Solution1[0])
    gain0 = copy.deepcopy(Solution1[1])
    cost0 = copy.deepcopy(Solution1[2])
    t0 = copy.deepcopy(Solution1[3])
    weight0 = copy.deepcopy(Solution1[4])
    for time in t0:
        while t0.count(time) > 1:
            t0.remove(time)
    print('First iteration:', S0, gain0)
    Weight_solution.append(['First solution', 0, weight0, t0, S0, gain0, cost0])
    Weight_set = np.linspace(0, 600, 121)
    flag_weight = 1

    while flag_weight == 1:
        weight_selected = []
        weight_idx = []
        time_coefficient = []
        allocated_time = []
        flag_weight = 0
        # Ph=0.0256*weight+26.6172  # hovering power
        # Pf=0.0256*weight+25.7519  # flight power at 5m/s
        weight_total = TimeAllocation(S0, weight)[4]
        for route in S0:
            i = route[0]
            j = route[1]

            for m in range(len(RouteSet)):
                for n in range(len(RouteSet[m])):
                    if (RouteSet[m][n][0][0] == i) & (RouteSet[m][n][0][1] == j):
                        t = TimeAllocation(S0, weight)[0]
                        for time in t:
                            while t.count(time) > 1:
                                t.remove(time)
                        sum_t = 0
                        for time in t:
                            if (time[1] == i) & (time[2] == j):
                                sum_t = sum_t + time[0]

                        weight_bound[i][j] = min(900, (beta[i][j] * v - 25.7519 * Distance[m][n][1] - \
                                                       26.6172 * v * sum_t) / (
                                                         0.0256 * v * sum_t + 0.0256 * Distance[m][n][1]))
                        time1 = 0.0256 * sum_t + 0.0256 * (Distance[m][n][1] - Distance[m][n][0]) / v
                        weight_selected.append(weight_bound[i][j])
                        weight_idx.append([i, j])
                        time_coefficient.append(time1)
                        allocated_time.append([sum_t, i, j])
                    # weight_coefficient=
        for time in allocated_time:
            while allocated_time.count(time) > 1:
                allocated_time.remove(time)
        time_sort = np.argsort(time_coefficient)  # [::-1]  # increasing order
        Delta = B / lamb
        print(weight_selected, time_coefficient)
        for index in time_sort:
            i = weight_idx[index][0]
            j = weight_idx[index][1]  # find weight in the increasing order of its time coefficient
            for m in range(len(RouteSet)):
                if RouteSet[m][0][0][0] == i:
                    for n in range(len(RouteSet[m])):
                        if RouteSet[m][n][0][1] == j:
                            if Delta <= 0:
                                break
                            else:
                                sum_t = 0
                                for time in t:
                                    if (time[1] == i) & (time[2] == j):
                                        sum_t = sum_t + time[0]
                                del_d = Distance[m][n][1] - Distance[m][n][0]
                                delta_threshold = (Delta - (26.6172 * sum_t + 25.7519 * del_d / v)) / (
                                            0.0256 * sum_t + 0.0256 * del_d / v)
                                if delta_threshold >= 5:
                                    weight1 = math.floor(min(weight_selected[index], delta_threshold))
                                    weight_selected[index] = copy.deepcopy(weight1 - (weight1 % 5))  # discretization                                    
                                    print(weight_selected[index])
                                    Delta = Delta - (
                                            weight_selected[index] * (0.0256 * sum_t + 0.0256 * del_d / v) + \
                                            26.6172 * sum_t + 25.7519 * del_d / v)
                                    weight[i][j] = copy.deepcopy(weight_selected[index])
                                else:
                                    weight_selected[index] = copy.deepcopy(0)
                                    weight[i][j] = copy.deepcopy(0)
                if Delta <= 0:
                    break
            if Delta <= 0:
                break
        gain = gain0 + (sum(weight_selected) - weight_total) * 0.5
        cost = cost0

        Weight_solution.append(['New weight', 1, weight_selected, t0, S0, gain, cost])
        print('Update weight:', S0, gain)

        Solution1 = RouteSchedule(RouteSet, Distance, Utility, Indicator, S0, weight)  # new solution
        S1 = copy.deepcopy(Solution1[0])
        gain1 = copy.deepcopy(Solution1[1])
        cost1 = copy.deepcopy(Solution1[2])
        t1 = copy.deepcopy(Solution1[3])
        for time in t1:
            while t1.count(time) > 1:
                t1.remove(time)
        if gain1 > (1 + 0.001) * gain0:  # epsilon=0.1
            flag_weight = 1  # if the new solution increases utility
            S0 = copy.deepcopy(S1)
            gain0 = copy.deepcopy(gain1)
            cost0 = copy.deepcopy(cost1)
            t0 = copy.deepcopy(t1)
            Weight_solution.append(['New solution', 2, weight_selected, t0, S0, gain0, cost0])
            print('Update route:', S0, gain0)
        else:
            flag_weight = 0
            Weight_solution.append(['End', 3, weight_selected, t1, S1, gain1, cost1])

    with open(folder + r'\WeightAssign' + str(int(B)) + '.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        for row in Weight_solution:
            wr.writerow(row)

    return [weight, S0, gain0, cost0]


################################### main ############################################
# parameters
q=2 # 2 matroid constraints
p=1 # exchange 1 route 
v=5 # flying speed of the drone in m/s
lamb= 0.5/3.6  # cost per Joul
epsilon=100         #parameter

folder=r'D:\ZhouYanlin\仿真实验\模拟退火\weight_aware\B'
# budge range from 150000 to 200000
Energycost=np.linspace(150000,200000,6)
Budget=Energycost*lamb
#beta= 4e4+ (7.5e4-4e4)*np.random.rand(Size[0],Size[1]) # [60,100] energy constraint
# experiment
for budget_idx in range(len(Budget)):
    B=Budget[budget_idx]
    Route1=[]
    Route2=[]
    Route3=[]
    Route4 = []
    Route5 = []
    Route6=[]

    Gain1=[]
    Gain2=[]
    Gain3=[]
    Gain4 = []
    Gain5 = []
    Gain6=[]
    
    file=folder+'\B='+str(int(B))+'.csv'
    for time in range(5):
        # DG_num controls the DG number 
        # Route_num controls the Route number
        # They cannot exceed the size of original route set
        DG_num=50
        Route_num=4
        
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
        for m in range(Size[0][0]):
            for n in range (Size[0][1]):
                w=math.floor(Weight[m][n])-math.floor(Weight[m][n])%5
                Weight[m][n]=copy.deepcopy(w)
        weight_avg = np.mean(Weight, dtype=np.float64)
        #Ph=0.0256*Weight+26.6172  # hovering power 
        #Pf=0.0256*Weight+25.7519  # flight power at 5m/s
        route_num=Size[0][0]*Size[0][1] 
        size=route_num         #number
        
        
        Solution1 = WeightAssign(RouteSet,Distance,Utility,Indicator,Weight)
        '''
        Solution2 = WeightGU(RouteSet, Distance, Utility, Indicator, Weight)
        Solution3 = WeightGE(RouteSet, Distance, Utility, Indicator, Weight)
        Solution4 = WeightGD(RouteSet, Distance, Utility, Indicator, Weight)
        Solution5 = WeightRA(RouteSet, Distance, Utility, Indicator, Weight)
        Solution6 = WeightSA(RouteSet, Distance, Utility, Indicator, Weight)
        '''
        
        Route1.append(Solution1[1])
        '''
        Route2.append(Solution2[1])
        Route3.append(Solution3[1])
        Route4.append(Solution4[1])
        Route5.append(Solution5[1])
        Route6.append(Solution6[1])
        '''
        
        
        Gain1.append(Solution1[2])
        '''
        Gain2.append(Solution2[2])
        Gain3.append(Solution3[2])
        Gain4.append(Solution4[2])
        Gain5.append(Solution5[2])
        Gain6.append(Solution6[2])
        '''

    title = 'B=' + str(int(B))
    with open(file, 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(['Ours', B, Route1, Gain1])
        '''
        wr.writerow(title)
        wr.writerow(['Ours', B, Route1, Gain1])
        wr.writerow(['GU', B, Route2, Gain2])
        wr.writerow(['GE', B, Route3, Gain3])
        wr.writerow(['GD', B, Route4, Gain4])
        wr.writerow(['RA', B, Route5, Gain5])
        wr.writerow(['SA', B, Route6, Gain6])
        '''
        
