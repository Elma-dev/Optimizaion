#!/usr/bin/env python
# coding: utf-8

# In[8]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import itertools as itr
import warnings
warnings.filterwarnings("ignore")
import random as rd
import math as mt


# # SIMULATED ANNEALING

# In[9]:


def improve(x,y,k):
    # for the decision variable x -->
    rand_num_x_1 = np.random.rand() # increase or decrease x value?
    rand_num_x_2 = np.random.rand() # by how much?

    if rand_num_x_1 >= 0.5: # greater than 0.5, we increase
        step_size_x = k * rand_num_x_2 # make sure we make a smalle                                           
    else:
        step_size_x = -k * rand_num_x_2 # less than 0.5, we decrease

    # for the decision variable y -->
    rand_num_y_1 = np.random.rand() # increase or decrease y value?
    rand_num_y_2 = np.random.rand() # by how much?

    if rand_num_y_1 >= 0.5: # greater than 0.5, we increase
        step_size_y = k * rand_num_y_2 # make sure we make a smaller                                         
    else:
        step_size_y = -k * rand_num_y_2 # less than 0.5, we decrease
        
    # temporary because we still need to know if we should take this
    # new solution or stay where we are and look again
    x_temporary = x + step_size_x # add or subtract the x value
    y_temporary = y + step_size_y # add or subtract the y value
                
    return (x_temporary,y_temporary) 


def update(x,y,x_temporary,y_temporary,T0):
    rand_num = np.random.rand()
    
    # the possible new move with the temporary values need to see if better or worse than where we currently are
    obj_val_possible = ((x_temporary**2)+y_temporary-11)**2+(x_temporary+(y_temporary**2)-7)**2

    # where we are currently
    obj_val_current = ((x**2)+y-11)**2+(x+(y**2)-7)**2
    
    formula = 1/(np.exp((obj_val_possible - obj_val_current)/T0))
    
    if obj_val_possible <= obj_val_current: 
        x = x_temporary
        y = y_temporary

    elif rand_num <= formula: 
        x = x_temporary
        y = y_temporary

    else: 
        x = x
        y = y
        
    return (x,y,obj_val_current)


# # TABU SEARCH

# In[10]:


def cost(distance,flow,solution):
    
    # Make a dataframe of the initial solution
    New_Dist_DF = distance.reindex(columns=solution, index=solution)
    New_Dist_Arr = np.array(New_Dist_DF)
    
    # Make a dataframe of the cost of the initial solution
    Objfun1_start = pd.DataFrame(New_Dist_Arr*flow)
    Objfun1_start_Arr = np.array(Objfun1_start)
    sum_start_int = sum(sum(Objfun1_start_Arr))
    
    return sum_start_int

def creat_neighbours(x):
    # To create all surrounding neighborhood
    List_of_N = list(itr.combinations(x, 2)) # From X0, it shows how many combinations of 2's can it get; 8 choose 2
    Counter_for_N = 0
    All_N_for_i = np.empty((0,len(x)))

    for i in List_of_N:
        X_Swap = []
        A_Counter = List_of_N[Counter_for_N] # Goes through each set
        A_1 = A_Counter[0] # First element
        A_2 = A_Counter[1] # Second element

        # ["D","A","C","B","G","E","F","H"]

        # Making a new list of the new set of departments, with 2-opt (swap)
        u= 0
        for j in x: # For elements in X0, swap the set chosen and store it
            if x[u] == A_1:
                X_Swap = np.append(X_Swap,A_2)
            elif x[u] == A_2:
                X_Swap = np.append(X_Swap,A_1)
            else:
                X_Swap = np.append(X_Swap,x[u])

            X_Swap = X_Swap[np.newaxis] # New "X0" after swap

            u = u+1

        All_N_for_i = np.vstack((All_N_for_i,X_Swap)) # Stack all the combinations

        Counter_for_N = Counter_for_N+1
        
    return All_N_for_i

def neighbours_cost(x,distance,flow,All_N_for_i):
    OF_Values_for_N_i = np.empty((0,len(x)+1)) # +1 to add the OF values
    OF_Values_all_N = np.empty((0,len(x)+1)) # +1 to add the OF values

    N_Count = 1

    # Calculating OF for the i-th solution in N
    for i in All_N_for_i:

        Total_Cost_N_i = cost(distance,flow,i)  
        i = i[np.newaxis]
        OF_Values_for_N_i = np.column_stack((Total_Cost_N_i,i)) # Stack the OF value to the deertment vector
        OF_Values_all_N = np.vstack((OF_Values_all_N,OF_Values_for_N_i))
        N_Count = N_Count+1
        
    return OF_Values_all_N

def best_solution(OF_Values_all_N_Ordered,Tabu_List,L):
    
    t = 0
    Current_Sol = OF_Values_all_N_Ordered[t] # Current solution


    while Current_Sol[0] in Tabu_List[:,0]: # If current solution is in Tabu list
        Current_Sol = OF_Values_all_N_Ordered[t]
        t = t+1


    if len(Tabu_List) >= L: # If Tabu list is full
        Tabu_List = np.delete(Tabu_List, (L-1), axis=0) # Delete the last row

    Tabu_List = np.vstack((Current_Sol,Tabu_List))
    
    return (Current_Sol,Tabu_List)

def dynamic_list(x,Current_Sol,Iterations,L):
    
    Mod_Iterations = Iterations%10  

    Ran_1 = np.random.randint(1,len(x)+1)
    Ran_2 = np.random.randint(1,len(x)+1)
    Ran_3 = np.random.randint(1,len(x)+1)

    if Mod_Iterations == 0:
        Xt = []
        A1 = Current_Sol[Ran_1]
        A2 = Current_Sol[Ran_2]

        # Making a new list of the new set of departments
        S_Temp = Current_Sol

        w = 0
        for i in S_Temp:
            if S_Temp[w] == A1:
                Xt=np.append(Xt,A2)
            elif S_Temp[w] == A2:
                Xt=np.append(Xt,A1)
            else:
                Xt=np.append(Xt,S_Temp[w])
            w = w+1

        Current_Sol = Xt

        # Same department gets switched
        Xt = []
        A1 = Current_Sol[Ran_1]
        A2 = Current_Sol[Ran_3]

        # Making a new list of the new set of departments
        w = 0
        for i in Current_Sol:
            if Current_Sol[w] == A1:
                Xt=np.append(Xt,A2)
            elif Current_Sol[w] == A2:
                Xt=np.append(Xt,A1)
            else:
                Xt=np.append(Xt,Current_Sol[w])
            w = w+1

        Current_Sol = Xt


    x = Current_Sol[1:]

    Iterations = Iterations+1

    # Change length of Tabu List every 5 runs, between 5 and 20, dynamic Tabu list
    if Mod_Iterations == 5 or Mod_Iterations == 0:
        L = np.random.randint(5,20)
        
    return L,x,Iterations

def final_soluction(Save_Solutions_Here):
    t = 0
    Final_Here = []
    for i in Save_Solutions_Here:

        if (Save_Solutions_Here[t,0]) <= min(Save_Solutions_Here[:,0]):
            Final_Here = Save_Solutions_Here[t,:]
        t = t+1
        
    return Final_Here


# # GENETIC ALGORITHM

# In[13]:


################################################
### CALCULATING THE Fitness value ###
################################################

# calculate fitness value for the chromosome of 0s and 1s
def fitness(chromosome):  
    
    lb_x = -6 # lower bound for chromosome x
    ub_x = 6 # upper bound for chromosome x
    len_x = (len(chromosome)//2) # length of chromosome x
    lb_y = -6 # lower bound for chromosome y
    ub_y = 6 # upper bound for chromosome y
    len_y = (len(chromosome)//2) # length of chromosome y
    
    precision_x = (ub_x-lb_x)/((2**len_x)-1) # precision for decoding x
    precision_y = (ub_y-lb_y)/((2**len_y)-1) # precision for decoding y
    
    z = 0 # because we start at 2^0, in the formula
    t = 1 # because we start at the very last element of the vector [index -1]
    x_bit_sum = 0 # initiation (sum(bit)*2^i is 0 at first)
    for i in range(len(chromosome)//2):
        x_bit = chromosome[-t]*(2**z)
        x_bit_sum = x_bit_sum + x_bit
        t = t+1
        z = z+1   
    
    z = 0 # because we start at 2^0, in the formula
    t = 1 + (len(chromosome)//2) # [6,8,3,9] (first 2 are y, so index will be 1+2 = -3)
    y_bit_sum = 0 # initiation (sum(bit)*2^i is 0 at first)
    for j in range(len(chromosome)//2):
        y_bit = chromosome[-t]*(2**z)
        y_bit_sum = y_bit_sum + y_bit
        t = t+1
        z = z+1
    
    # the formulas to decode the chromosome of 0s and 1s to an actual number, the value of x or y
    decoded_x = (x_bit_sum*precision_x)+lb_x
    decoded_y = (y_bit_sum*precision_y)+lb_y
    
    # the himmelblau function
    # min ((x**2)+y-11)**2+(x+(y**2)-7)**2
    # objective function value for the decoded x and decoded y
    fitness_value = ((decoded_x**2)+decoded_y-11)**2+(decoded_x+(decoded_y**2)-7)**2
    
    return decoded_x,decoded_y,fitness_value # the defined function will return 3 values



#################################################
### SELECTING TWO PARENTS FROM THE POPULATION ###
### USING TOURNAMENT SELECTION METHOD ###########
#################################################

# finding 2 parents from the pool of solutions
# using the tournament selection method 
def find_parents_ts(all_solutions):
    
    # make an empty array to place the selected parents
    parents = np.empty((0,np.size(all_solutions,1)))
    
    for i in range(2): # do the process twice to get 2 parents
        
        # select 3 random parents from the pool of solutions you have
        
        # get 3 random integers
        indices_list = np.random.choice(len(all_solutions),3,replace=False)
        
        # get the 3 possible parents for selection
        posb_parent_1 = all_solutions[indices_list[0]]
        posb_parent_2 = all_solutions[indices_list[1]]
        posb_parent_3 = all_solutions[indices_list[2]]
        
        # get objective function value (fitness) for each possible parent
        # index no.2 because the objective_value function gives the fitness value at index no.2
        obj_func_parent_1 = fitness(posb_parent_1)[2] # possible parent 1
        obj_func_parent_2 = fitness(posb_parent_2)[2] # possible parent 2
        obj_func_parent_3 = fitness(posb_parent_3)[2] # possible parent 3
        
        # find which parent is the best
        min_obj_func = min(obj_func_parent_1,obj_func_parent_2,obj_func_parent_3)
        
        if min_obj_func == obj_func_parent_1:
            selected_parent = posb_parent_1
        elif min_obj_func == obj_func_parent_2:
            selected_parent = posb_parent_2
        else:
            selected_parent = posb_parent_3
        
        # put the selected parent in the empty array we created above
        parents = np.vstack((parents,selected_parent))
        
    parent_1 = parents[0,:] # parent_1, first element in the array
    parent_2 = parents[1,:] # parent_2, second element in the array
    
    return parent_1,parent_2 # the defined function will return 2 arrays


####################################################
### CROSSOVER THE TWO PARENTS TO CREATE CHILDREN ###
####################################################

# crossover between the 2 parents to create 2 children
# functions inputs are parent_1, parent_2, and the probability you would like for crossover
# default probability of crossover is 1
def crossover(parent_1,parent_2,prob_crsvr=1):
    
    child_1 = np.empty((0,len(parent_1)))
    child_2 = np.empty((0,len(parent_2)))
    
    
    rand_num_to_crsvr_or_not = np.random.rand() # do we crossover or no???
    
    if rand_num_to_crsvr_or_not < prob_crsvr:
        index_1 = np.random.randint(0,len(parent_1))
        index_2 = np.random.randint(0,len(parent_2))
        
        # get different indices
        # to make sure you will crossover at least one gene
        while index_1 == index_2:
            index_2 = np.random.randint(0,len(parent_1))
        
        
        # IF THE INDEX FROM PARENT_1 COMES BEFORE PARENT_2
        # e.g. parent_1 = 0,1,>>1<<,1,0,0,1,0 --> index = 2
        # e.g. parent_2 = 0,0,1,0,0,1,>>1<<,1 --> index = 6
        if index_1 < index_2:
            
            
            ### FOR PARENT_1 ###
            ### FOR PARENT_1 ###
            ### FOR PARENT_1 ###
            
            # first_seg_parent_1 -->
            # for parent_1: the genes from the beginning of parent_1 to the
                    # beginning of the middle segment of parent_1
            first_seg_parent_1 = parent_1[:index_1]
            
            # middle segment; where the crossover will happen
            # for parent_1: the genes from the index chosen for parent_1 to
                    # the index chosen for parent_2
            mid_seg_parent_1 = parent_1[index_1:index_2+1]
            
            # last_seg_parent_1 -->
            # for parent_1: the genes from the end of the middle segment of
                    # parent_1 to the last gene of parent_1
            last_seg_parent_1 = parent_1[index_2+1:]
            
            
            ### FOR PARENT_2 ###
            ### FOR PARENT_2 ###
            ### FOR PARENT_2 ###
            
            # first_seg_parent_2 --> same as parent_1
            first_seg_parent_2 = parent_2[:index_1]
            
            # mid_seg_parent_2 --> same as parent_1
            mid_seg_parent_2 = parent_2[index_1:index_2+1]
            
            # last_seg_parent_2 --> same as parent_1
            last_seg_parent_2 = parent_2[index_2+1:]
            
            
            ### CREATING CHILD_1 ###
            ### CREATING CHILD_1 ###
            ### CREATING CHILD_1 ###
            
            # the first segmant from parent_1
            # plus the middle segment from parent_2
            # plus the last segment from parent_1
            child_1 = np.concatenate((first_seg_parent_1,mid_seg_parent_2,
                                      last_seg_parent_1))
            
            
            ### CREATING CHILD_2 ###
            ### CREATING CHILD_2 ###
            ### CREATING CHILD_2 ###
            
            # the first segmant from parent_2
            # plus the middle segment from parent_1
            # plus the last segment from parent_2
            child_2 = np.concatenate((first_seg_parent_2,mid_seg_parent_1,
                                      last_seg_parent_2))
        
        
        
        ### THE SAME PROCESS BUT INDEX FROM PARENT_2 COMES BEFORE PARENT_1
        # e.g. parent_1 = 0,0,1,0,0,1,>>1<<,1 --> index = 6
        # e.g. parent_2 = 0,1,>>1<<,1,0,0,1,0 --> index = 2
        else:
            
            first_seg_parent_1 = parent_1[:index_2]
            mid_seg_parent_1 = parent_1[index_2:index_1+1]
            last_seg_parent_1 = parent_1[index_1+1:]
            
            first_seg_parent_2 = parent_2[:index_2]
            mid_seg_parent_2 = parent_2[index_2:index_1+1]
            last_seg_parent_2 = parent_2[index_1+1:]
            
            
            child_1 = np.concatenate((first_seg_parent_1,mid_seg_parent_2,
                                      last_seg_parent_1))
            child_2 = np.concatenate((first_seg_parent_2,mid_seg_parent_1,
                                      last_seg_parent_2))
     
    # when we will not crossover
    # when rand_num_to_crsvr_or_not is NOT less (is greater) than prob_crsvr
    # when prob_crsvr == 1, then rand_num_to_crsvr_or_not will always be less
            # than prob_crsvr, so we will always crossover then
    else:
        child_1 = parent_1
        child_2 = parent_2
    
    return child_1,child_2 # the defined function will return 2 arrays



############################################################
### MUTATING THE TWO CHILDREN TO CREATE MUTATED CHILDREN ###
############################################################

# mutation for the 2 children
# functions inputs are child_1, child_2, and the probability you would like for mutation
# default probability of mutation is 0.2
def mutation(child_1,child_2,prob_mutation=0.2):
    
    # mutated_child_1
    mutated_child_1 = np.empty((0,len(child_1)))
      
    t = 0 # start at the very first index of child_1
    for i in child_1: # for each gene (index)
        
        rand_num_to_mutate_or_not_1 = np.random.rand() # do we mutate or no???
        
        # if the rand_num_to_mutate_or_not_1 is less that the probability of mutation
                # then we mutate at that given gene (index we are currently at)
        if rand_num_to_mutate_or_not_1 < prob_mutation:
            
            if child_1[t] == 0: # if we mutate, a 0 becomes a 1
                child_1[t] = 1
            
            else:
                child_1[t] = 0  # if we mutate, a 1 becomes a 0
            
            mutated_child_1 = child_1
            
            t = t+1
        
        else:
            mutated_child_1 = child_1
            
            t = t+1
    
       
    # mutated_child_2
    # same process as mutated_child_1
    mutated_child_2 = np.empty((0,len(child_2)))
    
    t = 0
    for i in child_2:
        
        rand_num_to_mutate_or_not_2 = np.random.rand() # prob. to mutate
        
        if rand_num_to_mutate_or_not_2 < prob_mutation:
            
            if child_2[t] == 0:
                child_2[t] = 1
           
            else:
                child_2[t] = 0
            
            mutated_child_2 = child_2
            
            t = t+1
        
        else:
            mutated_child_2 = child_2
            
            t = t+1
    
    return mutated_child_1,mutated_child_2 # the defined function will return 2 arrays# GENETIC ALGORITHM

def initialisez(x_y_string,N):
    
    # create an empty array to put initial population
    pool_of_solutions = np.empty((0,len(x_y_string)))

    # shuffle the elements in the initial solution (vector)
    # shuffle n times, where n is the no. of the desired population
    for i in range(N):
        rd.shuffle(x_y_string)
        pool_of_solutions = np.vstack((pool_of_solutions,x_y_string))
        
    return pool_of_solutions

def create_population(x_y_string,N,pool_of_solutions,pc,pm):
    
    # an empty array for saving the new generation
    new_population = np.empty((0,len(x_y_string)))
    
    # an empty array for saving the new generation plus its obj func val
    new_population_with_obj_val = np.empty((0,len(x_y_string)+1))
    
    # an empty array for saving the best solution (chromosome) for each generation
    sorted_best_for_plotting = np.empty((0,len(x_y_string)+1))
    
    for j in range(int(N/2)): # population/2 because each gives 2 parents
          
        # selecting 2 parents using tournament selection
        parent_1 = find_parents_ts(pool_of_solutions)[0]
        parent_2 = find_parents_ts(pool_of_solutions)[1]
        
        
        # crossover the 2 parents to get 2 children
        child_1 = crossover(parent_1,parent_2,prob_crsvr=pc)[0]
        child_2 = crossover(parent_1,parent_2,prob_crsvr=pc)[1]
        
        
        # mutating the 2 children to get 2 mutated children
        mutated_child_1 = mutation(child_1,child_2,prob_mutation=pm)[0]  
        mutated_child_2 = mutation(child_1,child_2,prob_mutation=pm)[1] 
        
        
        # getting the obj val (fitness value) for the 2 mutated children
        obj_val_mutated_child_1 = fitness(mutated_child_1)[2]
        obj_val_mutated_child_2 = fitness(mutated_child_2)[2]

        
        # for each mutated child, put its obj val next to it
        mutant_1_with_obj_val = np.hstack((obj_val_mutated_child_1,mutated_child_1)) # lines 103 and 111
        
        mutant_2_with_obj_val = np.hstack((obj_val_mutated_child_2,mutated_child_2)) # lines 105 and 112
        
        
        # we need to create the new population for the next generation
        new_population = np.vstack((new_population,mutated_child_1,mutated_child_2))
        
        
        # same explanation as above, but we include the obj val for each solution as well check line 64
        new_population_with_obj_val = np.vstack((new_population_with_obj_val,mutant_1_with_obj_val,mutant_2_with_obj_val))
        
    return (new_population,new_population_with_obj_val)


# In[ ]:




