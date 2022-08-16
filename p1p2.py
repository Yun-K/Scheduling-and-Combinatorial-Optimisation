# %%
# !python -m pip install --upgrade --user ortools
# !pip install  ortools

# %% [markdown]
# # Part 1: Mathematical Programming

# %% [markdown]
# Below `CloudResourceAllocation` class model the cloud resource allocation dataset that contains: 
# > N = number of jobs,
# >
# > Q1 = CPU capacity,
# >
# > Q2 = memory capacity AND
# >
# > a list of cloud resources **jobs** that contains these columns: 
# > > **ID, CPU demand, memory demand,and payment of a job.**
# 
# The goal of the resource allocation problem is to decide which job to accept (and which to decline), so that the CPU and memory capacity is not exceeded by the accepted jobs, and the total charged payment is maximised.
# 
# The goal of the resource allocation problem is to decide which job to accept (and which to decline), so that the CPU and memory capacity is not exceeded by the accepted jobs, and the total charged payment is maximised
# 
# Below defines the **Bounding method**
# > Bounding: This is to find the upper/lower bound of the optimal solution of a branch/sub-problem based on optimistic estimate.
# 
# Relax the integer constraints of the $x_i$ so that the variables can take continous values. It refers to the cpu and memory capacity can take continous values.
# 

# %%
from distutils.command.build_scripts import first_line_re
from tkinter.tix import COLUMN
import pandas as pd
import numpy as np
from ortools.linear_solver import pywraplp
from ortools.init import pywrapinit
from ortools.sat.python import cp_model
# Import deque for the stack structure, copy for deep copy nodes
from collections import deque
import copy
print('-'*100)
print('-'*100)
print("Part 1: ")
print('-'*100)
print('-'*100)



class CloudResourceAllocation:
    '''For each instance, we have following fields: 
            the 1st line of the csv files contains the number of jobs N,
            2nd line contains the CPU capacity Q1, and memory capacity Q2, 
            3rd line onwards contains the 
                    ID, CPU demand, memory demand,
                    and payment of a job.
    '''
        
    # main constructor 
    def __init__(self, N, Q1, Q2, jobs):
        '''N is the number of jobs, i.e. len(jobs)'''
        self.N = N
        self.Q1 = Q1
        self.Q2 = Q2
        self.jobs = jobs
        
    @classmethod
    def constructFromFile(cls, filePath):
        '''Read from file and construct an instance of CloudResourceAllocation'''
        with open(filePath, 'r') as file:
            first_line = file.readline()
            second_line = file.readline()
            N = int(first_line.split(',')[0])
            Q1,Q2 = int(second_line.split(',')[0]),int(second_line.split(',')[1])
            
        jobs = pd.read_csv(filePath, skiprows=range(2), header=None)
        jobs.columns = ['Job ID', 'CPUDemand', 'MemoryDemand','payment']
        return cls(N, Q1, Q2, jobs)
    
    def define_maths_models(self):
        '''For defining the mathematical models.
            https://developers.google.com/optimization/cp/channeling
        Maximize the total payment counted by the selected jobs denoated as CiXi where the job i is charged a payment of Ci, subject to the following constraints:
            1. The selected accepted jobs' CPU demand must be less than or equal to the CPU capacity Q1.
            2. The selected accepted jobs' memory demand must be less than or equal to the memory capacity Q2.
            3. xi is binary, i.e. 0 or 1. if xi = 1, then the job is selected.
            4. i : 1 to N, i.e. the i-th job is selected if xi = 1.
        
        '''
        self.solver = pywraplp.Solver('SolveAssignmentProblemMIP', pywraplp.Solver.CBC_MIXED_INTEGER_PROGRAMMING)
        # self.solver = pywraplp.Solver.CreateSolver('SCIP')
        # self.solver = cp_model.CpModel()
        if not self.solver:
            return
        # define variable, we only need N number of x for each job
        # other variable like c, di1,di2, ...,dn will be used as the coefficients 
        self.x = {}
        for i in range(self.N):
            self.x[i] = self.solver.IntVar(0, 1, 'x[%i]' % i)
        # print(f"num of variables: {self.solver.NumVariables()}")
        
        constraint_expr1,constraint_expr2, obj_express= [],[],[]
        for i in range(self.N):
            constraint_expr1.append(self.jobs['CPUDemand'][i] * self.x[i])
            constraint_expr2.append(self.jobs['MemoryDemand'][i] * self.x[i])
            obj_express.append(self.jobs['payment'][i] * self.x[i])
        # define constraints
        self.solver.Add(sum(constraint_expr1)<= self.Q1,"cpu_capacity_constraint")
        self.solver.Add(sum(constraint_expr2) <= self.Q2,"memory_capacity_constraint")
        
        # define objective function
        self.solver.Maximize(sum(obj_express))
        
    
    def solve_assignment_problem(self):
        '''Solve the assignment problem'''
        status = self.solver.Solve()
        if status == pywraplp.Solver.OPTIMAL:
            # print 2 constraints: cpu capacity constraint and memory capacity constraint
            total_cpu_used , total_memory_used = 0,0
            x_sol_vals_str, x_sol_vals_is_1 = "",[]
            for j in range(self.N):
                x_temp = self.x[j].solution_value()
                x_sol_vals_str += f"\t{self.x[j].name()} : {x_temp} \n"
                if x_temp == 1:
                    x_sol_vals_is_1.append(self.x[j].name())
                # count the total cpu and memory used
                total_cpu_used += self.jobs['CPUDemand'][j] * x_temp
                total_memory_used += self.jobs['MemoryDemand'][j] * x_temp
                
            print(f'Objective value(Total Payment) ={self.solver.Objective().Value()}')
            print(f"Total CPU used = {total_cpu_used} out of the CPU Capacity:{self.Q1}")
            print(f"Total Memory used = {total_memory_used} out of the Memory Capacity:{self.Q2}")
            print(f"Solution: (The obtained  xi values): \n Value is 1, which means they are selected:\n\t{x_sol_vals_is_1} \nAll:\n\t{x_sol_vals_str}")
            
            
            # Statistics.
            print('-'*15)
            print('Statistics')
            print(' Problem solved in %f milliseconds ' % self.solver.wall_time())
            print(' Problem solved in %d iterations ' % self.solver.iterations())
            print(' Problem solved in %d branch-and-bound nodes ' % self.solver.nodes())
        else:
            print('The problem does not have an optimal solution.')

    
    def get_jobs(self):
        return self.jobs
    
    def __str__(self) -> str:
        return f'N: {self.N}, \nCPU Capacity Q1: {self.Q1}, Memort Capacity Q2: {self.Q2}, \n Jobs left:\n{self.jobs}'
    
    
smallFilePath = 'cloud_resource_allocation/small.csv'
largeFilePath = 'cloud_resource_allocation/large.csv'
smallDS = CloudResourceAllocation.constructFromFile(smallFilePath)
largeDS = CloudResourceAllocation.constructFromFile(largeFilePath)
# smallDS.get_jobs()['CPUDemand'][0]

# %%
print('-'*80)
print("Small Cloud Resource Allocation dataset:")
print('-'*20)
smallDS.define_maths_models()
smallDS.solve_assignment_problem()
print('-'*80)
print()

# %%
print('-'*80)
print("Large Cloud Resource Allocation dataset:")
print('-'*20)
largeDS.define_maths_models()
largeDS.solve_assignment_problem()
print('-'*80)

# %%
# # large = CloudResourceAllocation(largeFilePath)
# print(smallDS.__str__())
# print('-'*70)
# print(largeDS.__str__())

# %%
# len(smallDS.get_jobs())
# smallDS.get_jobs().iloc[1:]['CPUDemand'] 
# smallDS.get_jobs().iloc[0:]

# %% [markdown]
# ## The following code is inspired from the given tutorial on [GitHub](https://github.com/meiyi1986/tutorials/blob/master/notebooks/knapsack-branch-bound.ipynb)
# 
# `They are not used as the final submitted solution of the part 1`

# %%
# branch and bound 
import fractions
def bounding(ds:CloudResourceAllocation):
    bound = 0
    
    # payments, weights, q1_cpu_capacity, q2_memory_capacity 
    remaining_q1_cpu_capacity ,remaining_q2_memory_capacity = ds.Q1,ds.Q2
    
    # define the efficiency by adding payment per cpuDemand and payment per memoryDemand
    efficiency = [ds.get_jobs().iloc[i]['payment'] / ds.get_jobs().iloc[i]['CPUDemand'] 
                    + ds.get_jobs().iloc[i]['payment'] / ds.get_jobs().iloc[i]['MemoryDemand'] for i in range(len(ds.get_jobs()))]
    sorted_idx = sorted(range(len(efficiency)), reverse=True, key=efficiency.__getitem__)
    print(sorted_idx)
    for i in sorted_idx:
        q1_exceed = ds.get_jobs().iloc[i]['CPUDemand'] > remaining_q1_cpu_capacity
        q2_exceed = ds.get_jobs().iloc[i]['MemoryDemand'] >remaining_q2_memory_capacity 
        if q1_exceed or q2_exceed :
            # fraction of the job that can be allocated
            # fraction = min(remaining_q1_cpu_capacity / ds.get_jobs().iloc[i]['CPUDemand'],
            #                 remaining_q2_memory_capacity / ds.get_jobs().iloc[i]['MemoryDemand'])
            fraction = remaining_q1_cpu_capacity / ds.get_jobs().iloc[i]['CPUDemand'] if q1_exceed else remaining_q2_memory_capacity / ds.get_jobs().iloc[i]['MemoryDemand']

            frac_value = ds.get_jobs().iloc[i]['payment'] * fraction
            bound += frac_value
            return bound
            
        bound += ds.get_jobs().iloc[i]['payment']
        remaining_q1_cpu_capacity -= ds.get_jobs().iloc[i]['CPUDemand']
        remaining_q2_memory_capacity -= ds.get_jobs().iloc[i]['MemoryDemand']
    return bound

# bounding(smallDS)

# %%


def cloudResourceAllocation_bb_dfs(ds:CloudResourceAllocation):#(values, weights, capacity):
    # payments, weights, q1_cpu_capacity, q2_memory_capacity 
    remaining_q1_cpu_capacity ,remaining_q2_memory_capacity = ds.Q1,ds.Q2
    
    # Initialise the root, where 'expanded_item' indicates the item to be expanded at this node
    root = {
        'solution': [0] * len(ds.get_jobs()),
        'total payment': 0,
        'total cpu used': 0,
        'total memory used': 0,
        'expanded_item': 0
    }
    
    # Initially, the fringe contains the root node only
    best_solution = root
    fringe = deque()
    fringe.append(root)
    
    while len(fringe) > 0:
        # Depth-first-search, Last-In-First-Out of the stack
        node = fringe.pop()
        
        # Check if the node is a leaf node
        if node['expanded_item'] == len(ds.get_jobs()):
            if node['total payment'] > best_solution['total payment']:
                best_solution = node
                continue
        
        # Obtain the sub-problem: values, weights, capacity
        node_sub_jobs = ds.get_jobs().iloc[node['expanded_item']:]
        node_sub_q1_cpu_capacity = ds.Q1 - node['total cpu used']
        node_sub_q2_mem_capacity = ds.Q2 - node['total memory used']
        
        # Bounding on the sub-problem, and then add the value of the current solution
        bound = node['total payment'] + bounding(
            CloudResourceAllocation(
                len(node_sub_jobs),
                node_sub_q1_cpu_capacity,
                node_sub_q2_mem_capacity,
                node_sub_jobs)
        )
        # Prune the branch
        if bound <= best_solution['total payment']:
            continue
            
        # Branching on the expanded item, 0 or 1
        expanded_item = node['expanded_item']
        
        # Child 1: unselect the expanded item
        child1 = copy.deepcopy(node)
        child1['solution'][expanded_item] = 0
        child1['expanded_item'] = expanded_item + 1
        fringe.append(child1)
        
        # Child 2: select the expanded item if the capacity is enough
        new_cpu_demand = node['total cpu used']+ds.get_jobs().iloc[expanded_item]['CPUDemand']
        new_mem_demand = node['total memory used']+ds.get_jobs().iloc[expanded_item]['MemoryDemand']
        
        if new_cpu_demand <= ds.Q1 and new_mem_demand <= ds.Q2:
            child2 = copy.deepcopy(node)
            child2['solution'][expanded_item] = 1
            child2['total payment'] = node['total payment']+ ds.get_jobs().iloc[expanded_item]['payment']
            child2['total cpu used'] = new_cpu_demand
            child2['total memory used'] = new_mem_demand
            child2['expanded_item'] = expanded_item + 1
            fringe.append(child2)
    return best_solution


# %%
def printResult(ds:CloudResourceAllocation):
    for k,v in cloudResourceAllocation_bb_dfs(ds).items():
        suffix = ''
        # if k contains cpu
        if k.find('cpu') != -1:
            suffix = f" out of {ds.Q1}"
        if k.find('mem') != -1:
            suffix = f" out of {ds.Q2}"
        
        print(f"{k}: {v} {suffix}")
    print('-'*50)
    print('-'*50)
    return cloudResourceAllocation_bb_dfs(ds)

# %%
# printResult(smallDS)
# printResult(largeDS)
# print("Incorrect, hence they are deprecated, the above that use Google-OR tool is the correct way to do it")

# %% [markdown]
# # Part 2: Greedy Heuristic
# 
# The above cloud resource allocation problem can be seen as a 2-dimensional knapsack problem, where the weight of items has two dimensions rather than a single dimension.
# 
# Instead of solving it by mathematical programming, we can use our domain knowledge to design greedy heuristics.
# 
# The greedy heuristics should be as follows:
# 1. Sort the jobs by some criterion.
# 2. Initially, no job is accepted, and used CPU and memory are 0.
# 3. Scan each sorted job. If the job can fit into the CPU and memory requirement, then accept it. Otherwise, decline it.
# 
# ## Requirement
# In this part, you are required to design and implement two different greedy heuristics, i.e., two criteria to sort
# the jobs for the cloud resource allocation problem.
# 1. Implement two greedy heuristics and apply them to the small.csv and large.csv instances, to generate a
# solution for each instance.
# 2. In the report, clearly describe the two heuristics, i.e., sorting criteria of the jobs you designed
# 3. In the report, clearly describe the solutions obtained by the two heuristics. Specifically, for each heuristic, you need to list the “[selected order, job ID, CPU demand, memory demand, payment, sorting criterion]” of each selected job, in the order that they are selected by the heuristic. An example is as follows:
# 
# 4. In the report, for each of the two instances, compare the solutions obtained by your two greedy heuristics, and make deep and comprehensive discussions based on your observations

# %%
print('-'*100)
print('-'*100)
print("Part 2: ")
print('-'*100)
print('-'*100)

def greedy_cloudAllocationProblem(ds:CloudResourceAllocation):
    '''
    The greedy algorithm for the cloud resource allocation problem.
    It takes the dataset with the sorted jobs, and selectes the jobs one by one until both cpu and memorycapacities are exhausted.
    Return selected jobs
    '''
    selected = []
    remaining_q1_cpu_capacity ,remaining_q2_memory_capacity = ds.Q1,ds.Q2
    for i in range(len(ds.get_jobs())):
        if remaining_q1_cpu_capacity >= ds.get_jobs().iloc[i]['CPUDemand'] and remaining_q2_memory_capacity >= ds.get_jobs().iloc[i]['MemoryDemand']:
            selected.append(ds.get_jobs().iloc[i])
            remaining_q1_cpu_capacity -= ds.get_jobs().iloc[i]['CPUDemand']
            remaining_q2_memory_capacity -= ds.get_jobs().iloc[i]['MemoryDemand']
            
    return selected

def print_solution(sel_items:list, ds:CloudResourceAllocation, method_name:str, dataset_name:str):
    # selected_names = [item['name'] for item in sel_items]
    selected_payments = sum([item['payment'] for item in sel_items])
    selected_cpu_used = sum([item['CPUDemand'] for item in sel_items])
    selected_mem_used = sum([item['MemoryDemand'] for item in sel_items])
    print('-'*100)
    print('-'*100)
    print(f"{method_name} for {dataset_name} :")
    print(f' Objective value(Total payment): {selected_payments} \n\t Total CPU used: {selected_cpu_used} out of CPU capacity:{ds.Q1} \n\t Total memory used: {selected_mem_used} out of Memory capacity:{ds.Q2}.')

    print(f'{len(sel_items)} selected jobs: ')
    # convert to dataframe
    sel_items = pd.DataFrame(sel_items)
    # 
    print(sel_items)
    return sel_items

# %%
def sort_by_efficiency(ds:CloudResourceAllocation):
    '''
    Sort the jobs by efficiency, and return the sorted ds
    The efficiency refer to by adding payment per cpuDemand and payment per memoryDemand
    '''
    # payments, weights, q1_cpu_capacity, q2_memory_capacity 
    remaining_q1_cpu_capacity ,remaining_q2_memory_capacity = ds.Q1,ds.Q2
    
    # define the efficiency by adding payment per cpuDemand and payment per memoryDemand
    efficiency = [ds.get_jobs().iloc[i]['payment'] / ds.get_jobs().iloc[i]['CPUDemand'] 
                    + ds.get_jobs().iloc[i]['payment'] / ds.get_jobs().iloc[i]['MemoryDemand'] 
                    for i in range(len(ds.get_jobs()))]
    # add the efficiency to the dataset
    ds.jobs['sorting criterion'] = efficiency
    # sort the jobs by sorting criterion
    ds.jobs.sort_values(by='sorting criterion', ascending=False, inplace=True)
    # insert the selected order to the dataset
    ds.jobs.insert(0, 'selected order', [i+1 for i in range(len(ds.get_jobs()))])
    return ds

def sort_by_payment_per_cpuDemand(ds:CloudResourceAllocation):
    '''
    Sort the jobs by efficiency, and return the sorted ds
    The efficiency refer to by  payment per cpuDemand 
    '''
    # payments, weights, q1_cpu_capacity, q2_memory_capacity 
    remaining_q1_cpu_capacity ,remaining_q2_memory_capacity = ds.Q1,ds.Q2
    
    # define the efficiency by adding payment per cpuDemand and payment per memoryDemand
    efficiency = [ds.get_jobs().iloc[i]['payment'] / ds.get_jobs().iloc[i]['CPUDemand'] 
                    for i in range(len(ds.get_jobs()))]
    # add the efficiency to the dataset
    ds.get_jobs()['sorting criterion'] = efficiency
    # sort the get_jobs() by sorting criterion
    ds.get_jobs().sort_values(by='sorting criterion', ascending=False, inplace=True)
    # insert the selected order to the dataset
    ds.get_jobs().insert(0, 'selected order', [i+1 for i in range(len(ds.get_jobs()))])
    return ds

# sort_by_efficiency(smallDS).get_jobs()


# %%
def get_final_efficiency_criteria_result(ds:CloudResourceAllocation,dataset_name:str):
    ds = sort_by_efficiency(ds)
    selectedJobs = greedy_cloudAllocationProblem(ds)
    print_solution(selectedJobs, ds, '(payment/cpu ratio + payment/memory) heurstic', dataset_name)
    
def get_final_payment_per_cpu_criteria_result(ds:CloudResourceAllocation,dataset_name:str):
    ds = sort_by_payment_per_cpuDemand(ds)
    selectedJobs = greedy_cloudAllocationProblem(ds)
    print_solution(selectedJobs, ds, '(payment/cpu ratio) heurstic', dataset_name)
    
smallDS1 = CloudResourceAllocation.constructFromFile(smallFilePath)
get_final_efficiency_criteria_result(smallDS1,"small cloud resource dataset")


# %%
largeDS1 = CloudResourceAllocation.constructFromFile(largeFilePath)
get_final_efficiency_criteria_result(largeDS1,"large cloud resource dataset")


# %%

smallDS2 = CloudResourceAllocation.constructFromFile(smallFilePath)
get_final_payment_per_cpu_criteria_result(smallDS2,"small cloud resource dataset")


# %%
largeDS2 = CloudResourceAllocation.constructFromFile(largeFilePath)
get_final_payment_per_cpu_criteria_result(largeDS2,"large cloud resource dataset")


