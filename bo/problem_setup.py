import sys
import os
import pandas as pd 
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from utils import *



behaviours = ['expert','idiot','trusting','random']

file_names = []
for behaviour in behaviours:
    aq = 'UCB'
    problem_data = {}
    problem_data["sample_initial"] = 4
    problem_data["gp_ms"] = 8
    problem_data["alternatives"] = 4
    problem_data["NSGA_iters"] = 50
    problem_data["plotting"] = True
    problem_data['regret_tolerance'] = 0.0001
    problem_data['max_iterations'] = 50
    problem_data['lengthscale'] = 0.5
    problem_data['human_behaviour'] = 'expert' # idiot, expert, trusting, random, or Float
    problem_data['acquisition_function'] = aq
    df = pd.DataFrame(problem_data,index=[0])
    file_name = 'bo/problem_data/'+str(behaviour)+'.csv'
    file_names.append(file_name)
    df.to_csv(file_name,index=False)

# save file_names as a .txt
with open('bo/problem_data/file_names.txt', 'w') as f:
    for item in file_names:
        f.write("%s\n" % item)

problems = 20
f_keys = [np.random.randint(0, 10000) for i in range(problems)]
df = pd.DataFrame({'f_keys':f_keys})
df.to_csv('function_creation/f_keys.csv',index=False)
