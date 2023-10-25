import systems
import numpy as np
import matplotlib.pyplot as plt 
from tqdm import tqdm
import pandas as pd 


def self_opt_noiseless(x):
    plant = systems.Static_PDE_reaction_system()
    f = plant.objective_noise_less
    return f(x)
import os 
try:
    os.mkdir('function_creation/datasets')
except FileExistsError:
    pass

n = 20
x1 = np.linspace(40,150,n)
x2 = np.linspace(0.5,10,n)
X1,X2 = np.meshgrid(x1,x2)
Z = np.zeros((n,n))
x_list = []
z_list = []
df = {}
for i in tqdm(range(n)):
    for j in tqdm(range(n)):
        x = [X1[i,j],X2[i,j]]
        Z[i,j] = self_opt_noiseless([X1[i,j],X2[i,j]])
        x_list.append(x)
        z_list.append(Z[i,j])
        df['Reactor Temperature (C)'] = [xi[0] for xi in x_list]
        df['Damkohler Number'] = [xi[1] for xi in x_list]
        df['Objective'] = z_list
        df_pd = pd.DataFrame(df)  

        df_pd.to_csv('function_creation/datasets/selfopt_dataset.csv',index=False)

plt.figure
plt.contourf(X1,X2,Z,levels=200)
plt.savefig('function_creation/datasets/selfopt_dataset.png',dpi=300)


