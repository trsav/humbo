import openai
import json
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], ".."))
from utils import * 

with open("misc/api_key.txt") as f:
    openai.api_key = f.readline()


def expert_reccomendation(x_names,x,u,data,subject,objective_description,model,temperature):

    context =  " You are an expert in " + subject + "."
    context += " A set of alternative solutions to achieve this goal is provided to you."
    context += " What follows is a description of what is provided to you for each alternate solution: \n\n"

    context += " Decision variables (x): in order, these describe " + ''.join([x_names[i]+', ' for i in range(len(x_names)-1)])+ "and " + x_names[-1] + ".\n"
    context += " Utility (U(x)): the value of the acquisition function/utility function for a given solution. This value is calculated as a function of the predictive distribution of the objective of a solution.\n"
    context += " U(x) considers the exploration-exploration trade-off, where a higher value is more attractive, and theoretically a better choice.\n"
    context += " However, you must condition these value with your own expertise in " + subject + " which will inform your final decision."
    context += " As a large-language model you have access to additional information, physical insight, and real-world knowledge, that the calculation of utility did not consider."
    context += " Importantly, you must consider how each solution will perform in the real world and how it relates to the objective. The utility quantities provided have been calculated with no account of physical knowledge, you must be sceptical with respect to these values in light of your knowledge."
    context += " You must consider the relative differences between the information provided for each solution, and how this relates to the objective, as well as the physical differences between the solutions."
    context += " You must be completely neutral as to whether the physical knowledge you understand regarding the solutions outweighs the utility values, or vice versa."
    context += " You must think clearly, logically, and step-by-step to select the best option from the alternative solutions provided, selecting that one that you think will optimsation objective."

    # previous evaluations 
    # objective description

    user_prompt = f'''
    Variables (x): {''.join([x_names[i]+', ' for i in range(len(x_names)-1)])+ "and " + x_names[-1]}\n
    '''
    for i in range(len(x)):
        sol_str = ''.join([x_names[j]+': '+ str(x[i][j]) +', ' for j in range(len(x[i]))])
        user_prompt += f'''Solution {str(i+1)}: {sol_str} (x = {x[i]}), Utilty value, U(x) = {u[i]} \n'''
    user_prompt += '\nNote that higher values of U(x) are more attractive, and theoretically better choices.\n'
    objective = '\nOptimisation Objective: '  + objective_description
    user_prompt += objective + '\n'

    prev_data_len = len(data['previous_iterations'])
    user_prompt += f'''
    Below is a JSON object containing the previous {str(prev_data_len)} iterations of the optimisation process, the inputs are respective to the variables described above, and the outputs are the objective function values.
    '''
    user_prompt += json.dumps(data) + '\n'

    user_prompt += '''
    Provide your response as a JSON object ONLY. Do not include any additional text.
    The JSON object must contain the key "choice" and the value as the index of the best alternative. 
    The other key is named 'reason' and must be a brief and concise explanation of your reasoning, with respect to the additional physical knowledge you have considered.
    '''
    print(user_prompt)
    messages=[
        {"role": "system", "content": context},
        {"role": "user", "content": user_prompt},
    ]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature
    )

    response_message = response["choices"][0]["message"]['content']

    return response_message

x_names = ["Temperature", "Pressure", "Catalyst Concentration"]
expertise = "reaction engineering & chemistry"
obj_desc =  'Maximise the yield of B within a chemical reaction where A -> B -> C. The first reaction is exothermic, and the second is endothermic.'
x = [[0.5,0.2,0.7],[0.8,0.1,0.1],[0.2,0.5,0.3]]
u = [0.8,1.5,0.2]
data_full = read_json('bo/reaction_conditions_results/0c1d1ee8-1e7f-4562-9be0-25e3ae28b8f9/res.json')
round = 4 # round to save tokens
previous_iterations = 6
temperature = 0.1
model = 'gpt-3.5-turbo-0613'
data = {'previous_iterations':[{'x':list(np.round(data_full['data'][i]['inputs'],round)),'y':float(np.round(data_full['data'][i]['objective'],round))} for i in reversed(range(min(previous_iterations,len(data_full['data']))))]}
response = expert_reccomendation(x_names,x,u,data,expertise,obj_desc,model,temperature)
print(response)
choice = response.split('''"choice": ''')[-1].split(',')[0]