import itertools
import numpy as np
from numba import jit

@jit(nopython = True)

#------------------------------------------------------------------------------functions

def distance(x,y):
    
    return np.sqrt(np.sum((x-y)**2))

def create_hidden_network(size):
    
    global position
    position = np.random.uniform(0,L,(N,2)) 
    global output_position
    output_position = np.array([L/2,L])
    global input_position
    input_position = np.array([[0,0],[0,L/3],[0,2*L/3],[0,L]])
    
    out_neighbors = [[] for i in range(size)]   
    g_out = [[] for i in range(size)]
    out_neighbors_inputs = [[] for i in range(len(inputs))]
    g_out_inputs = [[] for i in range(len(inputs))]
    for node in (hidden_network):
        p_d = np.random.exponential(d0 , kout)
        p_d = p_d * np.sqrt(2) / max(p_d)
        for dis in p_d:
            length = 10
            for neib in hidden_network:
                if (abs(distance(position[node],position[neib])-dis) < length
                    and neib not in out_neighbors[node]):
                    neighbor = neib
                    length = abs(distance(position[node],position[neib])-dis) 
            out_neighbors[node].append(neighbor)
            g_out[node].append(0.1)
    
    for inp in range(len(inputs)):
        n = []
        for neib in hidden_network:
            n.append(np.array([neib,distance(input_position[inp],position[neib])]))
        n = np.array(n)
        n = n[n[:,1].argsort()][:10]
        for i in n:
            out_neighbors_inputs[inp].append(int(i[0]))
            g_out_inputs[inp].append(1)
    
    n = []
    for neib in hidden_network:
        n.append(np.array([int(neib),distance(output_position,position[neib])]))
    n = np.array(n)
    n = n[n[:,1].argsort()][:10]
    for i in n:
        del out_neighbors[int(i[0])][-1]
        out_neighbors[int(i[0])].append(output)
        del g_out[int(i[0])][-1]
        g_out[int(i[0])].append(0.1)
    
    g_out = np.array(g_out)
    for i in range(size):
        g_out[i] = np.array(g_out[i])
    HN = {'weights' : g_out,
          'input weights' : g_out_inputs,
          'neighbors' : out_neighbors,
          'input neighbors' : out_neighbors_inputs}
    return HN 

def lead_to_critical_state(network , rule):
    
    out_fire = True
    
    while out_fire:
        all_fire = []
        for r in rule:
            
            v_last_HN = np.zeros(N)
            v_now_HN = np.zeros(N)
            v_last_inputs = np.zeros(len(inputs))
            eta_last = np.ones(N)
            eta_now = np.ones(N)
            v_output = 0
            v_last_inputs = r[:4]
            
            for t in itertools.count():
                x = 0
                if t == 0:
                    
                    for i in range(len(inputs)):
                        if v_last_inputs[i] >= vmax:
                            for j,j_id in enumerate(network['input neighbors'][i]):
                                v_now_HN[j_id] = v_last_HN[j_id] + 1
                    v_last_HN = np.copy(v_now_HN)
                    
                if t != 0 :
                    
                    fire = []
                    for i in (hidden_network):
                        if v_last_HN[i] >= vmax:
                            x += 1
                            eta_now[i] -= delta_eta
                            fire.append(i)
                            all_fire.append(i)
                            for j,j_id in enumerate(network['neighbors'][i]):
                                if j_id == 'out':
                                    v_output += network['weights'][i][j]*eta_last[i]
                                    if v_output >= 1 : out_fire = False
                                else:
                                    if v_last_HN[j_id] >= vmax: continue
                                    else: v_now_HN[j_id] += network['weights'][i][j]*eta_last[i]
                    
                    eta_now[eta_now < 0] = 0
                    eta_last = np.copy(eta_now)
                    v_now_HN[fire] = 0
                    v_last_HN = np.copy(v_now_HN)
                    
                if x == 0 and t != 0: break
            
        all_fire = list(set(all_fire))
        network['weights'][all_fire] += 0.001
        
        return network

def adaptation(network , rule):
    
    check = np.array([False for i in range(len(rule))])
    counter = 0
    while check.all() != True:
        
        output_disturb = np.zeros(len(rule))
        results = np.zeros(len(rule))
        
        for r_index,r in enumerate(rule):
                
            v_last_HN = np.zeros(N)
            v_now_HN = np.zeros(N)
            v_last_inputs = np.zeros(len(inputs))
            eta_last = np.ones(N)
            eta_now = np.ones(N)
            v_output = 0
            v_last_inputs = r[:4]
            rule_fire = []
            n_act = np.zeros(N) 
            
            for t in itertools.count():
                x = 0
                for i in network['input neighbors']:
                    v_now_HN[i] += dv                                              #self exite term causing synchronization
                if t == 0:
                    
                    for i in range(len(inputs)):
                        if v_last_inputs[i] >= vmax:
                            for j,j_id in enumerate(network['input neighbors'][i]):
                                v_now_HN[j_id] = v_last_HN[j_id] + 1
                    v_last_HN = np.copy(v_now_HN)
                    
                if t != 0 :
                    
                    fire = []
                    for i in (hidden_network):
                        if v_last_HN[i] >= vmax:
                            x += 1
                            eta_now[i] -= delta_eta
                            fire.append(i)
                            rule_fire.append(i)
                            n_act[i] += 1
                            for j_id,j in enumerate(network['neighbors'][i]):
                                if j == 'out':
                                    v_output += network['weights'][i][j_id]*eta_last[i]
                                    output_disturb[r_index] = 1
                                    if v_output >= vmax: results[r_index] = 1
                                else:
                                    if v_last_HN[j] >= vmax: continue
                                    else: v_now_HN[j] = v_last_HN[j] + network['weights'][i][j_id]*eta_last[i]
                    
                    eta_now[eta_now < 0] = 0
                    eta_last = np.copy(eta_now)
                    v_now_HN[fire] = 0
                    v_last_HN = np.copy(v_now_HN)
                # print(t,x)
                if (x == 0 and t != 0) or v_output >= vmax: break
            
            if results[r_index] == 0 and wanted[r_index] == 1:
                counter += 1
                # print(counter)
                for i in rule_fire:
                    for g_index,g in enumerate(network['weights'][i]):
                        if network['neighbors'][i][g_index] != 'out':
                            network['weights'][i][g_index] += alfa*g*n_act[i]
                    network['weights'][i][network['weights'][i] > g_max] = g_max                                                                                        
            elif results[r_index] == 1 and wanted[r_index] == 0:
                counter += 1
                # print(counter)
                for i in rule_fire:
                    for g_index,g in enumerate(network['weights'][i]):
                        if network['neighbors'][i][g_index] != 'out':
                            network['weights'][i][g_index] -= alfa*g*n_act[i]
                    # network['weights'][i][network['weights'][i] < 0] = 0   
            else: continue
        
        if sum(output_disturb) == 0 :
            network['weights'] += alfa*network['weights']

        check = results == wanted
        # print(len(check[check == True]) , counter)
        if counter >= Tmax: break

    if check.all() == True:
        return [1 , counter]
    else:
        return [0 , counter]

#------------------------------------------------------------------------------global variables

inputs = np.array(['in1','in2','in3','in4'])
output = 'out'
N = 250
L = (N/100)**0.5
kout = 10
hidden_network = np.arange(N)
d0 = 2
vmax = 1
delta_eta = 0.2
alfa = 0.01
g_max = 2
Tmax = 10000
rule_count = 9
dv = 0.01

#------------------------------------------------------------------------------input patterns and expected output

Rule = np.loadtxt('/home/farzad/Desktop/thesis/papers/xor_rule_paper.txt')
Rule = Rule[:rule_count]
wanted = Rule[:,4]

#-----------------------------------------------------------------------------

l = adaptation(lead_to_critical_state(create_hidden_network(N) , Rule) , Rule)
