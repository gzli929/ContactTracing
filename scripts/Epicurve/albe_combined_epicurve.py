import networkx as nx
import pandas as pd
from ctrace.runner import *
from ctrace.utils import load_graph_cville_labels, load_graph_montgomery_labels, read_extra_edges
from ctrace.simulation import *
from ctrace.recommender import *
from collections import namedtuple
from ctrace import PROJECT_ROOT

def run_full():
    I_full = {}
    for agent in [DegGreedy_fair, DepRound_fair, EC, NoIntervention]:
        print(agent.__name__)
        
        G = load_graph_cville_labels()
        pop_size = len(G.nodes)
    
        with open(PROJECT_ROOT/"data/SIR_Cache/albe.json", 'r') as infile:
            j = json.load(infile)
    
            (S, I1, I2, R) = (j["S"], j["I1"], j["I2"], j["R"])
            infections = [k/pop_size for k in j["infections"]]
    
        #G = read_extra_edges(G, 0.15)
        G.centrality = nx.algorithms.eigenvector_centrality_numpy(G)
    
        state = InfectionState(G, (S, I1, I2, R), 1250, "none", 0.05, True, -1, True, 1)
    
        while len(state.SIR.I1) + len(state.SIR.I2) != 0:
            to_quarantine = agent(state)
            state.step(to_quarantine)
            #print(len(state.SIR.I2))
            infections.append(len(state.SIR.I2)/pop_size)
    
        I_full[agent.__name__] = infections
    
    return I_full

def run_manual():
    I_manual = {}
    
    for agent in [SegDegree, Random, NoIntervention]:
        print(agent.__name__)
        
        G = load_graph_cville_labels()
        pop_size = len(G.nodes)
    
        with open(PROJECT_ROOT/"data/SIR_Cache/albe.json", 'r') as infile:
            j = json.load(infile)
    
            (S, I1, I2, R) = (j["S"], j["I1"], j["I2"], j["R"])
            infections = [k/pop_size for k in j["infections"]]
    
        #G = read_extra_edges(G, 0.15)
        
        state = InfectionState(G, (S, I1, I2, R), 1250, "none", 0.05, False, -1, False, 1)
    
        while len(state.SIR.I1) + len(state.SIR.I2) != 0:
            to_quarantine = agent(state)
            state.step(to_quarantine)
            infections.append(len(state.SIR.I2)/pop_size)
    
        I_manual[agent.__name__] = infections
        
    return I_manual

def run_digital():
    with open(PROJECT_ROOT/"output"/"albe_digital_epicurve_3.json") as f:
        I_digital = json.load(f)
    return I_digital

I_full = run_full()
I_digital = run_digital()
I_manual = run_manual()

max_timestep = max([len(i) for i in I_full.values()]+[len(i) for i in I_digital.values()]+[len(i) for i in I_manual.values()])

for key, l in I_manual.items():
    I_manual[key] = list(l) + [0]*(max_timestep-len(l))

for key, l in I_full.items():
    I_full[key] = list(l) + [0]*(max_timestep-len(l))

for key, l in I_digital.items():
    I_digital[key] = list(l) + [0]*(max_timestep-len(l))

with open(PROJECT_ROOT/"output"/"albe_combined_epicurve.json", 'w') as f:
    data = {"I_manual": I_manual, "I_full": I_full, "I_digital": I_digital}
    json.dump(data, f)