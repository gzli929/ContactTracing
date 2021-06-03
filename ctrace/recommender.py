import random
import math
import numpy as np
import networkx as nx

from ctrace.round import D_prime
from ctrace.utils import pct_to_int
from ctrace.simulation import *
from ctrace.problem import *

def NoIntervention(state: InfectionState):
    return set()

def Random(state: InfectionState):
    return set(random.sample(state.V1, min(state.budget, len(state.V1))))

def Degree(state: InfectionState):
    degrees: List[Tuple[int, int]] = []
    for u in state.V1:
        count = sum([1 for v in state.G.neighbors(u) if v in state.V2 and state.Q[u][v]!=0])
        degrees.append((count, u))
        
    degrees.sort(reverse=True)
    return {i[1] for i in degrees[:state.budget]}

def SegDegree(state: InfectionState, split_pcts=[0.75, 0.25], alloc_pcts=[.25, .75], carry=True, rng=np.random, DEBUG=False):
    """
    pcts are ordered from smallest degree to largest degree
    split_pcts: segment size percentages
    alloc_pcts: segment budget percentages
    Overflow Mechanic: the budget may exceed the segment size.
    We fill from right to left 
    (greater chance of overflow: larger degree usually have fewer members but higher budget), 
    and excess capacity is carried over to the next category.
    """
    if not math.isclose(1, sum(split_pcts)):
        raise ValueError(
            f"split_pcts '{split_pcts}' sum to {sum(split_pcts)}, not 1")
    if not math.isclose(1, sum(alloc_pcts)):
        raise ValueError(
            f"alloc_pcts '{alloc_pcts}' sum to {sum(alloc_pcts)}, not 1")

    budget = state.budget
    G = state.G
    split_amt = pct_to_int(len(state.V1), split_pcts)
    alloc_amt = pct_to_int(budget, alloc_pcts)

    v1_degrees = [(n, G.degree(n)) for n in state.V1]
    v1_sorted = [n for n, d in sorted(v1_degrees, key=lambda x: x[1])]

    v1_segments = np.split(v1_sorted, np.cumsum(split_amt[:-1]))

    overflow = 0
    samples = []
    for segment, amt in reversed(list(zip(v1_segments, alloc_amt))):
        # Overflow is carried over to the next segment
        segment_budget = amt
        if carry:
            segment_budget += overflow

        # Compute overflow
        if segment_budget > len(segment):
            overflow = segment_budget - len(segment)
            segment_budget = len(segment)
        else:
            overflow = 0

        sample = rng.choice(segment, segment_budget, replace=False)
        samples.extend(sample)

        if DEBUG:
            print(f"{segment_budget} / {len(segment)} (overflow: {overflow})")
            print("segment: ", segment)
            print("sample: ", sample)
            print("--------------")
            assert len(samples) <= budget
    return samples

def DegGreedy_fair(state: InfectionState):
    P, Q = state.P, state.Q
    
    weights: List[Tuple[int, int]] = []
    
    if state.compliance_known:
        for u in state.V1:
            w_sum = sum([Q[u][v] for v in state.G.neighbors(u) if v in state.V2])
            weights.append((state.G.nodes[u]['compliance_rate']*P[u]*(w_sum), u))
    else:
        for u in state.V1:
            w_sum = sum([Q[u][v] for v in state.G.neighbors(u) if v in state.V2])
            weights.append((state.P[u] * (w_sum), u))
    
    weights.sort(reverse=True)
    if (state.policy == "none"):
        return {i[1] for i in weights[:state.budget]}
    
    quarantine = set()
    state.set_budget_labels()
    for label in state.labels:
        deg = [tup for tup in weights if state.G.nodes[tup[1]]["age_group"]==label]
        quarantine = quarantine.union({i[1] for i in deg[:min(state.budget_labels[label], len(deg))]})
    return quarantine

def DepRound_fair(state: InfectionState):
    state.set_budget_labels()
    
    problem = MinExposedLP(state)
    problem.solve_lp()
    probabilities = problem.get_variables()
    
    if state.policy == "none":
        rounded = D_prime(np.array(probabilities))
        return set([problem.quarantine_map[k] for (k,v) in enumerate(rounded) if v==1])
    
    rounded = np.array([0 for i in range(len(probabilities))])
    for label in state.labels:
        partial_prob = [probabilities[k] if state.G.nodes[problem.quarantine_map[k]]["age_group"]==label else 0 for k in 
                        range(len(probabilities))]
        rounded = rounded + D_prime(np.array(partial_prob))

    return set([problem.quarantine_map[k] for (k,v) in enumerate(rounded) if v==1])

'''
For realistic model conditions where transmission rates are completely unknown.
The policymaker assumes transmission rates of 1.
'''
def DepRound_simplified(state: InfectionState):
    state.set_budget_labels()
    
    problem = MinExposedLP(state, simp = True)
    problem.solve_lp()
    probabilities = problem.get_variables()
    
    if state.policy == "none":
        rounded = D_prime(np.array(probabilities))
        return set([problem.quarantine_map[k] for (k,v) in enumerate(rounded) if v==1])
    
    rounded = np.array([0 for i in range(len(probabilities))])
    for label in state.labels:
        partial_prob = [probabilities[k] if state.G.nodes[problem.quarantine_map[k]]["age_group"]==label else 0 for k in 
                        range(len(probabilities))]
        rounded = rounded + D_prime(np.array(partial_prob))

    return set([problem.quarantine_map[k] for (k,v) in enumerate(rounded) if v==1])