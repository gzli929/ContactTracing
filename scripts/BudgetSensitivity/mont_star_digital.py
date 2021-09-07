#%%
import networkx as nx
import pandas as pd
from ctrace.runner import *
from ctrace.utils import load_graph_cville_labels, load_graph_montgomery_labels, read_extra_edges
from ctrace.simulation import *
from ctrace.recommender import *
from collections import namedtuple
import random #Default random number generator,
#random.SecureRandom() provides high-quality randomness from /dev/urandom or similar
from fractions import Fraction #we will work with rational numbers

json_dir = PROJECT_ROOT / "data" / "SIR_Cache"

G = load_graph_montgomery_labels()
G = read_extra_edges(G, 0.15)

#sample uniformly from range(m)
#all randomness comes from calling this
def sample_uniform(m,rng):
    assert isinstance(m,int) #python 3
    #assert isinstance(m,(int,long)) #python 2
    assert m>0
    return rng.randrange(m)

#sample from a Bernoulli(p) distribution
#assumes p is a rational number in [0,1]
def sample_bernoulli(p,rng):
    assert isinstance(p,Fraction)
    assert 0 <= p <= 1
    m=sample_uniform(p.denominator,rng)
    if m < p.numerator:
        return 1
    else:
        return 0

#sample from a Bernoulli(exp(-x)) distribution
#assumes x is a rational number in [0,1]
def sample_bernoulli_exp1(x,rng):
    assert isinstance(x,Fraction)
    assert 0 <= x <= 1
    k=1
    while True:
        if sample_bernoulli(x/k,rng)==1:
            k=k+1
        else:
            break
    return k%2

#sample from a Bernoulli(exp(-x)) distribution
#assumes x is a rational number >=0
def sample_bernoulli_exp(x,rng):
    assert isinstance(x,Fraction)
    assert x >= 0
    #Sample floor(x) independent Bernoulli(exp(-1))
    #If all are 1, return Bernoulli(exp(-(x-floor(x))))
    while x>1:
        if sample_bernoulli_exp1(Fraction(1,1),rng)==1:
            x=x-1
        else:
            return 0
    return sample_bernoulli_exp1(x,rng)

#sample from a geometric(1-exp(-x)) distribution
#assumes x is a rational number >= 0
def sample_geometric_exp_slow(x,rng):
    assert isinstance(x,Fraction)
    assert x >= 0
    k=0
    while True:
        if sample_bernoulli_exp(x,rng)==1:
            k=k+1
        else:
            return k
            
#sample from a geometric(1-exp(-x)) distribution
#assumes x >= 0 rational
def sample_geometric_exp_fast(x,rng):
    assert isinstance(x,Fraction)
    if x==0: return 0 #degenerate case
    assert x>0

    t=x.denominator
    while True:
        u=sample_uniform(t,rng)
        b=sample_bernoulli_exp(Fraction(u,t),rng)
        if b==1:
            break
    v=sample_geometric_exp_slow(Fraction(1,1),rng)
    value = v*t+u
    return value//x.numerator
    
#sample from a discrete Laplace(scale) distribution
#Returns integer x with Pr[x] = exp(-abs(x)/scale)*(exp(1/scale)-1)/(exp(1/scale)+1)
#casts scale to Fraction
#assumes scale>=0
def sample_dlaplace(scale,rng=None):
    if rng is None:
        rng = random.SystemRandom()
    scale = Fraction(scale)
    assert scale >= 0
    while True:
        sign=sample_bernoulli(Fraction(1,2),rng)
        magnitude=sample_geometric_exp_fast(1/scale,rng)
        if sign==1 and magnitude==0: continue
        return magnitude*(1-2*sign)
        
#compute floor(sqrt(x)) exactly
#only requires comparisons between x and integer
def floorsqrt(x):
    assert x >= 0
    #a,b integers
    a=0 #maintain a^2<=x
    b=1 #maintain b^2>x
    while b*b <= x:
        b=2*b #double to get upper bound
    #now do binary search
    while a+1<b:
        c=(a+b)//2 #c=floor((a+b)/2)
        if c*c <= x:
            a=c
        else:
            b=c
    #check nothing funky happened
    #assert isinstance(a,int) #python 3
    #assert isinstance(a,(int,long)) #python 2
    return a
    
#sample from a discrete Gaussian distribution N_Z(0,sigma2)
#Returns integer x with Pr[x] = exp(-x^2/(2*sigma2))/normalizing_constant(sigma2)
#mean 0 variance ~= sigma2 for large sigma2
#casts sigma2 to Fraction
#assumes sigma2>=0
def sample_dgauss(sigma2,rng=None):
    if rng is None:
        rng = random.SystemRandom()
    sigma2=Fraction(sigma2)
    if sigma2==0: return 0 #degenerate case
    assert sigma2 > 0
    t = floorsqrt(sigma2)+1
    while True:
        candidate = sample_dlaplace(t,rng=rng)
        bias=((abs(candidate)-sigma2/t)**2)/(2*sigma2)
        if sample_bernoulli_exp(bias,rng)==1:
            return candidate
        
#########################################################################
#DONE That's it! Now some utilities

import math #need this, code below is no longer exact

#Compute the normalizing constant of the discrete gaussian
#i.e. sum_{x in Z} exp(-x^2/2sigma2)
#By Poisson summation formula, this is equivalent to
# sqrt{2*pi*sigma2}*sum_{y in Z} exp(-2*pi^2*sigma2*y^2)
#For small sigma2 the former converges faster
#For large sigma2, the latter converges faster
#crossover at sigma2=1/2*pi
#For intermediate sigma2, this code will compute both and check
def normalizing_constant(sigma2):
    original=None
    poisson=None
    if sigma2<=1:
        original = 0
        x=1000 #summation stops at exp(-x^2/2sigma2)<=exp(-500,000)
        while x>0:
            original = original + math.exp(-x*x/(2.0*sigma2))
            x = x - 1 #sum from small to large for improved accuracy
        original = 2*original + 1 #symmetrize and add x=0
    if sigma2*100 >= 1:
        poisson = 0
        y = 1000 #summation stops at exp(-y^2*2*pi^2*sigma2)<=exp(-190,000)
        while y>0:
            poisson = poisson + math.exp(-math.pi*math.pi*sigma2*2*y*y)
            y = y - 1 #sum from small to large
        poisson = math.sqrt(2*math.pi*sigma2)*(1+2*poisson)
    if poisson is None: return original
    if original is None: return poisson
    #if we have computed both, check equality
    scale = max(1,math.sqrt(2*math.pi*sigma2)) #tight-ish lower bound on constant
    assert -1e-15*scale <= original-poisson <= 1e-15*scale
    #10^-15 is about as much precision as we can expect from double precision floating point numbers
    #64-bit float has 56-bit mantissa 10^-15 ~= 2^-50
    return (original+poisson)/2

#compute the variance of discrete gaussian
#mean is zero, thus:
#var = sum_{x in Z} x^2*exp(-x^2/(2*sigma2)) / normalizing_constant(sigma2)
#By Poisson summation formula, we have equivalent expression:
# variance(sigma2) = sigma2 * (1 - 4*pi^2*sigma2*variance(1/(4*pi^2*sigma2)) )
#See lemma 20 https://arxiv.org/pdf/2004.00010v3.pdf#page=17
#alternative expression converges faster when sigma2 is large
#crossover point (in terms of convergence) is sigma2=1/(2*pi)
#for intermediate values of sigma2, we compute both expressions and check
def variance(sigma2):
    original=None
    poisson=None
    if sigma2<=1: #compute primary expression
        original=0
        x = 1000 #summation stops at exp(-x^2/2sigma2)<=exp(-500,000)
        while x>0: #sum from small to large for improved accuracy
            original = original + x*x*math.exp(-x*x/(2.0*sigma2))
            x=x-1
        original = 2*original/normalizing_constant(sigma2)
    if sigma2*100>=1:
        poisson=0 #we will compute sum_{y in Z} y^2 * exp(-2*pi^2*sigma2*y^2)
        y=1000 #summation stops at exp(-y^2*2*pi^2*sigma2)<=exp(-190,000)
        while y>0: #sum from small to large
            poisson = poisson + y*y*math.exp(-y*y*2*sigma2*math.pi*math.pi)
            y=y-1
        poisson = 2*poisson/normalizing_constant(1/(4*sigma2*math.pi*math.pi))
        #next convert from variance(1/(4*pi^2*sigma2)) to variance(sigma2)
        poisson = sigma2*(1-4*sigma2*poisson*math.pi*math.pi)
    if original is None: return poisson
    if poisson is None: return original
    #if we have computed both check equality
    assert -1e-15*sigma2 <= original-poisson <= 1e-15*sigma2
    return (original+poisson)/2

def DegGreedy_private(state: InfectionState, epsilon: float):
    
    weights: List[Tuple[int, int]] = []
    
    for u in state.V1:
        noise = sample_dgauss(1/(epsilon))
        deg_noise = len(set(state.G.neighbors(u))) + noise
        #deg_noise = len(set(state.G.neighbors(u)))
        w_sum = state.transmission_rate * max(1, deg_noise)
        
        #infected_nbr_count = len(set(G.neighbors(u)) & set(state.SIR.I2))
        #probability_infected = 1 - math.pow(1-state.transmission_rate, 
        #                                    max(1, infected_nbr_count + sample_dgauss(1/(epsilon))))
        
        probability_infected = 1 - math.pow(1-state.transmission_rate, 
                                                        max(1, deg_noise))
        
        weights.append((probability_infected * (w_sum), u))

    weights.sort(reverse=True)
    #plt.hist(noises)
    
    return {i[1] for i in weights[:state.budget]}

def Degree_total_noisy(state: InfectionState, epsilon: float):
    degrees: List[Tuple[int, int]] = []
        
    noises = []
    
    for u in state.V1:
        noise = sample_dgauss(1/epsilon)
        noises.append(noise)
        count = max(1, len(set(state.G.neighbors(u))) + noise)
        degrees.append((count, u))
    
    degrees.sort(reverse=True)
    return {i[1] for i in degrees[:state.budget]}


def DegGreedy_private_V2(state: InfectionState):
    epsilon = 1
    
    weights: List[Tuple[int, int]] = []
    
    for u in state.V1:

        deg_noise_V2 = sum([1 for i in set(state.G.neighbors(u)) if i in state.V2 and state.Q[u][i]!=0])
        infected_nbr_count = len(set(G.neighbors(u)) & set(state.SIR.I2))
        
        deg_noise_V2 += sample_dlaplace(1/(math.sqrt(2)*epsilon/2))
        infected_nbr_count += sample_dlaplace(1/(math.sqrt(2)*epsilon/2))
        
        w_sum = state.transmission_rate * max(1, deg_noise_V2)
        
        probability_infected = 1 - math.pow(1-state.transmission_rate, 
                                            max(1, infected_nbr_count))
        
        weights.append((probability_infected * (w_sum), u))

    weights.sort(reverse=True)
    
    return {i[1] for i in weights[:state.budget]}

def Degree_V2_noisy(state: InfectionState):
    epsilon = 1
    degrees: List[Tuple[int, int]] = []
    
    for u in state.V1:
        noise = sample_dgauss(1/epsilon)
        count = max(1, sum([1 for i in set(state.G.neighbors(u)) if i in state.V2 and state.Q[u][i] != 0]) + noise)
        degrees.append((count, u))
    
    degrees.sort(reverse=True)
    return {i[1] for i in degrees[:state.budget]}


'''config_ep = {
    "G" : [G],
    "budget": [750],
    "policy": ["none"],
    "transmission_rate": [0.05],
    "transmission_known": [True],
    "compliance_rate": [-1],
    "compliance_known": [True],
    "snitch_rate": [1],
    "from_cache": ["mont.json"],
    "agent": [Degree_V2_noisy, DegGreedy_private_V2]
}'''

state = InfectionState(G, ([], [], [], []), 0, "none", 0, True, -1, True, 0)
compliances = nx.get_node_attributes(state.G, 'compliance_rate')
compliance_avg = sum(compliances.values())/(2*len(compliances.values()))
print(compliance_avg)

config = {
    "G" : [G],
    "budget": [i for i in range(math.floor(len(G.nodes)*0.001), math.floor(len(G.nodes)*0.05), int((math.floor(len(G.nodes)*0.05)-math.floor(len(G.nodes)*0.001))/86))],
    "policy": ["none"],
    "transmission_rate": [0.05],
    "transmission_known": [True],
    "compliance_rate": [compliance_avg],
    "compliance_known": [True],
    "snitch_rate": [1],
    "from_cache": ["mont_star.json"],
    "agent": [DegGreedy_private_V2]
}

in_schema = list(config.keys())
out_schema = ["infection_count", "infections_step"]
TrackerInfo = namedtuple("TrackerInfo", out_schema)

def time_trial_tracker(G: nx.graph, budget: int, policy:str, transmission_rate: float, transmission_known: bool, compliance_rate: float, compliance_known:bool, snitch_rate: float, from_cache: str, agent, **kwargs):

    with open(PROJECT_ROOT / "data" / "SIR_Cache" / from_cache, 'r') as infile:
            j = json.load(infile)

            (S, I1, I2, R) = (j["S"], j["I1"], j["I2"], j["R"])
            infections = j["infections"]
    
    state = InfectionState(G, (S, I1, I2, R), budget, policy, transmission_rate, transmission_known, compliance_rate, compliance_known, snitch_rate)
    
    while len(state.SIR.I1) + len(state.SIR.I2) != 0:
        to_quarantine = agent(state)
        state.step(to_quarantine)
        infections.append(len(state.SIR.I2))
    
    return TrackerInfo(len(state.SIR.R), infections)

run = GridExecutorParallel.init_multiple(config, in_schema, out_schema, func=time_trial_tracker, trials=10)
run.exec(max_workers=40)
#%%
