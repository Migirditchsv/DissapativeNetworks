# -*- coding: utf-8 -*-
"""
How does capitalism end?
A simulation of dissapation driven percolation in competitive networks, and how
the developed structures decay once the dissapative rescource runs out.
Python 3.7
By Sam Migirditch and Thomas Varley
samigir@iu.edu

To Do:

1) Define Rescource, Producer and Consumer classes
2) Define base versions of functions for node spawning, prefferential
attachment, rescource consumption, rescource output and node death.
3) Define main loop.
4) Get on NPR
    
Change Log:
01/25 SVM: Created
"""

# Imports
import networkx as nx
import numpy as np
import math as m

# Globals
## Numeric control
epsilon = 10**-8
big_num = 10**8
## Safety triggers
popultion_limit = 10**5
time_limit = 10**5
## Experimental parameters

#rescource volume regeneration and metabolism controls
# Regen and metabolism are seperated because we'll probably want to look at non
#-linear metabolic/regen scaling effects at some future point. 
source_initial_volume = 10**(5)
producer_initial_volume = 1.0
source_regen_ratio = 0.0
produce_regen_ratio = 0.0
consumer_regen_ratio = 0.0
source_metabolic_rate = 0.0
producer_metabolic_rate = 0.1
consumer_metabolic_rate = 0.1

#niche controls
niche_creep_rate = 0.1 # rate of increase in niche dist. mean per # of nodes

#Network Growth Controls
consumer_delay_time=100#How many time steps to wait until consumers appear
producer_spawn_ratio=1.1
consumer_spawn_ratio=1.2

#Global indicies and trakcers
max_niche_score = 0.0
niche_array = []#useful for on the fly statistics. 


# SVM 01/27: All node objects must be hashable for nice networkx features. 

class Node:
    # Class Attributes
    role = 'node'
    
    # Initializer & attributes
    def __init__(self, node_index ):
        self.attributes= {
        "targets":[],
        "node_index" : node_index,
        "volume" : 0,
        "niche_score" : -1,
        "niche_lb" : -1,
        "niche_ub" : -1,
        "regen_ratio" : 0,
        "metabolic_ratio" : 0,
        "consumption_ratio" : 0
        }
        
### Useful Functions
def set_niche_score(node_index):
    global max_niche_score
    niche_score = node_index * niche_creep_rate
    niche_array.append(niche_score)
    if (niche_score>max_niche_score):
        max_niche_score=niche_score
    return( niche_score )

def create_source(node_index):
    #init node
    G.add_node(node_index)
    #set properties
    G.nodes[node_index]["role"]="Source"
    G.nodes[node_index]["volume"]= source_initial_volume
    G.nodes[node_index]["niche_score"]= 0
    G.nodes[node_index]["niche_lb"]= 0
    G.nodes[node_index]["niche_ub"]= 0
    G.nodes[node_index]["metabolic_ratio"]= 0
    G.nodes[node_index]["consumption_ratio"]= 0
    G.nodes[node_index]["regen_ratio"]= 0.0

def create_producer(node_index):
    #init node
    G.add_node(node_index)
    #set properties
    G.nodes[node_index]["role"]="Producer"
    G.nodes[node_index]["volume"]= producer_initial_volume
    G.nodes[node_index]["niche_score"]= set_niche_score( node_index )
    G.nodes[node_index]["niche_lb"]= np.random.uniform(0,ub)
    G.nodes[node_index]["niche_ub"]= np.random.uniform(0,max_niche_score)
    G.nodes[node_index]["metabolic_ratio"]= producer_metabolic_rate
    G.nodes[node_index]["consumption_ratio"]= np.random.uniform(0,1)
    G.nodes[node_index]["regen_ratio"]= 0.0
    #find targets
    


def find_targets( node_index ):
    ub = 0
    
def kill_node( node_index ):
    node_index = str( node_index )
    node = G.node( node_index )
    node.attributes["niche_score"]
    G.remove_node( node_index )
    niche_array.remove(niche_score)
    
    
    
    
# Init the world
G = nx.DiGraph()

# Create source node
create_source(0)


### GRIND
run_condition=1
t=0
while (run_condition):

    # Update State Variables
    t+=1
    population = list( G.nodes() )
    population_size = len(population)
    
    # run through and-listed run conditions
    run_condition *= t<time_limit 
    run_condition *= population_size < popultion_limit
    run_condition *= population_size > 0 
    print( "TIME:", t,"| POPULATION:", population)
    
    for node in G.nodes:
        
        node_index = G.node[node]["node_index"]
        niche_score = node.attributes["niche_score"]
        
        targets = [ t for t in G.neighbors(node) ]
        target_number = len( targets )
        
        node_volume = node.attributes["volume"]
        node_quota = node.attributes["consumption_ratio"] * node_volume
        node_metabolism = node.attributes["metabolic_ratio"] * node_volume
        
        # Die if starved
        if (node_metabolism>=node_volume):
            kill_node( node_index )
            continue
        
        # Hunger
        per_capita_quota = node_volume / target_number
        node_volume-=node_metabolism
        intake = 0.0
        for target in targets:
            target_index = target.attributes["node_index"]
            target_volume = target.attributes["volume"]
            if per_capita_quota >= target_volume:
                intake += target_volume
                kill_node( target_index )
            
        # Spawn
    
    


