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

#Global indicies and trakcers
max_niche_score = 0.0
niche_array = []#useful for on the fly statistics. 


# SVM 01/27: All node objects must be hashable for nice networkx features. 

class Node:
    # Class Attributes
    role = 'node'
    
    # Initializer & attributes
    def __init__(self, node_index ):
        self.targets = []
        self.node_index = node_index
        self.volume = 0
        self.niche_score = -1
        self.niche_lb, self.niche_ub = (-1,-1)
        self.regen_ratio = 0
        self.metabolic_ratio = 0
        self.consumption_ratio = 0
    
    # Instance Fxns
    def set_niche_score(self, node_index):
        global max_niche_score
        niche_score = node_index * niche_creep_rate
        if (niche_score>max_niche_score):
            max_niche_score=niche_score
        return( niche_score )
    def rand_niche_bounds(self ):
        global max_niche_score
        mu = self.niche_score
        sigma = max_niche_score
        ub = np.random.uniform(0,max_niche_score)
        lb = np.random.uniform(0,ub)
        return(lb,ub)

        


### Useful Functions
def create_producer(node_index):
    new_producer = Node(node_index)
    new_producer.volume = producer_initial_volume
    new_producer.set_niche_score(node_index)
    new_producer.niche_lb,new_producer.niche_ub=new_producer.rand_niche_bounds()
    new_producer.metabolic_ratio = producer_metabolic_rate
    new_producer.consumption_ratio = np.random.uniform(0,1)
    niche_array.append(new_producer.niche_score)
    return( new_producer )
    
def node_update(node_name):
    return(0)
    
    
    
# Init the world
G = nx.DiGraph()

# Create source node
source = Node(0)
source.role="Source"
source.volume = source_initial_volume
source.niche_score = 0.0
niche_array.append[0.0]
G.add_node(source, source_attributes)

run_condition=1
t=0
while (run_condition):

    # Update State Variables
    t+=1
    population = G.size
    
    # run through and-listed run conditions
    run_condition *= t<time_limit 
    run_condition *= population < popultion_limit
    run_condition *= population > 0 
    
    


