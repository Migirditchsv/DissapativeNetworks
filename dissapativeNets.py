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
produce_growth_ratio = 1.12
consumer_growth_ratio = 1.10

# SVM 01/27: Making a big design decision here: When declaring nodes, special things
#must be done when defining nodes with unhashable objects describing them. I 
#don't think we'll need anything other than a set of scalars to describe
#each node, so I will be assuming that all node objects are hashable. 

class Node:
    
    # Class Attributes
    role = 'node'
    
    # Initializer & attributes
    def __init__(self, node_index, init_volume,
                 niche_score, niche_lb, niche_ub ):
        self.node_index = node_index
        self.volume = init_volume
        self.niche_score = niche_score
        self.niche_lb = niche_lb
        self.niche_ub = niche_ub
    
    # Instance Fxns
    def set_volume(self, new_volume):
        self.volume = new_volume
    def set_niche_score(self, new_niche_score):
        self.niche_score = new_niche_score
    def set_niche_lb(self, new_niche_lb):
        self.niche_lb = new_niche_lb
    def set_niche_ub(self, new_niche_ub):
        self.niche_ub = new_niche_ub
    
class Rescource(Node):
    
    # Class Attributes
    role = 'rescource'

    # Init
    def __init__(self,regen_rate):
        self.niche_score = 0.0
        self.niche_lb = 0.0
        self.niche_ub = big_num
        self.regen_rate = regen_rate

# Useful Functions
def node_update(node_name):
    return(0)
    
    
# Init the world
G = nx.MultiDiGraph
