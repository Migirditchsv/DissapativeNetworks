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
## Safety triggers
popultion_limit = 10**5
time_limit = 100#**5
## Experimental parameters

#rescource volume regeneration and metabolism controls
# Regen and metabolism are seperated because we'll probably want to look at non
#-linear metabolic/regen scaling effects at some future point. 
source_initial_volume = 10**(8)
producer_initial_volume = 1.0
source_regen_ratio = 0.0
produce_regen_ratio = 0.0
consumer_regen_ratio = 0.0
source_metabolic_rate = 0.0
producer_metabolic_rate = 0.1
consumer_metabolic_rate = 0.1
DEATH_LIMIT = 0.1

#niche controls
niche_creep_rate = 1.1 # rate of increase in niche dist. mean per # of nodes

#Network Growth Controls
consumer_delay_time=100#How many time steps to wait until consumers appear
producer_spawn_ratio=0.01
consumer_spawn_ratio=0.1
producer_seed_number = 5

#Global indicies and trakcers
max_niche_score = 0.0
niche_array = []#useful for on the fly statistics. 
kill_list = []

# SVM 01/27: All node objects must be hashable for nice networkx features. 

### Useful Functions
def set_niche_score(node_index):
    global max_niche_score
    niche_score = np.random.uniform(0,niche_creep_rate*max_niche_score)
    niche_array.append(niche_score)
    if (niche_score>max_niche_score):
        max_niche_score=niche_score
    return( niche_score )

def create_source(node_index):
    #init node
    G.add_node(node_index)
    #set properties
    G.nodes[node_index]["node_index"]=node_index
    G.nodes[node_index]['role']="Source"
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
    G.nodes[node_index]["node_index"]=node_index
    G.nodes[node_index]["role"]="Producer"
    G.nodes[node_index]["volume"]= producer_initial_volume
    G.nodes[node_index]["niche_score"]= set_niche_score( node_index )
    ub = np.random.uniform(0,niche_creep_rate * max_niche_score)
    G.nodes[node_index]["niche_ub"]= ub
    G.nodes[node_index]["niche_lb"]= np.random.uniform(0,ub)
    G.nodes[node_index]["metabolic_ratio"]= producer_metabolic_rate
    G.nodes[node_index]["consumption_ratio"]= np.random.uniform(0,1)
    G.nodes[node_index]["regen_ratio"]= 0.0
    #find targets
    


def find_target( node_index ):
    ub = G.node[node_index]["niche_ub"]
    lb = G.node[node_index]["niche_lb"]
    best_score = 0.0
    best = -1
    for target in G.nodes:
        if target != node_index and target >0:
            target_niche_score = G.node[target]["niche_score"]
            if ( lb<=target_niche_score<=ub ):
                target_volume = G.node[target]["volume"]
                target_degree = G.degree(target)
                target_score = target_volume / max(1,target_degree)
                if ( target_score > best_score ):
                    best_score = target_score
                    best = target
    if (best > 0):
        G.add_edge(node_index,best)
                    
def kill_node( node_index ):
    global niche_array
    niche_score = G.node[node_index]["niche_score"]
    G.remove_node( node_index )
    niche_array.remove(niche_score)
        
    
def run_kill_list():
    global kill_list
    #remove duplicates
    kill_list = list(set(kill_list))
    for k in kill_list:
        kill_node(k)
    kill_list = []

def do_producer_step(node):
    global kill_list
    node_index = G.node[node]["node_index"]
    #iche_score = G.node[node]["niche_score"]
    
    targets = [ t for t in G.neighbors(node) ]
    target_number = len( targets )
    
    node_volume = G.node[node]["volume"]
    node_quota = G.node[node]["consumption_ratio"] * node_volume
    node_metabolism = G.node[node]["metabolic_ratio"] * node_volume
        
    
    # Die if starved
    if (node_volume<=DEATH_LIMIT):
        print( "t:",t,"node:", node,"fate: 0 starved\n")
        kill_list.append(node_index)
        return()
    
    # Hunger
    if (target_number>0):
        per_capita_quota = node_volume / target_number
        node_volume-=node_metabolism
        intake = 0.0
        for target in targets:
            target_index = G.node[target]["node_index"]
            target_volume = G.node[target]["volume"]
            if (per_capita_quota >= target_volume):
                intake += target_volume
                kill_list.append( target_index )
                print( "t:",t,"node:", node,"killed",target,"\n")
            else:
                intake += per_capita_quota
                target_volume += -per_capita_quota
                G.node[target]["volume"] = target_volume
        # eat gathered rescource
        G.node[node]["volume"] += intake
        # serch for new targets if under quota
        if (intake < node_quota):
            find_target( node_index )
    else: find_target(node_index)
    G.node[node_index]["volume"] -= node_metabolism

    
# Init the world
G = nx.DiGraph()

# Create source node & some seed producers
create_source(0)
for t in range(1, producer_seed_number):
    create_producer(t)
    #force into source niche range
    G.node[t]["niche_lb"]=0.0

### GRIND
run_condition=1
t=0
index_max = producer_seed_number
while (run_condition):

    # Update State Variables
    t+=1
    population = list( G.nodes() )
    population_size = len(population)
    
    # run through and-listed run conditions
    run_condition *= t<time_limit 
    run_condition *= population_size < popultion_limit
    run_condition *= population_size > 0 
    print( "TIME:", t,"| POPULATION:", population_size,"|",population)
    
    for node in G.nodes:
        if(G.node[node]["role"] == "Producer"):
            do_producer_step(node)
    
    # Reap nodes
    run_kill_list()
    # spawn Nodes
    spawn_number = m.ceil(population_size*producer_spawn_ratio)
    for spawn in range(0,spawn_number):
        index_max +=1
        create_producer(index_max)
        find_target(index_max)        

    
    


