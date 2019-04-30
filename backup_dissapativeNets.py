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
import statistics as stats
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
import math as m
import scipy.stats as st
import scipy.signal as ss
import os
from datetime import datetime

#Misc path stuff
path = os.getcwd()
time = datetime.now()
target_dir = "run_" + str(time.hour) + ":" + str(time.minute) + "." + str(time.second) + "/"
os.mkdir(path + "/" + target_dir)
os.mkdir(path + "/" + target_dir + "plots")

imgdir = path + "/" + target_dir + "plots/"

# Globals
## Safety triggers
popultion_limit = 10**8
time_limit = 10**2
## Experimental parameters

#rescource volume regeneration and metabolism controls

# Regen and metabolism are seperated because we'll probably want to look at non
#-linear metabolic/regen scaling effects at some future point. 
source_initial_volume = 10**8
producer_initial_volume = 1
source_regen_ratio = 0
produce_regen_ratio = 0.0
consumer_regen_ratio = 0.0
source_metabolic_rate = 0.0
producer_metabolic_ratio = 0.005
#consumer_metabolic_ratio = 0.1
producer_consumption_ratio = 0.01
DEATH_LIMIT = 0.01

#niche controls
niche_creep_rate = 0.01 # rate of increase in niche dist. mean per # of nodes

#Network Growth Controls
consumer_delay_time=100#How many time steps to wait until consumers appear
producer_spawn_ratio= 0.1 # Comment out #spawn section of grind and set to cnst if you don't want exp growth
producer_seed_number = 10

#Plotting controls
scale_factor = 200.0 # scales node volumes
plot_frequency = 1
savefigures = True #Toggle to not save figures. 


#Global indicies and trakcers
max_niche_score = 0.0 
niche_list = []
kill_list = []

# SVM 01/27: All node objects must be hashable for nice networkx features. 

### Useful Functions
def set_niche_score_2():
    global max_niche_score
    moment = []
    total_volume = 0
    for node_index in G.nodes:
        role = G.node[node_index]["role"]
        #if role != "Producer": continue
        try:
            niche_score = G.node[node_index]["niche_score"]
            volume = G.node[node_index]["volume"]
            moment.append( volume * niche_score )
            total_volume += volume
        except: continue
    if len(moment)<2: return( niche_creep_rate,0,niche_creep_rate)
    sd = stats.stdev( moment )
    moment = stats.mean(moment)
    lb = moment - (niche_creep_rate * sd)
    lb = max(0, lb)
    ub = moment + ( niche_creep_rate * sd)
    ub = min( max_niche_score, ub)
    return(moment,lb,ub)
    
def set_niche_score(node_index):
    global max_niche_score
    niche_score = first_moment() + niche_creep_rate
    #print("SETTING NICHE:", niche_score)
    update_niche_stats
    niche_list.append(niche_score)
    return( niche_score )
    
def update_niche_list():
    global niche_list, max_niche_score
    niche_list = []
    for node_index in G.nodes:
        niche_score = G.node[node_index]["niche_score"]
        niche_list.append(niche_score)
        if (niche_score > max_niche_score):
            max_niche_score = niche_score

# For updating individual w/0 updating all
def update_niche_stats(node_index):
    global max_niche_score, niche_list
    niche_score = G.node[node_index]["niche_score"]
    if niche_score > max_niche_score: max_niche_score = niche_score


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
    update_niche_list()

def create_producer(node_index):
    #init node
    G.add_node(node_index)
    #set properties
    G.nodes[node_index]["node_index"]=node_index
    G.nodes[node_index]["role"]="Producer"
    G.nodes[node_index]["volume"]= producer_initial_volume 
    niche, lb, ub = set_niche_score_2()
    G.nodes[node_index]["niche_score"]= niche
    G.nodes[node_index]["niche_ub"]= ub
    G.nodes[node_index]["niche_lb"]= 0.01*ub
    G.nodes[node_index]["metabolic_ratio"]= producer_metabolic_ratio
    G.nodes[node_index]["consumption_ratio"]= producer_consumption_ratio/(ub)
    G.nodes[node_index]["regen_ratio"]= 0.0
    #update trackers
    update_niche_list()
    #find targets
    find_target( node_index )

def find_target( node_index ):
    ub = G.node[node_index]["niche_ub"]
    lb = G.node[node_index]["niche_lb"]
    best_score = 0.0
    best = -1
    possible_targets = list(G.nodes())
    np.random.shuffle(possible_targets)
    for target in possible_targets:
        # don't make parallel or self loops
        blacklist = [node_index]
        edges = G.out_edges(node_index)
        blacklist.extend(edges)
        if target not in blacklist:
            target_niche_score = G.node[target]["niche_score"]
            if ( lb<=target_niche_score<=ub ):
                target_volume = G.node[target]["volume"]
                target_degree = G.degree(target)
                target_score = target_volume / max(1,target_degree)
                if ( target_score > best_score ):
                    best_score = target_score
                    best = target
    if (best_score > 0):
        G.add_edge(node_index,best)
    #else: print( "EDGE:",node_index," failed to find target")
    
def force_connect_source(target):
    if G.node[target]["role"]=="Source": return
    if 0 not in G.nodes: return
    G.add_edge(target,0)
                    
def kill_node( node_index ):
    #global niche_array
    #niche_score = G.node[node_index]["niche_score"]
    G.remove_node( node_index )
    #niche_array.remove(niche_score)

def remove_isolates():
    global kill_list
    isolates = nx.isolates(G)
    kill_list.extend(isolates)
    
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
        #print( "t:",t,"node:", node,"fate: 0 starved\n")
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
                #print( "t:",t,"node:", node,"killed",target,"\n")
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


def change_niche_score(node_index, new_score):
    global max_niche_score, niche_list
    #swap
    old_score = G.node[node_index]["niche_score"]
    G.node[node_index]["niche_score"] = new_score
    #update stats
    niche_list.remove(old_score)
    niche_list.append(new_score)
    if (new_score > max_niche_score):
        max_niche_score = new_score
    
    
def plotter(target_dir, show = True):
    pos=nx.spring_layout(G) # positions for all nodes
    labels={}
    max_volume = 0.0
    for node in G.nodes:
        role = G.node[node]["role"]
        if(role=="Source"): continue
        volume = G.node[node]["volume"]
        if volume>max_volume:
            max_volume = volume

# Draw nodes
    for node_index in G.nodes:
        # Type-> color
        role = G.node[node_index]["role"]
        if (role=="Source"):
            color = "green"
        elif (role=="Producer"):
            color = "cornflowerblue"
        elif (role=="consumer"):
            color = "red"
        else: 
            color = "black"
        # volume -> size
        volume = scale_factor*min(1,G.node[node_index]["volume"] / max_volume )
        
        #Draw node
        nx.draw_networkx_nodes(G,pos,
                               nodelist=[node_index],
                               node_color= color,
                               node_size=volume,
                           alpha=0.8) 
        # draw edges
        edges = G.out_edges(node_index)
        nx.draw_networkx_edges(G,pos,
                               edgelist=edges,
                               width=1,
                               alpha=0.5,
                               edge_color=color)
        
        # some math labels
        labels[node_index]=node_index
        
    nx.draw_networkx_labels(G,pos,labels,font_size=16)

    plt.axis('off')
    local_now = datetime.now()
    #plt.savefig(target_dir + str(local_now) + "_graph.png") # save as png
   
    plt.show() # display
    #nx.write_gexf(G, target_dir + str(local_now) + "_graph.gexf")
    
### Script   
# Init the world
G = nx.DiGraph()

# Create source node & some seed producers
create_source(0)

for t in range(1, producer_seed_number):
    create_producer(t)
    #force into source niche range
    G.node[t]["niche_lb"]=0.0
    G.node[t]["niche_ub"]=niche_creep_rate
    change_niche_score(t,0.1*niche_creep_rate)

### GRIND

run_condition=1
t=0
index_max = producer_seed_number

#Various graph property time-series
num_producers = []
num_nodes = []
size_source = []
total_volumes = []
min_niche_lb = []
max_niche_ub = []
num_components = []
global_eff = []
local_eff = []
mean_degree = []
std_degree = []
var_degree = []
ent_degree = []
density = []
largest_comp = []
avg_clustering_coeff = []
avg_harmonic_centrality = []
alg_conn = []
alg_conn_norm = []

node_lifetime = {}
avg_node_clustering_coeff = {}
avg_node_harmonic_centrality = {}
avg_in_degree_centrality = {}
avg_out_degree_centrality = {}
avg_page_rank = {}
avg_volume = {}

ticker = 0

while (run_condition):

    # Update State Variables
    num_producers.append(len([x for x in G.nodes() if G.node[x]["role"] == "Producer"]))
    num_nodes.append(len(G.nodes()))
    
    undir_G = nx.to_undirected(G)
    
    if 0 in G.nodes:
        size_source.append(G.node[0]["volume"])
    
    
    #Density of the graph
    prob = len(G.edges) / ((len(G.nodes)*(len(G.nodes)-1))/2) 
    density.append(prob)
    
    #Useful for finding peaks near 10 and when the resource runs out
    peaks = ss.find_peaks(num_nodes)[0]
    
    #Get volumes of each node as dict
    volumes = nx.get_node_attributes(G, "volume")
    
    #Dictionaries of graph properties
    clustering_coeff = nx.clustering(G)
    harmonic_centrality = nx.harmonic_centrality(G)
    degs = nx.degree(G)
    in_deg_cent = nx.in_degree_centrality(G)
    out_deg_cent = nx.out_degree_centrality(G)
    
    #In and out degree seqs
    in_deg_sequence = list(x[1] for x in G.in_degree())
    out_deg_sequence = list(x[1] for x in G.out_degree())
    
    config_global_eff_list, config_local_eff_list, config_avg_clustering_coeff_list, = [], [], []
    config_avg_harmonic_centrality_list, config_alg_conn_list = [], []
    
    #Making configuraiton models 
    """
    for i in range(1):
        config_G = nx.directed_configuration_model(in_deg_sequence, out_deg_sequence)
        undir_config_G = nx.Graph(nx.to_undirected(config_G))
        
        config_global_eff_list.append(nx.global_efficiency(undir_config_G))
        config_local_eff_list.append(nx.local_efficiency(undir_config_G))
        config_avg_clustering_coeff_list.append(nx.average_clustering(undir_config_G))
        config_avg_harmonic_centrality_list.append(config_G)
        config_alg_conn_list.append(undir_config_G)
    """
    #Calculating average config statistics for summary stats. 
    config_global_eff = 1#np.mean(config_global_eff_list)
    config_local_eff = 1#np.mean(config_local_eff_list)
    config_avg_clustering_coeff = 1#np.mean(config_avg_clustering_coeff_list)
    config_avg_harmonic_centrality = 1#np.mean(config_avg_harmonic_centrality_list)
    config_alg_conn = 1#np.mean(config_alg_conn_list)
    
    #Calculating general model descriptors
    total_volumes.append(np.sum([G.node[i]["volume"] for i in G.nodes() if i != 0]))
    min_niche_lb.append(min([G.node[i]["niche_lb"] for i in G.nodes() ]))
    max_niche_ub.append(max([G.node[i]["niche_ub"] for i in G.nodes() ]))
    
    #Degree-distrubtion-specific graph measures. 
    mean_degree.append(np.mean([x[1] for x in degs if x[0] != 0]))
    std_degree.append(np.std([x[1] for x in degs if x[0] != 0]))
    var_degree.append(np.var([x[1] for x in degs if x[0] != 0]))
    ent_degree.append(st.entropy([x[1] for x in degs if x[0] != 0])/len(G))
    num_components.append(len(list(nx.connected_component_subgraphs(undir_G))))
    largest_comp.append(len(max(nx.connected_component_subgraphs(undir_G), key=len)) / len(undir_G))
    
    #Measures normalized by config_models.
    global_eff.append(nx.global_efficiency(undir_G) / config_global_eff)
    local_eff.append(nx.local_efficiency(undir_G) / config_local_eff)
    avg_clustering_coeff.append(nx.average_clustering(G) / config_avg_clustering_coeff)
    avg_harmonic_centrality.append(np.mean([harmonic_centrality[i] for i in G.nodes()]) / config_avg_harmonic_centrality)
    alg_conn_norm.append(nx.algebraic_connectivity(undir_G) / config_alg_conn)
    alg_conn.append(nx.algebraic_connectivity(undir_G))
    
    for i in G.nodes():
        if i != 0:
            
            if i in node_lifetime:
                node_lifetime[i] += 1
            elif i not in node_lifetime:
                node_lifetime[i] = 1
            
            if i in avg_node_clustering_coeff:
                avg_node_clustering_coeff[i] = avg_node_clustering_coeff[i] + (clustering_coeff[i] - avg_node_clustering_coeff[i])/ticker
            elif i not in avg_node_clustering_coeff:
                avg_node_clustering_coeff[i] = clustering_coeff[i]
            
            if i in avg_node_harmonic_centrality:
                avg_node_harmonic_centrality[i] = avg_node_harmonic_centrality[i] + (harmonic_centrality[i] - avg_node_harmonic_centrality[i])/ticker
            elif i not in avg_node_harmonic_centrality:
                avg_node_harmonic_centrality[i] = harmonic_centrality[i]
                
            if i in avg_in_degree_centrality:
                avg_in_degree_centrality[i] = avg_in_degree_centrality[i] + (in_deg_cent[i] - avg_in_degree_centrality[i])/ticker
            elif i not in avg_in_degree_centrality:
                avg_in_degree_centrality[i] = in_deg_cent[i]
            
            if i in avg_out_degree_centrality:
                avg_out_degree_centrality[i] = avg_out_degree_centrality[i] + (out_deg_cent[i] - avg_out_degree_centrality[i])/ticker
            elif i not in avg_out_degree_centrality:
                avg_out_degree_centrality[i] = out_deg_cent[i]
                
            if i in avg_volume:
                avg_volume[i] = avg_volume[i] + (volumes[i] - avg_volume[i])/ticker
            elif i not in avg_volume:
                avg_volume[i] = volumes[i]
                
    ticker += 1
    
    t+=1
    population = list( G.nodes() )
    population_size = len(population)
    
    # run through and-listed run conditions
    run_condition *= t<time_limit 
    run_condition *= population_size < popultion_limit
    run_condition *= population_size > 3
    #print( "TIME:", t,"| POPULATION:", population_size,"|",population)
    
    
    update_list = list(G.nodes())
    np.random.shuffle(update_list)
    for node in update_list:
        if(G.node[node]["role"] == "Producer"):
            update_niche_stats(node)
            do_producer_step(node)
    
    # Reap nodes
    #remove_isolates()
    run_kill_list()
    #plot
    if t%plot_frequency ==0:
        plotter(target_dir)
    # spawn Nodes
    
    #skip spawn if rescource has died
    if 0 not in G.nodes: continue
    # force connect random node to seed node if disconnected
    random_nodes = list(G.nodes())
    np.random.shuffle(random_nodes)
    condition = nx.degree(G)[0] <= producer_seed_number
    while condition > 0:
        target = random_nodes.pop(0)
        force_connect_source(target)
        cond_1 = nx.degree(G)[0] <= producer_seed_number
        cond_2 = len(random_nodes) > 0
        condition = cond_1 * cond_2
        
    
    spawn_number = m.ceil(population_size*producer_spawn_ratio)
    for spawn in range(0,spawn_number):
        create_producer(index_max)
        find_target(index_max)
        index_max +=1
        

timeline = [x for x in range(ticker)]

sns.set(style = "darkgrid")


plt.subplots()
plt.plot(size_source)
plt.xlabel("Time")
plt.ylabel("Source Volume")
plt.title("Source Volume Over Time")
if savefigures == True:
    plt.savefig(imgdir + "volume_time.png", dpi = 250)

plt.subplots()
plt.plot(min_niche_lb, label = "Min Niche Score")
plt.plot(max_niche_ub, label = "Max Niche Score")
plt.xlabel("Time")
plt.ylabel("Min/Max Niche Score")
plt.yscale("log")
plt.title("Niche Range")
plt.legend()
if savefigures == True:
    plt.savefig(imgdir + "niche.png", dpi = 250)

plt.subplots()
plt.plot(num_nodes)
plt.plot(timeline[peaks[0]], num_nodes[peaks[0]],
         marker="*",
         color="black",
         markersize="10",
         label="First Die Off")
plt.plot(timeline[peaks[1]], num_nodes[peaks[1]],
         marker="X",
         color="black",
         markersize="10",
         label="Exhaust Resource")
plt.xlabel("Time")
plt.ylabel("Population")
plt.title("Population Over Time")
plt.legend()
if savefigures == True:
    plt.savefig(imgdir + "population.png", dpi = 250)


plt.subplots()
plt.plot(num_components)
plt.plot(timeline[peaks[0]], num_components[peaks[0]],
         marker="*",
         color="black",
         markersize="10",
         label="First Die Off")
plt.plot(timeline[peaks[1]], num_components[peaks[1]],
         marker="X",
         color="black",
         markersize="10",
         label="Exhaust Resource")
plt.xlabel("Time")
plt.ylabel("Count")
plt.title("Number of Connected Components")
if savefigures == True:
    plt.savefig(imgdir + "num_components.png", dpi = 250)

plt.subplots()
plt.plot(largest_comp)
plt.plot(timeline[peaks[0]], largest_comp[peaks[0]],
         marker="*",
         color="black",
         markersize="10",
         label="First Die Off")
plt.plot(timeline[peaks[1]], largest_comp[peaks[1]],
         marker="X",
         color="black",
         markersize="10",
         label="Exhaust Resource")
plt.xlabel("Time")
plt.ylabel("Largest Component Percentage")
plt.title("Largest Component")
if savefigures == True:
    plt.savefig(imgdir + "largest_component.png", dpi = 250)

plt.subplots()
plt.plot(global_eff)
plt.plot(timeline[peaks[0]], global_eff[peaks[0]],
         marker="*",
         color="black",
         markersize="10",
         label="First Die Off")
plt.plot(timeline[peaks[1]], global_eff[peaks[1]],
         marker="X",
         color="black",
         markersize="10",
         label="Exhaust Resource")
plt.xlabel("Time")
plt.ylabel("Global Efficiency")
plt.title("Global Efficiency")# (Normalized w/ Config Model)")
plt.legend()
if savefigures == True:
    plt.savefig(imgdir + "global_efficiency.png", dpi = 250)

plt.subplots()
plt.plot(local_eff)
plt.plot(timeline[peaks[0]], local_eff[peaks[0]],
         marker="*",
         color="black",
         markersize="10",
         label="First Die Off")
plt.plot(timeline[peaks[1]], local_eff[peaks[1]],
         marker="X",
         color="black",
         markersize="10",
         label="Exhaust Resource")
plt.xlabel("Time")
plt.ylabel("Local Efficiency")
plt.title("Local Efficiency")# (Normalized w/ Config Model)")
plt.legend()
if savefigures == True:
    plt.savefig(imgdir + "local_efficiency.png", dpi = 250)

plt.subplots()
plt.plot(avg_harmonic_centrality)
plt.plot(timeline[peaks[0]], avg_harmonic_centrality[peaks[0]],
         marker="*",
         color="black",
         markersize="10",
         label="First Die Off")
plt.plot(timeline[peaks[1]], avg_harmonic_centrality[peaks[1]],
         marker="X",
         color="black",
         markersize="10",
         label="Exhaust Resource")
plt.xlabel("Time")
plt.ylabel("Average Harmonic Centrality")
plt.title("Average Harmonic Centrality")# (Normalized w/ Config Model)")
plt.legend()
if savefigures == True:
    plt.savefig(imgdir + "harmonic_centrality.png", dpi = 250)

plt.subplots()
plt.plot(avg_clustering_coeff)
plt.plot(timeline[peaks[0]], avg_clustering_coeff[peaks[0]],
         marker="*",
         color="black",
         markersize="10",
         label="First Die Off")
plt.plot(timeline[peaks[1]], avg_clustering_coeff[peaks[1]],
         marker="X",
         color="black",
         markersize="10",
         label="Exhaust Resource")
plt.xlabel("Time")
plt.ylabel("Average Clustering Coefficint")
plt.title("Average Clustering Coefficient")# (Normalized w/ Config Model)")
plt.legend()
if savefigures == True:
    plt.savefig(imgdir + "clustering_coeff.png", dpi = 250)

plt.subplots()
plt.plot(mean_degree, label = "Mean Degree")
plt.plot([mean_degree[x] + std_degree[x] for x in range(len(mean_degree))],
          color = "darkgreen",
          linestyle = "--", 
          linewidth = 1,
          label = "Standard Deviation of Degree")
plt.plot([mean_degree[x] - std_degree[x] for x in range(len(mean_degree))],
          color = "darkgreen",
          linestyle = "--",
          linewidth = 1)
plt.plot(timeline[peaks[0]], mean_degree[peaks[0]],
         marker="*",
         color="black",
         markersize="10",
         label="First Die Off")
plt.plot(timeline[peaks[1]], mean_degree[peaks[1]],
         marker="X",
         color="black",
         markersize="10",
         label="Exhaust Resource")
plt.xlabel("Time")
plt.ylabel("Mean Degree")
plt.title("Mean Degree")
plt.legend()
if savefigures == True:
    plt.savefig(imgdir + "mean_std_degree.png", dpi = 250)

plt.subplots()
plt.plot(var_degree)
plt.plot(timeline[peaks[0]], var_degree[peaks[0]],
         marker="*",
         color="black",
         markersize="10",
         label="First Die Off")
plt.plot(timeline[peaks[1]], var_degree[peaks[1]],
         marker="X",
         color="black",
         markersize="10",
         label="Exhaust Resource")
plt.xlabel("Time")
plt.ylabel("Degree Variance")
plt.title("Variance in Degree Distribution")
plt.legend()
if savefigures == True:
    plt.savefig(imgdir + "variance_degree.png", dpi = 250)

plt.subplots()
plt.plot(ent_degree)
plt.plot(timeline[peaks[0]], ent_degree[peaks[0]],
         marker="*",
         color="black",
         markersize="10",
         label="First Die Off")
plt.plot(timeline[peaks[1]], ent_degree[peaks[1]],
         marker="X",
         color="black",
         markersize="10",
         label="Exhaust Resource")
plt.xlabel("Time")
plt.ylabel("Entropy")
plt.title("Entropy of Degree Distribution")
plt.legend()
if savefigures == True:
    plt.savefig(imgdir + "entropy_degree.png", dpi = 250)

plt.subplots()
plt.plot(density)
plt.plot(timeline[peaks[0]], density[peaks[0]],
         marker="*",
         color="black",
         markersize="10",
         label="First Die Off")
plt.plot(timeline[peaks[1]], density[peaks[1]],
         marker="X",
         color="black",
         markersize="10",
         label="Exhaust Resource")
plt.xlabel("Time")
plt.ylabel("Density")
plt.title("Graph Density Over Time")
plt.legend()
if savefigures == True:
    plt.savefig(imgdir + "density.png", dpi = 250)

plt.subplots()
plt.scatter(list(node_lifetime.keys()), list(node_lifetime.values()),
            c = list(node_lifetime.keys()), cmap = "winter", label = "Creation Order")
plt.xlabel("Node Creation Order")
plt.ylabel("Node Lifetime")
plt.title("Creation Order vs. Node Lifetime")
plt.colorbar(label = "Creation Order")
if savefigures == True:
    plt.savefig(imgdir + "creation_order_lifetimes.png", dpi = 250)

plt.subplots()
plt.scatter(list(avg_node_clustering_coeff.values()), list(node_lifetime.values()),
            c = list(node_lifetime.keys()), cmap = "winter", label = "Creation Order")
plt.xlabel("Average Clustering Coefficient")
plt.ylabel("Node Lifetime")
plt.title("Clustering Coefficient vs. Node Lifetime")
plt.colorbar(label = "Creation Order")
if savefigures == True:
    plt.savefig(imgdir + "clustering_lifetimes.png", dpi = 250)

plt.subplots()
plt.scatter(list(avg_node_harmonic_centrality.values()), list(node_lifetime.values()),
            c = list(node_lifetime.keys()), cmap = "winter", label = "Creation Order")
plt.xlabel("Average Harmonic Centrality")
plt.ylabel("Node Lifetime")
plt.title("Harmonic Centrality vs. Node Lifetime")
plt.colorbar(label = "Creation Order")
if savefigures == True:
    plt.savefig(imgdir + "harmonic_lifetimes.png", dpi = 250)

plt.subplots()
plt.scatter(list(avg_in_degree_centrality.values()), list(node_lifetime.values()),
            c = list(node_lifetime.keys()), cmap = "winter", label = "Creation Order")
plt.xlabel("Average In-Degree Centrality")
plt.ylabel("Node Lifetime")
plt.title("In-Degree Centrality vs. Node Lifetime")
plt.colorbar(label = "Creation Order")
if savefigures == True:
    plt.savefig(imgdir + "in_degree_lifetimes.png", dpi = 250)

plt.subplots()
plt.scatter(list(avg_out_degree_centrality.values()), list(node_lifetime.values()),
            c = list(node_lifetime.keys()), cmap = "winter", label = "Creation Order")
plt.xlabel("Average Out-Degree Centrality")
plt.ylabel("Node Lifetime")
plt.title("Out-Degree Centrality vs. Node Lifetime")
plt.colorbar(label = "Creation Order")
if savefigures == True:
    plt.savefig(imgdir + "out_degree_lifetimes.png", dpi = 250)

plt.subplots()
plt.scatter(list(avg_volume.values()), list(node_lifetime.values()),
            c = list(node_lifetime.keys()), cmap = "winter", label = "Creation Order")
plt.xlabel("Average Node Volume")
plt.ylabel("Node Lifetime")
plt.colorbar(label = "Creation Order")
plt.title("Average Node Volume vs. Node Lifetime")
if savefigures == True:
    plt.savefig(imgdir + "volume_lifetimes.png", dpi = 250)

plt.subplots()
plt.plot(alg_conn)
plt.plot(timeline[peaks[0]], alg_conn_norm[peaks[0]],
         marker="*",
         color="black",
         markersize="10",
         label="First Die Off")
plt.plot(timeline[peaks[1]], alg_conn_norm[peaks[1]],
         marker="X",
         color="black",
         markersize="10",
         label="Exhaust Resource")
plt.xlabel("Time")
plt.ylabel("Algebraic Connectivity")
plt.title("Algebraic Connectivity")#(Normalized w/ Config Model)")
plt.legend()
#plt.ylim([-0.01,0.25])
if savefigures == True:
    plt.savefig(imgdir + "alg_conn.png", dpi = 250)

plt.subplots()
plt.scatter(ent_degree, var_degree, c = timeline, cmap = "autumn_r")
plt.plot(ent_degree[peaks[0]], var_degree[peaks[0]], 
         marker="*", 
         color = "black", 
         markersize = 10, label = "First Die-Off")
plt.plot(ent_degree[peaks[1]], var_degree[peaks[1]], 
         marker="X", 
         color = "black", 
         markersize = 10, label = "Exhaust Resource")
plt.xlabel("Entropy of Degree Distribution")
plt.ylabel("Variance in Degree Distribution")
plt.title("Entropy vs. Variance Over Time")
plt.colorbar(label = "Time" )
plt.legend()
if savefigures == True:
    plt.savefig(imgdir + "ent_var_degree.png", dpi = 250)

plt.subplots()
plt.plot(ent_degree[peaks[0]], alg_conn[peaks[0]], 
         marker="*", 
         color = "black", 
         markersize = 10, label = "First Die-Off")
plt.plot(ent_degree[peaks[1]], alg_conn[peaks[1]], 
         marker="X", 
         color = "black", 
         markersize = 10, label = "Exhaust Resource")
plt.scatter(ent_degree, alg_conn, c = timeline, cmap = "autumn_r")
plt.xlabel("Entropy of Degree Distribution")
plt.ylabel("Algebraic Connectivity")
plt.title("Entropy vs. Algebraic Connectivity Over Time")
plt.colorbar(label = "Time" )
plt.legend()
if savefigures == True:
    plt.savefig(imgdir + "ent_alg_conn.png", dpi = 250)

plt.subplots()
plt.plot(var_degree[peaks[0]], alg_conn[peaks[0]], 
         marker="*", 
         color = "black", 
         markersize = 10, label = "First Die-Off")
plt.plot(var_degree[peaks[1]], alg_conn[peaks[1]], 
         marker="X", 
         color = "black", 
         markersize = 10, label = "Exhaust Resource")
plt.scatter(var_degree, alg_conn, c = timeline, cmap = "autumn_r")
plt.xlabel("Variance of Degree Distribution")
plt.ylabel("Algebraic Connectivity")
plt.title("Variance vs. Algebraic Connectivity Over Time")
plt.colorbar(label = "Time" )
plt.legend()
if savefigures == True:
    plt.savefig(imgdir + "var_alg_conn.png", dpi = 250)

plt.subplots()
plt.plot(ent_degree[peaks[0]], mean_degree[peaks[0]], 
         marker="*", 
         color = "black", 
         markersize = 10, label = "First Die-Off")
plt.plot(ent_degree[peaks[1]], mean_degree[peaks[1]], 
         marker="X", 
         color = "black", 
         markersize = 10, label = "Exhaust Resource")
plt.scatter(ent_degree, mean_degree, c = timeline, cmap = "autumn_r")
plt.xlabel("Entropy of Degree Distribution")
plt.ylabel("Mean Degree")
plt.title("Entropy vs. Mean Degree Over Time")
plt.colorbar(label = "Time" )
plt.legend()
if savefigures == True:
    plt.savefig(imgdir + "ent_mean_degree.png", dpi = 250)

plt.subplots()
plt.plot(alg_conn[peaks[0]], mean_degree[peaks[0]], 
         marker="*", 
         color = "black", 
         markersize = 10, label = "First Die-Off")
plt.plot(alg_conn[peaks[1]], mean_degree[peaks[1]], 
         marker="X", 
         color = "black", 
         markersize = 10, label = "Exhaust Resource")
plt.scatter(alg_conn, mean_degree, c = timeline, cmap = "autumn_r")
plt.xlabel("Algebraic Connectivity")
plt.ylabel("Mean Degree")
plt.title("Algebraic Connectivity vs. Mean Degree Over Time")
plt.colorbar(label = "Time" )
plt.legend()
if savefigures == True:
    plt.savefig(imgdir + "alg_conn_mean_degree.png", dpi = 250)