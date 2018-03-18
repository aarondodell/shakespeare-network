import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pytictoc import TicToc


# TIC
tt = TicToc()
tt.tic()

'''
# creating Vader_Analyzer object
vader_analyzer = SentimentIntensityAnalyzer()
'''

# initial read of all plays
allplays_df = pd.read_csv('Shakespeare_data.csv')
allplays_df[['Act','Scene','Line']] = allplays_df['ActSceneLine'].str.split('.',expand = True)
allplays_df[['Act','Scene','Line']] = allplays_df[['Act','Scene','Line']].astype('float')

'''
# creating SentimentDict and SentimentNum column
allplays_df['SentimentDict'] = allplays_df['PlayerLine'].apply(lambda x: vader_analyzer.polarity_scores(x))
allplays_df['SentimentNum'] = allplays_df['SentimentDict'].apply(lambda x: x['compound'])
'''

# proper titles for Play column
allplays_df['Play'] = allplays_df['Play'].apply(lambda x: x.title() if x.find('Henry') is -1 and x.find('Richard') is -1 else x)

# unique values for Play iteration
play_lst = allplays_df['Play'].unique().tolist()
play_fname_lst = [x.replace(' ','') for x in play_lst]

os.chdir('.\OutputGraphs')

for tmp_play_name, tmp_play_fname in zip(play_lst, play_fname_lst):

    # filter for the TmpPlayName, and remove any stage directions (where ActSceneLine is NA)
    fullplay_df = allplays_df[(allplays_df['Play'] == tmp_play_name) & pd.notna(allplays_df['ActSceneLine'])]
    
    # get each Player's Line count in each Act and Scene
    playerlinecounts_df = fullplay_df.groupby(['Act','Scene','Player']).size().reset_index()
    playerlinecounts_df.rename(columns = {0: 'Count'}, inplace = True)
    
    # total Play Line Count for weighting later...
    totalplaylines_num = sum(playerlinecounts_df['Count'])
    
    # Master Player permutation to concatenate
    master_playerpermutation_df = pd.DataFrame(data = None)
    
    # iterating through each Act and Scene
    for (tmp_act, tmp_scene), tmp_playerlinecounts_df in playerlinecounts_df.groupby(['Act','Scene']):
        #all ActScene Players, and their Counts
        tmp_player_lst = tmp_playerlinecounts_df['Player'].tolist()
        tmp_count_lst = tmp_playerlinecounts_df['Count'].tolist()
            
        # creating empty PLayerPermutation List to populate
        tmp_playerpermutation_lst = []
        
        # iterating over player list, creating permutation between each player
        # and their weighted line contribution to the whole Play
        for i in range(len(tmp_player_lst)-1):
            for j in range(i+1,len(tmp_player_lst)):
                tmp_playerpermutation_lst.append((tmp_player_lst[i], tmp_player_lst[j],
                                                  (tmp_count_lst[i]*tmp_count_lst[j]) / totalplaylines_num,
                                                  tmp_act, tmp_scene))
        
        # creating PlayerPermutation DataFrame
        tmp_playerpermutation_df = pd.DataFrame(data = tmp_playerpermutation_lst,
                                                columns = ['PlayerA','PlayerB','WeightedContribution', 'Act', 'Scene'])
        
        # appending PlayerPermutation DF onto Master PlayerPermutation DF
        master_playerpermutation_df = master_playerpermutation_df.append(tmp_playerpermutation_df)
        
    # summing up the Master permutations
    summed_masterplayerperm_df = master_playerpermutation_df.groupby(['PlayerA','PlayerB']) \
                                    ['WeightedContribution'].sum().reset_index()
    summed_masterplayerperm_df.sort_values('WeightedContribution', ascending = False ,inplace = True)
    
    # creating NX Summed DF for NetworkX visualization
    nx_summed_df = summed_masterplayerperm_df.copy()
    nx_summed_df.rename(columns = {'PlayerA': 'from', 'PlayerB': 'to', 'WeightedContribution': 'weight'}, inplace= True)
    
    # filtering NX Summed DF for the top 10 players with the most lines
    topten_players = playerlinecounts_df.groupby('Player')['Count'].sum() \
                        .sort_values(ascending = False).index[:10].tolist()
    nx_summed_df = nx_summed_df[nx_summed_df['from'].isin(topten_players) & nx_summed_df['to'].isin(topten_players)]
    
    # creating NX graph object
    fullplay_graph = nx.Graph()
    fullplay_graph.add_edges_from([(tmp_from, tmp_to, {'weight': tmp_weight}) \
                                      for tmp_from, tmp_to, tmp_weight \
                                      in zip(nx_summed_df['from'], nx_summed_df['to'], nx_summed_df['weight'])])
    
    # finding Weight list for Edges
    weight_lst = [fullplay_graph[a][b]['weight'] for a, b in fullplay_graph.edges]
        
    # finding Centrality list for Nodes
    playercounts_wholeplay = playerlinecounts_df.groupby('Player')['Count'].sum().sort_values(ascending = False)[:10]
    playercounts_wholeplay = playercounts_wholeplay / playercounts_wholeplay.max()
    
    nodenum_lst = [playercounts_wholeplay[node] for node in fullplay_graph.nodes]
    
    # drawing NX graph object
    plt.figure(figsize = (8,6), dpi = 180)
    plt.title(tmp_play_name)
    
    nx.draw_networkx(fullplay_graph, with_labels=True,
            node_color= nodenum_lst, node_size = 1500,
            cmap = plt.cm.Blues, vmin = -0.5, vmax = 1.25,
            font_size = 8, font_weight = 'semibold',
            width=weight_lst,
            edge_color = weight_lst, edge_cmap=plt.cm.RdPu,
            pos = nx.spring_layout(fullplay_graph))
    
    plt.axis('off')
    plt.savefig(tmp_play_fname + '_Spring.png')
    plt.close()
    
# returning to normal directory
os.chdir('..')

# TOC
tt.toc()