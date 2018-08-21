#!/usr/bin/python3
import matplotlib.pylab as plt
import numpy as np
import networkx as nx
import csv
from sklearn import cluster
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score

#the plotting component which helps the plotting the results of cluster algorithms
def plotCluster(lab,edgeMat,FGcleared,comments):
	G1n = []
	G2n = []
	G3n = []
	G4n = []
	G5n = []
	G6n = [] 
	G7n = []
	G8n = []
	G9n = []
	G10n = []
	G11n = []
	for i in range(len(lab)):
		if lab[i] == 0:
			G1n.append(edgeMat.columns[i])
		elif lab[i] == 1:
			G2n.append(edgeMat.columns[i])
		elif lab[i] == 2:
			G3n.append(edgeMat.columns[i])
		elif lab[i] == 3:
			G4n.append(edgeMat.columns[i])
		elif lab[i] == 4:
			G5n.append(edgeMat.columns[i])
		elif lab[i] == 5:
			G6n.append(edgeMat.columns[i])
		elif lab[i] == 6:
			G7n.append(edgeMat.columns[i])
		elif lab[i] == 7:
			G8n.append(edgeMat.columns[i])
		elif lab[i] == 8:
			G9n.append(edgeMat.columns[i])
		elif lab[i] == 9:
			G10n.append(edgeMat.columns[i])
		else:
			G11n.append(edgeMat.columns[i])    
	pos = nx.spring_layout(FGcleared)
	nx.draw_networkx_nodes(FGcleared, pos, nodelist=G1n, node_color=colors[0], node_size=50, alpha=0.8)
	nx.draw_networkx_nodes(FGcleared, pos, nodelist=G2n, node_color=colors[1], node_size=50, alpha=0.8)
	nx.draw_networkx_nodes(FGcleared, pos, nodelist=G3n, node_color=colors[2], node_size=50, alpha=0.8)
	nx.draw_networkx_nodes(FGcleared, pos, nodelist=G4n, node_color=colors[3], node_size=50, alpha=0.8)
	nx.draw_networkx_nodes(FGcleared, pos, nodelist=G5n, node_color=colors[4], node_size=50, alpha=0.8)
	nx.draw_networkx_nodes(FGcleared, pos, nodelist=G6n, node_color=colors[5], node_size=50, alpha=0.8)
	nx.draw_networkx_nodes(FGcleared, pos, nodelist=G7n, node_color=colors[6], node_size=50, alpha=0.8)
	nx.draw_networkx_nodes(FGcleared, pos, nodelist=G8n, node_color=colors[7], node_size=50, alpha=0.8)
	nx.draw_networkx_nodes(FGcleared, pos, nodelist=G9n, node_color=colors[8], node_size=50, alpha=0.8)
	nx.draw_networkx_nodes(FGcleared, pos, nodelist=G10n, node_color=colors[9], node_size=50, alpha=0.8)
	nx.draw_networkx_nodes(FGcleared, pos, nodelist=G11n, node_color=colors[10], node_size=50, alpha=0.8)	
	plt.title("Clustering algorithm: " + comments)
	nx.draw_networkx_edges(FGcleared, pos, alpha=0.5)
	plt.show()

#clustering algorithms implementation found online and modified
def clusteringAlgorithms(kClusters,groundTruth,edgeMat,pos,FGcleared):
	results = []
	nmiResults = []
	arsResults = []
	# Spectral Clustering Model
	spectral = cluster.SpectralClustering(n_clusters=kClusters, affinity="precomputed", n_init=200)
	spectral.fit(edgeMat)
	plotCluster(spectral.labels_,edgeMat,FGcleared,"Spectral")
	# Transform our data to list form and store them in results list
	results.append(list(spectral.labels_))

	# -----------------------------------------

	# Agglomerative Clustering Model
	agglomerative = cluster.AgglomerativeClustering(n_clusters=kClusters, linkage="ward")
	agglomerative.fit(edgeMat)
	plotCluster(agglomerative.labels_,edgeMat,FGcleared,"Agglomerative")
	# Transform our data to list form and store them in results list
	results.append(list(agglomerative.labels_))

	# -----------------------------------------

	# K-means Clustering Model
	kmeans = cluster.KMeans(n_clusters=kClusters, n_init=200)
	kmeans.fit(edgeMat)
	plotCluster(kmeans.labels_,edgeMat,FGcleared,"K-means")
	labelsK = list(kmeans.labels_) 

	# Transform our data to list form and store them in results list
	results.append(list(kmeans.labels_))
	#print(results[-1])
	# -----------------------------------------

	# Affinity Propagation Clustering Model
	affinity = cluster.affinity_propagation(S=edgeMat, max_iter=200, damping=0.6)
	plotCluster(affinity[1],edgeMat,FGcleared,"Affinity Propagation")
	# Transform our data to list form and store them in results list
	results.append(list(affinity[1]))

	# Append the results into lists
	for x in results:
		nmiResults.append(normalized_mutual_info_score(groundTruth, x))
		arsResults.append(adjusted_rand_score(groundTruth, x))
	#print(nmiResults, arsResults)
	# -----------------------------------------
	# Code for plotting results

	# Average of NMI and ARS
	y = [sum(x) / 2 for x in zip(nmiResults, arsResults)]
	xlabels = ['Spectral', 'Agglomerative', 'Kmeans', 'Affinity Propagation']

	fig = plt.figure()
	ax = fig.add_subplot(111)

	# Set parameters for plotting
	ind = np.arange(len(y))
	width = 0.35

	# Create barchart and set the axis limits and titles
	ax.bar(ind, y, width, color='blue', error_kw=dict(elinewidth=2, ecolor='red'))
	ax.set_xlim(-width, len(ind) + width)
	ax.set_ylim(0, 2)
	ax.set_ylabel('Average Score (NMI,ARS)')
	ax.set_title('Score Evaluation with clusters equal to: ' + str(kClusters))

	# Add the xlabels to the chart
	ax.set_xticks(ind + width / 2)
	xtickNames = ax.set_xticklabels(xlabels)
	plt.setp(xtickNames, fontsize=12)

	# Add the actual value on top of each chart
	for i, v in enumerate(y):
		ax.text(i, v, str(round(v, 2)), color='blue', fontweight='bold')

	plt.show()

#aux to showPlotK
def findElem(a,b):
	sol = []
	for i in range(len(a)):
		if a[i] in b:
			sol.append(a[i])
	return sol

#used to plot the subgraphs with node degree greater than k
def showPlotK(G1,G2,G3,G4,G5,G6,G7,FGclearedkCore,k):
	c = list(FGclearedkCore)
	pos = nx.spring_layout(FGclearedkCore)
	nx.draw_networkx_nodes(FGclearedkCore, pos, nodelist=findElem(G1,c), node_color=colors[0], node_size=500, alpha=0.8)
	nx.draw_networkx_nodes(FGclearedkCore, pos, nodelist=findElem(G2,c), node_color=colors[1], node_size=500, alpha=0.8)
	nx.draw_networkx_nodes(FGclearedkCore, pos, nodelist=findElem(G3,c), node_color=colors[2], node_size=500, alpha=0.8)
	nx.draw_networkx_nodes(FGclearedkCore, pos, nodelist=findElem(G4,c), node_color=colors[3], node_size=500, alpha=0.8)
	nx.draw_networkx_nodes(FGclearedkCore, pos, nodelist=findElem(G5,c), node_color=colors[4], node_size=500, alpha=0.8)
	nx.draw_networkx_nodes(FGclearedkCore, pos, nodelist=findElem(G6,c), node_color=colors[5], node_size=500, alpha=0.8)
	plt.title("Full graph cleared of outsiders, with more than k degree: " + str(k))
	nx.draw_networkx_edges(FGclearedkCore, pos, alpha=0.5)
	plt.show()

#the actual institutes in list form  
institutes = ['Institute of Language, Cognition and Computation', 'Institute of Perception, Action and Behaviour', 'Institute for Adaptive and Neural Computation', 'Laboratory for Foundations of Computer Science', 'Institute for Computing Systems Architecture', 'Centre for Intelligent Systems and their Applications']

#names in nameTypeInst.csv
namesIn = []
names = []
namesInInstitutes = []
colors = ['red', 'green', 'blue', 'yellow', 'purple', 'grey', 'black', 'orange','silver','gold','violet']
with open('./inf-research/nameTypeInst.csv', newline='') as csvfile:
	spamreader = csv.reader(csvfile)
	for row in spamreader:
		namesIn.append(row)


fullGraph = []
with open('./inf-research/graph_full.csv', newline='') as csvfile:
	spamreader = csv.reader(csvfile)
	for row in spamreader:
		fullGraph.append(row)

G=nx.Graph()
for i in range(0,len(namesIn)):
	G.add_node(namesIn[i][0])
	names.append(namesIn[i][0])
v1 = namesIn[0][0]
v2 = namesIn[1][0]
G.add_edge(v1,v2)


#people who are part of institutes
peopleInstitutes  = 0
institutesOfNames = []
j2 = 0
institutes2 = []
for i in range(0,len(namesIn)):
	if len(namesIn[i])>3:
		str1 = (namesIn[i][3]).strip()
		if str1 not in institutes:
			institutes2.append(str1)
		if str1 in institutes:
			j2 = j2 +1
			namesInInstitutes.append(namesIn[i][0])
			institutesOfNames.append(institutes.index(str1))
		peopleInstitutes =peopleInstitutes + 1


FG 		  		= nx.Graph()
FGcleared 		= nx.Graph()
#all names
AllNames  		= []
#all names that correspond to institutes
AllNamesCleared = []
#institutes nod list
G1				= []
G2				= []
G3				= []
G4				= []
G5				= []
G5				= []
G6				= []
G7				= []
G8				= []
belongsIn		= []
#edges inside each institute
edgesIn1		= 0
edgesIn2		= 0
edgesIn3		= 0
edgesIn4		= 0
edgesIn5		= 0
edgesIn6		= 0
edgesIn7		= 0
#out edges inside insitutes
edgesOut1		= 0
edgesOut2		= 0
edgesOut3		= 0
edgesOut4		= 0
edgesOut5		= 0
edgesOut6		= 0
edgesOut7		= 0
#out edges without the outsiders of institutes
edgesOutW1		= 0
edgesOutW2		= 0
edgesOutW3		= 0
edgesOutW4		= 0
edgesOutW5		= 0
edgesOutW6		= 0
edgesOutW7		= 0

for i in range(0,len(fullGraph)):
	#add edges and nodes to the FG 
	FG.add_node(fullGraph[i][0])
	FG.add_node(fullGraph[i][1])
	FG.add_edge((fullGraph[i][0]),(fullGraph[i][1]))
	#add unique names(nodes) to the AllNames.
	if fullGraph[i][0] not in AllNames:
		AllNames.append(fullGraph[i][0])
	if fullGraph[i][1] not in AllNames:
		AllNames.append(fullGraph[i][1])
	#the indexColour actually corresponds to which institute each name belongs
	indexColour1 = -1 #initialize in case of G7
	indexColour2 = -1 #initialize in case of G7
	#check if first name belongs at an institute
	if (fullGraph[i][0] in namesInInstitutes):
		#if yes add it to the FGcleared and to the cleared names, and find each exactly institute
		index1 = namesInInstitutes.index(fullGraph[i][0])
		indexColour1 = institutesOfNames[index1]
		FGcleared.add_node(fullGraph[i][0])
		#here we find the exact node of the first name
		if fullGraph[i][0] not in AllNamesCleared:
			AllNamesCleared.append(fullGraph[i][0])
			belongsIn.append(indexColour1)
		if indexColour1==0:
			G1.append(fullGraph[i][0])
		elif indexColour1==1:
			G2.append(fullGraph[i][0])
		elif indexColour1==2:
			G3.append(fullGraph[i][0])
		elif indexColour1==3:
			G4.append(fullGraph[i][0])
		elif indexColour1==4:
			G5.append(fullGraph[i][0])
		else:
			G6.append(fullGraph[i][0])
	else:
		if fullGraph[i][0] not in G7:
			G7.append(fullGraph[i][0])
	#same process for the second name(node) as with the first.
	if (fullGraph[i][1] in namesInInstitutes):
		index2 = namesInInstitutes.index(fullGraph[i][1])
		indexColour2 = institutesOfNames[index2]
		FGcleared.add_node(fullGraph[i][1])		
		if fullGraph[i][1] not in AllNamesCleared:
			AllNamesCleared.append(fullGraph[i][1])
			belongsIn.append(indexColour2)
		if indexColour2==0:
			G1.append(fullGraph[i][1])
		elif indexColour2==1:
			G2.append(fullGraph[i][1])
		elif indexColour2==2:
			G3.append(fullGraph[i][1])
		elif indexColour2==3:
			G4.append(fullGraph[i][1])
		elif indexColour2==4:
			G5.append(fullGraph[i][1])
		else:
			G6.append(fullGraph[i][1])
	else:
		if fullGraph[i][1] not in G7:
			G7.append(fullGraph[i][1])
	if (fullGraph[i][0] in namesInInstitutes) and (fullGraph[i][1] in namesInInstitutes):
		FGcleared.add_edge(fullGraph[i][0],fullGraph[i][1])
	
	#Find out how many edges are inside and going outside each institute
	if indexColour1 == 0:
		if indexColour2 == 0:
			edgesIn1 = edgesIn1 + 1
		elif indexColour2 == -1:
			edgesOutW1 = edgesOutW1 + 1
		else:
			edgesOut1 = edgesOut1 + 1
	if indexColour1 == 1:
		if indexColour2 == 1:
			edgesIn2 = edgesIn2 + 1
		elif indexColour2 == -1:
			edgesOutW2 = edgesOutW2 + 1
		else:
			edgesOut2 = edgesOut2 + 1
	if indexColour1 == 2:
		if indexColour2 == 2:
			edgesIn3 = edgesIn3 + 1
		elif indexColour2 == -1:
			edgesOutW3 = edgesOutW3 + 1
		else:
			edgesOut3 = edgesOut3 + 1
	if indexColour1 == 3:
		if indexColour2 == 3:
			edgesIn4 = edgesIn4 + 1
		elif indexColour2 == -1:
			edgesOutW4 = edgesOutW4 + 1
		else:
			edgesOut4 = edgesOut4 + 1
	if indexColour1 == 4:
		if indexColour2 == 4:
			edgesIn5 = edgesIn5 + 1
		elif indexColour2 == -1:
			edgesOutW5 = edgesOutW5 + 1
		else:
			edgesOut5 = edgesOut5 + 1
	if indexColour1 == 5:
		if indexColour2 == 5:
			edgesIn6 = edgesIn6 + 1
		elif indexColour2 == -1:
			edgesOutW6 = edgesOutW6 + 1
		else:
			edgesOut6 = edgesOut6 + 1
	if indexColour1 == -1:
		if indexColour2 == 0:
			edgesOutW1 +=1 
		elif indexColour2 == 1:
			edgesOutW2 +=1 
		elif indexColour2 == 2:
			edgesOutW3 +=1 
		elif indexColour2 == 3:
			edgesOutW4 +=1 
		elif indexColour2 == 4:
			edgesOutW5 +=1 
		elif indexColour2 == 5:
			edgesOutW6 +=1 
		else:
			edgesOutW7 +=1
#remove self edges in FG and FGcleared for better results
FG.remove_edges_from(nx.selfloop_edges(FG))
FGcleared.remove_edges_from(nx.selfloop_edges(FGcleared))

#main component of fulll graph
FGclearedMainComponent =  max(nx.connected_component_subgraphs(FGcleared), key=len)

#main component of cleared graph
FGMainComponent = max(nx.connected_component_subgraphs(FG), key=len)


#getting rid of double elements
G1 = list(set(G1))
G2 = list(set(G2))
G3 = list(set(G3))
G4 = list(set(G4))
G5 = list(set(G5))
G6 = list(set(G6))
G7 = list(set(G7))

#printing the edges inside and the edges outside of each Insitute
print(institutes[0],": ","edges inside: ",edgesIn1,"edges outside but inside institutes: ", edgesOut1, "edges completely outside: ", edgesOutW1)
print(institutes[1],": ","edges inside: ",edgesIn2,"edges outside but inside institutes: ", edgesOut2, "edges completely outside: ", edgesOutW2)
print(institutes[2],": ","edges inside: ",edgesIn3,"edges outside but inside institutes: ", edgesOut3, "edges completely outside: ", edgesOutW3)
print(institutes[3],": ","edges inside: ",edgesIn4,"edges outside but inside institutes: ", edgesOut4, "edges completely outside: ", edgesOutW4)
print(institutes[4],": ","edges inside: ",edgesIn5,"edges outside but inside institutes: ", edgesOut5, "edges completely outside: ", edgesOutW5)
print(institutes[5],": ","edges inside: ",edgesIn6,"edges outside but inside institutes: ", edgesOut6, "edges completely outside: ", edgesOutW6)
print("Size of institutes,", institutes[0], len(G1), institutes[1], len(G2), institutes[2], len(G3), institutes[3], len(G4) , institutes[4], len(G5) , institutes[5], len(G6), " outsiders ", len(G7))

#edge expansion of each insitute
print(institutes[0],": ","edge expansion only with institutes: ",edgesOut1/(min(len(G1),len(AllNamesCleared))), "And with whole network: ",(edgesOut1+edgesOutW1)/(min(len(G1),len(AllNames))) )
print(institutes[1],": ","edge expansion only with institutes: ",edgesOut2/(min(len(G2),len(AllNamesCleared))), "And with whole network: ",(edgesOut2+edgesOutW2)/(min(len(G2),len(AllNames))) )
print(institutes[2],": ","edge expansion only with institutes: ",edgesOut3/(min(len(G3),len(AllNamesCleared))), "And with whole network: ",(edgesOut3+edgesOutW3)/(min(len(G3),len(AllNames))) )
print(institutes[3],": ","edge expansion only with institutes: ",edgesOut4/(min(len(G4),len(AllNamesCleared))), "And with whole network: ",(edgesOut4+edgesOutW4)/(min(len(G4),len(AllNames))) )
print(institutes[4],": ","edge expansion only with institutes: ",edgesOut5/(min(len(G5),len(AllNamesCleared))), "And with whole network: ",(edgesOut5+edgesOutW5)/(min(len(G5),len(AllNames))) )
print(institutes[5],": ","edge expansion only with institutes: ",edgesOut6/(min(len(G6),len(AllNamesCleared))), "And with whole network: ",(edgesOut6+edgesOutW6)/(min(len(G6),len(AllNames))) )
print("Outsiders edge expansion only with outsiders: ",edgesOut7/(min(len(G7),len(AllNamesCleared))), "And with whole network: ",(edgesOut7+edgesOutW7)/(min(len(G7),len(AllNames))) )



#The full graph cleared of external/outsiders
pos = nx.spring_layout(FGcleared)
nx.draw_networkx_nodes(FGcleared, pos, nodelist=G1, node_color=colors[0], node_size=50, alpha=0.8)
nx.draw_networkx_nodes(FGcleared, pos, nodelist=G2, node_color=colors[1], node_size=50, alpha=0.8)
nx.draw_networkx_nodes(FGcleared, pos, nodelist=G3, node_color=colors[2], node_size=50, alpha=0.8)
nx.draw_networkx_nodes(FGcleared, pos, nodelist=G4, node_color=colors[3], node_size=50, alpha=0.8)
nx.draw_networkx_nodes(FGcleared, pos, nodelist=G5, node_color=colors[4], node_size=50, alpha=0.8)
nx.draw_networkx_nodes(FGcleared, pos, nodelist=G6, node_color=colors[5], node_size=50, alpha=0.8)
plt.title("Full graph cleared of outsiders")
nx.draw_networkx_edges(FGcleared, pos, alpha=0.5)
plt.show()

#The full graph
#HERE beginning comments
'''
pos = nx.spring_layout(FG)
nx.draw_networkx_nodes(FG, pos, nodelist=G1, node_color=colors[0], node_size=10, alpha=0.8)
nx.draw_networkx_nodes(FG, pos, nodelist=G2, node_color=colors[1], node_size=10, alpha=0.8)
nx.draw_networkx_nodes(FG, pos, nodelist=G3, node_color=colors[2], node_size=10, alpha=0.8)
nx.draw_networkx_nodes(FG, pos, nodelist=G4, node_color=colors[3], node_size=10, alpha=0.8)
nx.draw_networkx_nodes(FG, pos, nodelist=G5, node_color=colors[4], node_size=10, alpha=0.8)
nx.draw_networkx_nodes(FG, pos, nodelist=G6, node_color=colors[5], node_size=10, alpha=0.8)
nx.draw_networkx_nodes(FG, pos, nodelist=G7, node_color=colors[6], node_size=10, alpha=0.8)
plt.title("Full graph cleared including outsiders")
nx.draw_networkx_edges(FG, pos, alpha=0.5)
plt.show()
'''
#HERE ending comments

#Compute the average clustering coefficient for the graph G.
ccFG = nx.average_clustering(FG)
ccFGcleared = nx.average_clustering(FGcleared)
print("average clustering coefficient for the whole Graph: ",ccFG,"average clustering coefficient for the cleared Graph: ",ccFGcleared)

#A k-core is a maximal subgraph that contains nodes of degree k or more.
#HERE beginning comments
''' 
FGkCore 	   = nx.k_core(FG,1)
FGclearedkCore = nx.k_core(FGcleared,1)
showPlotK(G1,G2,G3,G4,G5,G6,[],FGclearedkCore,1)
#showPlotK(G1,G2,G3,G4,G5,G6,G7,FGkCore,1)

FGkCore 	   = nx.k_core(FG,2)
FGclearedkCore = nx.k_core(FGcleared,2)
showPlotK(G1,G2,G3,G4,G5,G6,[],FGclearedkCore,2)
#showPlotK(G1,G2,G3,G4,G5,G6,G7,FGkCore,2)

FGkCore 	   = nx.k_core(FG,3)
FGclearedkCore = nx.k_core(FGcleared,3)
showPlotK(G1,G2,G3,G4,G5,G6,[],FGclearedkCore,3)
#showPlotK(G1,G2,G3,G4,G5,G6,G7,FGkCore,3)

FGkCore 	   = nx.k_core(FG,4)
FGclearedkCore = nx.k_core(FGcleared,4)
showPlotK(G1,G2,G3,G4,G5,G6,[],FGclearedkCore,4)
#showPlotK(G1,G2,G3,G4,G5,G6,G7,FGkCore,4)

FGkCore 	   = nx.k_core(FG,5)
FGclearedkCore = nx.k_core(FGcleared,5)
showPlotK(G1,G2,G3,G4,G5,G6,[],FGclearedkCore,5)
#showPlotK(G1,G2,G3,G4,G5,G6,G7,FGkCore,5)

FGkCore 	   = nx.k_core(FG,6)
FGclearedkCore = nx.k_core(FGcleared,6)
showPlotK(G1,G2,G3,G4,G5,G6,[],FGclearedkCore,6)
#showPlotK(G1,G2,G3,G4,G5,G6,G7,FGkCore,6)

FGkCore 	   = nx.k_core(FG,8)
FGclearedkCore = nx.k_core(FGcleared,8)
showPlotK(G1,G2,G3,G4,G5,G6,[],FGclearedkCore,7)

#print(nx.core_number(FGcleared))
'''
#HERE ending comments

print("Number of connected componets in the cleared Graph: ", nx.number_connected_components(FGcleared))

#Max connected subraph 
Gc = max(nx.connected_component_subgraphs(G), key=len)

#Returns the approximate k-component structure of a graph G.
#kCompFG 	   = nx.k_components(FG,0.95)
#kCompFGcleared = nx.k_components(FGcleared,0.95) 
#print("k_components of FG cleared: ", kCompFGcleared)

#degree distribution
degreesFG 		 = nx.degree_histogram(FG)
degreesFGcleared = nx.degree_histogram(FGcleared)
print("degree histogram of FG: ", degreesFG,sep ='\n') 
print("degree histogram of FG cleared: ", degreesFGcleared,sep='\n')

#density of graphs
densityFG 		 = nx.density(FG)
densityFGcleared = nx.density(FGcleared)
print("density of FG: ", densityFG, "density of FG cleared", densityFGcleared)


#Clustering for the complete FG
#HERE beginning comments
'''
kClusters = 0
edgeMat = nx.to_pandas_dataframe(FGcleared)
groundTruth = edgeMat.values.tolist()
pos = nx.spring_layout(FGcleared)
for i in range(len(groundTruth)):
	if edgeMat.columns[i] in G1:
		groundTruth[i] = 0
	elif edgeMat.columns[i] in G2:
		groundTruth[i] = 1
	elif edgeMat.columns[i] in G3:
		groundTruth[i] = 2
	elif edgeMat.columns[i] in G4:
		groundTruth[i] = 3
	elif edgeMat.columns[i] in G5:
		groundTruth[i] = 4
	elif edgeMat.columns[i] in G6:
		groundTruth[i] = 5
	else:
		groundTruth[i] = 6
for kClusters in range(8):
	clusteringAlgorithms(kClusters+4,groundTruth,edgeMat,pos,FG)
'''
#HERE ending comments

#Clustering for the cleared FG
kClusters = 0
edgeMat = nx.to_pandas_dataframe(FGcleared)
groundTruth = edgeMat.values.tolist()
pos = nx.spring_layout(FGcleared)
for i in range(len(groundTruth)):
	if edgeMat.columns[i] in G1:
		groundTruth[i] = 0
	elif edgeMat.columns[i] in G2:
		groundTruth[i] = 1
	elif edgeMat.columns[i] in G3:
		groundTruth[i] = 2
	elif edgeMat.columns[i] in G4:
		groundTruth[i] = 3
	elif edgeMat.columns[i] in G5:
		groundTruth[i] = 4
	else:
		groundTruth[i] = 5
for kClusters in range(8):
	clusteringAlgorithms(kClusters+2,groundTruth,edgeMat,pos,FGcleared)