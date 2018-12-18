
import numpy as np
import bct 
import igraph
from igraph import Graph, ADJ_UNDIRECTED, VertexClustering
from itertools import combinations


def matrix_to_igraph(matrix,cost,binary=False,check_tri=True,interpolation='midpoint',normalize=False,mst=False,test_matrix=True):
	"""
	Convert a matrix to an igraph object
	matrix: a numpy matrix
	cost: the proportion of edges. e.g., a cost of 0.1 has 10 percent
	of all possible edges in the graph
	binary: False, convert weighted values to 1
	check_tri: True, ensure that the matrix contains upper and low triangles.
	if it does not, the cost calculation changes.
	interpolation: midpoint, the interpolation method to pass to np.percentile
	normalize: False, make all edges sum to 1. Convienient for comparisons across subjects,
	as this ensures the same sum of weights and number of edges are equal across subjects
	mst: False, calculate the maximum spanning tree, which is the strongest set of edges that
	keep the graph connected. This is convienient for ensuring no nodes become disconnected.
	"""
	matrix = np.array(matrix)
	matrix = threshold(matrix,cost,binary,check_tri,interpolation,normalize,mst)
	g = Graph.Weighted_Adjacency(matrix.tolist(),mode=ADJ_UNDIRECTED,attr="weight")
	print 'Matrix converted to graph with density of: ' + str(g.density())
	if abs(np.diff([cost,g.density()])[0]) > .005:
		print 'Density not %s! Did you want: ' %(cost)+ str(g.density()) + ' ?' 
	return g


def threshold(matrix,cost,binary=False,check_tri=True,interpolation='midpoint',normalize=False,mst=False,test_matrix=True):
	"""
	Threshold a numpy matrix to obtain a certain "cost".
	matrix: a numpy matrix
	cost: the proportion of edges. e.g., a cost of 0.1 has 10 percent
	of all possible edges in the graph
	binary: False, convert weighted values to 1
	check_tri: True, ensure that the matrix contains upper and low triangles.
	if it does not, the cost calculation changes.
	interpolation: midpoint, the interpolation method to pass to np.percentile
	normalize: False, make all edges sum to 1. Convienient for comparisons across subjects,
	as this ensures the same sum of weights and number of edges are equal across subjects
	mst: False, calculate the maximum spanning tree, which is the strongest set of edges that
	keep the graph connected. This is convienient for ensuring no nodes become disconnected.
	"""
	matrix[np.isnan(matrix)] = 0.0
	c_cost_int = 100-(cost*100)
	if check_tri == True:
		if np.sum(np.triu(matrix)) == 0.0 or np.sum(np.tril(matrix)) == 0.0:
			c_cost_int = 100.-((cost/2.)*100.)
	if c_cost_int > 0:
		if mst == False:
			matrix[matrix<np.percentile(matrix,c_cost_int,interpolation=interpolation)] = 0.
		else:
			if test_matrix == True: t_m = matrix.copy()
			assert (np.tril(matrix,-1) == np.triu(matrix,1).transpose()).all()
			matrix = np.tril(matrix,-1)
			mst = minimum_spanning_tree(matrix*-1)*-1
			mst = mst.toarray()
			mst = mst.transpose() + mst
			matrix = matrix.transpose() + matrix
			if test_matrix == True: assert (matrix == t_m).all() == True
			matrix[(matrix<np.percentile(matrix,c_cost_int,interpolation=interpolation)) & (mst==0.0)] = 0.
	if binary == True:
		matrix[matrix>0] = 1
	if normalize == True:
		matrix = matrix/np.sum(matrix)
	return matrix


def ave_consensus_costs_parition(matrix, min_cost, max_cost):
	'''Run a partition for every cost threshold using infomap, turn parition into identiy matrix, average
	identiy matrix across costs to generate consensus matrix, run infomap on consens matrix to obtain final
partition'''

	consensus_matricies = np.zeros((len(np.arange(min_cost, max_cost+0.01, 0.01)), matrix.shape[0], matrix.shape[1]))
	
	for i, cost in enumerate(np.arange(min_cost, max_cost+0.01, 0.01)):
		
		graph = matrix_to_igraph(matrix.copy(),cost=cost)
		infomap_paritition = graph.community_infomap(edge_weights='weight')
		consensus_matricies[i,:,:] = community_matrix(infomap_paritition.membership)

	ave_consensus = np.mean(consensus_matricies, axis=0)
	graph = matrix_to_igraph(ave_consensus,cost=1.)
	final_infomap_partition = graph.community_infomap(edge_weights='weight')	

	return final_infomap_partition.membership




def power_recursive_partition(matrix, min_cost, max_cost, min_community_size=5):
	''' this is the interpretation of what Power did in his 2011 Neuron paper, start with a high cost treshold, get infomap parition, then step down, but keep the
	parition that did not change across thresholds'''

	final_edge_matrix = matrix.copy()
	final_identity_matrix = np.zeros(matrix.shape)

	cost = max_cost

	while True:
		graph = matrix_to_igraph(matrix.copy(),cost=cost)
		partition = graph.community_infomap(edge_weights='weight')
		connected_nodes = []
		
		for node in range(partition.graph.vcount()):
			if partition.sizes()[partition.membership[node]] > min_community_size:
				connected_nodes.append(node)
		
		within_community_edges = []
		between_community_edges = []
		for edge in combinations(connected_nodes,2):
			if partition.membership[edge[0]] == partition.membership[edge[1]]:
				within_community_edges.append(edge)
			else:
				between_community_edges.append(edge)
		for edge in within_community_edges:
			final_identity_matrix[edge[0],edge[1]] = 1
			final_identity_matrix[edge[1],edge[0]] = 1
		for edge in between_community_edges:
			final_identity_matrix[edge[0],edge[1]] = 0
			final_identity_matrix[edge[1],edge[0]] = 0
		if cost < min_cost:
			break
		if cost <= .05:
			cost = cost - 0.001
			continue
		if cost <= .15:
			cost = cost - 0.01
			continue

	graph = matrix_to_igraph(final_identity_matrix,cost=1.)
	final_infomap_partition = np.array(graph.community_infomap(edge_weights='weight').membership)
	return final_infomap_partition 



def community_matrix(membership):
	'''To generate a identiy matrix where nodes that belong to the same community/patition has 
	edges set as "1" between them, otherwise 0 '''

	membership = np.array(membership).reshape(-1)
	
	final_matrix = np.zeros((len(membership),len(membership)))
	final_matrix[:] = np.nan
	connected_nodes = []
	for i in np.unique(membership):
		for n in np.array(np.where(membership==i))[0]:
			connected_nodes.append(int(n))
	
	within_community_edges = []
	between_community_edges = []
	connected_nodes = np.array(connected_nodes)
	for edge in combinations(connected_nodes,2):
		if membership[edge[0]] == membership[edge[1]]:
			within_community_edges.append(edge)
		else:
			between_community_edges.append(edge)
	
	# set edge as 1 if same community
	for edge in within_community_edges:
		final_matrix[edge[0],edge[1]] = 1
		final_matrix[edge[1],edge[0]] = 1
	for edge in between_community_edges:
		final_matrix[edge[0],edge[1]] = 0
		final_matrix[edge[1],edge[0]] = 0

	return final_matrix



# load matrix
matrix = np.genfromtxt('HCP_MMP1_roi-pair_corr.csv',delimiter=',',dtype=None)
matrix[np.isnan(matrix)] = 0.0  


# step through costs, do infomap, return final infomap across cost
max_cost = .15
min_cost = .01

# ave consensus across costs
partition = ave_consensus_costs_parition(matrix, min_cost, max_cost)
partition = np.array(partition) + 1

# import thresholded matrix to BCT, import partition, run WMD/PC
PCs = np.zeros((len(np.arange(min_cost, max_cost+0.01, 0.01)), matrix.shape[0]))
WMDs = np.zeros((len(np.arange(min_cost, max_cost+0.01, 0.01)), matrix.shape[0]))

for i, cost in enumerate(np.arange(min_cost, max_cost, 0.01)):
	
	tmp_matrix = threshold(matrix.copy(), cost)
	
	#PC
	PCs[i,:] = bct.participation_coef(tmp_matrix, partition)
	#WMD
	WMDs[i,:] = bct.module_degree_zscore(matrix, partition)

np.save("partition", partition)
np.save("PCs", PCs)
np.save("WMDs", WMDs)

#altantively, merge consensus using the power method
recursive_partition = power_recursive_partition(matrix, min_cost, max_cost)
recursive_partition = recursive_partition + 1

np.save('rescursive_partition', recursive_partition)


