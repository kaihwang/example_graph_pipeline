import numpy as np
import bct
import pandas as pd
from igraph import Graph, ADJ_UNDIRECTED, VertexClustering
from itertools import combinations
import nibabel as nib
from nibabel import cifti2
import os
import matplotlib.pyplot as plt
from nilearn.input_data import NiftiLabelsMasker
from scipy.stats import zscore

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
	print('Matrix converted to graph with density of:' + str(g.density()))
	if abs(np.diff([cost,g.density()])[0]) > .005:
		print('Density not %s! Did you want: ' %(cost)+ str(g.density()) + ' ?')
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


def test_pipline():
	''' A run thorugh test using a csv input from Aaron'''
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


def cal_dataset_adj(dset='HCP', roifile = 'CA_2mm'):
	'''short hand function to load timeseries from CIFTI(HCP) or NIFTI(others), need to say which dataset and give a ROI parcellation
	The default is using Cole's network wide parcellation in CIFTI label format for HCP.
	For MGH/NKI, a NIFTI version of that template is available as CA_2mm
	'''

	if dset=='HCP':

		subjects = np.loadtxt('/home/kahwang/bin/example_graph_pipeline/HCP_subjects', dtype=int)

		adj = []
		for s in subjects:

			rest1 = '/data/not_backed_up/shared/HCP/%s/MNINonLinear/Results/rfMRI_REST1_LR/rfMRI_REST1_LR_Atlas_hp2000_clean.dtseries.nii' %s
			rest2 = '/data/not_backed_up/shared/HCP/%s/MNINonLinear/Results/rfMRI_REST1_RL/rfMRI_REST1_RL_Atlas_hp2000_clean.dtseries.nii' %s
			rest3 = '/data/not_backed_up/shared/HCP/%s/MNINonLinear/Results/rfMRI_REST2_LR/rfMRI_REST2_LR_Atlas_hp2000_clean.dtseries.nii' %s
			rest4 = '/data/not_backed_up/shared/HCP/%s/MNINonLinear/Results/rfMRI_REST2_RL/rfMRI_REST2_RL_Atlas_hp2000_clean.dtseries.nii' %s

			rest_runs = [ rest1, rest2, rest3, rest4]
			roi='CA_CIFTI'
			parcel_template = '/data/backed_up/shared/ROIs/' + roi + '.nii'
			tmp_cifti = '/home/kahwang/tmp/tmpfile.ptseries.nii'

			ptseries = np.array([])
			for r in rest_runs:

				# export mean ts for each parcel using wb_command, because it deals with CIFTI....
				os.system('wb_command -cifti-parcellate ' + r + ' ' + parcel_template + ' COLUMN ' + tmp_cifti + ' -method MEAN')
				if ptseries.size ==0:
					ptseries = np.squeeze(nib.load(tmp_cifti).get_data()).T
				else:
					ptseries = np.concatenate([ptseries, np.squeeze(nib.load(tmp_cifti).get_data()).T], axis=1)

			#calculate corrcoef, then take fisher z transformation, append to list
			adj.append(np.arctanh(np.corrcoef(ptseries)))


	elif dset=='MGH':

		subjects = pd.read_csv('/home/kahwang/bin/example_graph_pipeline/MGH_Subjects', names=['ID'])['ID']
		roi=roifile
		parcel_template = '/data/backed_up/shared/ROIs/' + roi + '.nii.gz'
		masker = NiftiLabelsMasker(labels_img=parcel_template, standardize=False)

		adj = []
		for s in subjects:
			try:
				inputfile = '/data/backed_up/shared/MGH/MGH/%s/MNINonLinear/rfMRI_REST.nii.gz' %s
				ts = masker.fit_transform(inputfile).T
				adj.append(np.arctanh(np.corrcoef(ts)))
			except:
				continue

	elif dset=='NKI':

		subjects = pd.read_csv('/home/kahwang/bin/example_graph_pipeline/NKI_subjects', names=['ID'])['ID']
		roi=roifile
		parcel_template = '/data/backed_up/shared/ROIs/' + roi + '.nii.gz'
		masker = NiftiLabelsMasker(labels_img=parcel_template, standardize=False)

		adj = []
		for s in subjects:
			try:
				inputfile = '/data/backed_up/shared/NKI/%s/MNINonLinear/rfMRI_REST_mx_1400.nii.gz' %s
				ts = masker.fit_transform(inputfile).T
				adj.append(np.arctanh(np.corrcoef(ts)))
			except:
				continue

	else:
		print('no dataset??')
		return None

	#average across subjects
	avadj = np.nanmean(adj, axis=0)
	avadj[avadj==np.inf] = 1.0 #set diag

	return avadj, adj


def gen_groupave_adj(roifile):
	#HCP_avadj, _ = cal_dataset_adj(dset='HCP')
	#np.save('HCP_adj', HCP_avadj)
	NKI_avadj, _ = cal_dataset_adj(dset='NKI', roifile = roifile)
	fn = 'NKI_adj_%s' %roifile
	np.save(fn, NKI_avadj)
	MGH_avadj, _ = cal_dataset_adj(dset='MGH', roifile = roifile)
	fn = 'MGH_adj_%s' %roifile
	np.save(fn, MGH_avadj)

	return 	NKI_avadj, MGH_avadj


def write_graph_to_pscalar(graph_metric, fn):
	'''write parcel-wise graph metric to a parcel cifiti data for visuliziation
	assuming using the Cole 718 parcel'''

	tempcifti = nib.load('/home/kahwang/bin/ColeAnticevicNetPartition/tempate.pscalar.nii')
	tempscalar = tempcifti.get_data()
	tempscalar[0,:] = graph_metric
	tempcifti=cifti2.Cifti2Image(tempscalar, tempcifti.get_header())
	nib.save(tempcifti, fn)


def write_graph_to_vol_yeo_template_nifti(graph_metric, fn, resolution=400):
	'''short hand to write vol based nifti file of the graph metrics
	assuming Cole 718 parcels, voxels in each parcel will be replaced with the graph metric'''

	#roi_df = pd.read_csv('/home/kahwang/bin/example_graph_pipeline/Updated_CA_ROI_List.csv')
	#roi_df.loc[0:359,'KEYVALUE'] = np.arange(1,361)
	if resolution == 400:
		vol_template = nib.load('/home/kahwang/bsh/ROIs/Yeo425x17LiberalCombinedMNI.nii.gz')
	elif resluition == 900:
		vol_template = nib.load('/home/kahwang/bsh/ROIs/Schaefer900.nii.gz')
	else:
		print ('Error with template')
		return

	v_data = vol_template.get_data()
	graph_data = np.zeros((np.shape(v_data)))

	for i in np.arange(425):
		#key = roi_df['KEYVALUE'][i]
		graph_data[v_data == i+1] = graph_metric[i]

	new_nii = nib.Nifti1Image(graph_data, affine = vol_template.get_affine(), header = vol_template.get_header())
	nib.save(new_nii, fn)


if __name__ == "__main__":


	### Get group ave adj
	#NKI_avadj, MGH_avadj = gen_groupave_adj(roifile = 'Yeo425x17LiberalCombinedMNI')



	###Kitchen sink centrality loop, save nii file (cifti parcel) CA template
	# print('caluclate centraltiy metircs')

	# CI = np.loadtxt('/home/kahwang/bin/ColeAnticevicNetPartition/cortex_subcortex_parcel_network_assignments.txt', dtype=int)
	# roi_df = pd.read_csv('/home/kahwang/bin/example_graph_pipeline/Updated_CA_ROI_List.csv')


	# MGH_avadj = np.load('MGH_adj.npy')
	# NKI_avadj = np.load('NKI_adj.npy')
	# #HCP_avadj =  np.load('HCP_adj.npy')

	# max_cost = .15
	# min_cost = .01

	# MATS = [MGH_avadj, NKI_avadj]
	# dsets = ['MGH', 'NKI', 'HCP']


	# # import thresholded matrix to BCT, import partition, run WMD/PC
	# PC = np.zeros((len(np.arange(min_cost, max_cost+0.01, 0.01)), 718))
	# WMD = np.zeros((len(np.arange(min_cost, max_cost+0.01, 0.01)), 718))
	# EC = np.zeros((len(np.arange(min_cost, max_cost+0.01, 0.01)), 718))
	# GC = np.zeros((len(np.arange(min_cost, max_cost+0.01, 0.01)), 718))
	# SC = np.zeros((len(np.arange(min_cost, max_cost+0.01, 0.01)), 718))
	# ST = np.zeros((len(np.arange(min_cost, max_cost+0.01, 0.01)), 718))

	# for ix, matrix in enumerate(MATS):
	# 	for i, cost in enumerate(np.arange(min_cost, max_cost, 0.01)):

	# 			tmp_matrix = threshold(matrix.copy(), cost)

	# 			PC[i,:] = bct.participation_coef(tmp_matrix, CI)
	# 			WMD[i,:] = bct.module_degree_zscore(tmp_matrix,CI)
	# 			EC[i,:] = bct.eigenvector_centrality_und(tmp_matrix)
	# 			GC[i,:], _ = bct.gateway_coef_sign(tmp_matrix, CI)
	# 			SC[i,:] = bct.subgraph_centrality(tmp_matrix)
	# 			ST[i,:] = bct.strengths_und(tmp_matrix)

	# 	#fn = 'images/%s_PC.pscalar.nii' %dsets[ix]
	# 	#write_graph_to_pscalar(np.nanmean(PC,axis=0), fn)
	# 	fn = 'images/%s_PC.nii' %dsets[ix]
	# 	write_graph_to_vol_nifti(np.nanmean(PC,axis=0), fn)
	# 	fn = 'PC_%s' %dsets[ix]
	# 	roi_df[fn] = np.nanmean(PC,axis=0)

	# 	#fn = 'images/%s_WMD.pscalar.nii' %dsets[ix]
	# 	#write_graph_to_pscalar(np.nanmean(WMD,axis=0), fn)
	# 	fn = 'images/%s_WMD.nii' %dsets[ix]
	# 	write_graph_to_vol_nifti(np.nanmean(WMD,axis=0), fn)
	# 	fn = 'WMD_%s' %dsets[ix]
	# 	roi_df[fn] = np.nanmean(WMD,axis=0)

	# 	#fn = 'images/%s_EigenCent.pscalar.nii' %dsets[ix]
	# 	#write_graph_to_pscalar(np.nanmean(EC, axis=0), fn)
	# 	fn = 'images/%s_EigenCent.nii' %dsets[ix]
	# 	write_graph_to_vol_nifti(np.nanmean(EC,axis=0), fn)
	# 	fn = 'EigenCentrality_%s' %dsets[ix]
	# 	roi_df[fn] = np.nanmean(EC,axis=0)

	# 	#fn = 'images/%s_GatewayCent.pscalar.nii' %dsets[ix]
	# 	#write_graph_to_pscalar(np.nanmean(GC, axis=0), fn)
	# 	fn = 'images/%s_GatewayCent.nii' %dsets[ix]
	# 	write_graph_to_vol_nifti(np.nanmean(GC,axis=0), fn)
	# 	fn = 'GatewayCentrality_%s' %dsets[ix]
	# 	roi_df[fn] = np.nanmean(GC,axis=0)

	# 	#fn = 'images/%s_SubgraphCent.pscalar.nii' %dsets[ix]
	# 	#write_graph_to_pscalar(np.nanmean(SC, axis=0), fn)
	# 	fn = 'images/%s_SubgraphCent.nii' %dsets[ix]
	# 	write_graph_to_vol_nifti(np.nanmean(SC,axis=0), fn)
	# 	fn = 'SubgraphCentrality_%s' %dsets[ix]
	# 	roi_df[fn] = np.nanmean(SC,axis=0)

	# 	#fn = 'images/%s_WeightedDegree.pscalar.nii' %dsets[ix]
	# 	#write_graph_to_pscalar(np.nanmean(ST, axis=0), fn)
	# 	fn = 'images/%s_WeightedDegree.nii' %dsets[ix]
	# 	write_graph_to_vol_nifti(np.nanmean(ST,axis=0), fn)
	# 	fn = 'WeightedDegreeCentrality_%s' %dsets[ix]
	# 	roi_df[fn] = np.nanmean(ST,axis=0)


	# 	#zscore version, eseentialy ranking across parcels/rois
	# 	#fn = 'images/%s_zPC.pscalar.nii' %dsets[ix]
	# 	#write_graph_to_pscalar(zscore(np.nanmean(PC,axis=0)), fn)
	# 	fn = 'images/%s_zPC.nii' %dsets[ix]
	# 	write_graph_to_vol_nifti(zscore(np.nanmean(PC,axis=0)), fn)

	# 	#fn = 'images/%s_zWMD.pscalar.nii' %dsets[ix]
	# 	#write_graph_to_pscalar(zscore(np.nanmean(WMD,axis=0)), fn)
	# 	fn = 'images/%s_zWMD.nii' %dsets[ix]
	# 	write_graph_to_vol_nifti(zscore(np.nanmean(WMD,axis=0)), fn)

	# 	#fn = 'images/%s_zEigenCent.pscalar.nii' %dsets[ix]
	# 	#write_graph_to_pscalar(zscore(np.nanmean(EC, axis=0)), fn)
	# 	fn = 'images/%s_zEigenCent.nii' %dsets[ix]
	# 	write_graph_to_vol_nifti(zscore(np.nanmean(EC,axis=0)), fn)

	# 	#fn = 'images/%s_zGatewayCent.pscalar.nii' %dsets[ix]
	# 	#write_graph_to_pscalar(zscore(np.nanmean(GC, axis=0)), fn)
	# 	fn = 'images/%s_zGatewayCent.nii' %dsets[ix]
	# 	write_graph_to_vol_nifti(zscore(np.nanmean(GC,axis=0)), fn)

	# 	#fn = 'images/%s_zSubgraphCent.pscalar.nii' %dsets[ix]
	# 	#write_graph_to_pscalar(zscore(np.nanmean(SC, axis=0)), fn)
	# 	fn = 'images/%s_zSubgraphCent.nii' %dsets[ix]
	# 	write_graph_to_vol_nifti(zscore(np.nanmean(SC,axis=0)), fn)

	# 	#fn = 'images/%s_zWeightedDegree.pscalar.nii' %dsets[ix]
	# 	#write_graph_to_pscalar(zscore(np.nanmean(ST, axis=0)), fn)
	# 	fn = 'images/%s_zWeightedDegree.nii' %dsets[ix]
	# 	write_graph_to_vol_nifti(zscore(np.nanmean(ST,axis=0)), fn)





########################################################################
#Kitchen sink centrality loop, save nii file (cifti parcel) Yeo template
########################################################################
	# print('caluclate centraltiy metircs')
	#
	# CI = np.loadtxt('/home/kahwang/bin/example_graph_pipeline/Yeo425x17LiberalCombinedCommunityAffiliation.1D', dtype=int)
	# #roi_df = pd.read_csv('/home/kahwang/bin/example_graph_pipeline/Updated_CA_ROI_List.csv')
	#
	#
	# MGH_avadj = np.load('NKI_adj_Yeo425x17LiberalCombinedMNI.npy')
	# NKI_avadj = np.load('NKI_adj_Yeo425x17LiberalCombinedMNI.npy')
	# #HCP_avadj =  np.load('HCP_adj.npy')
	#
	# max_cost = .15
	# min_cost = .01
	#
	# MATS = [MGH_avadj, NKI_avadj]
	# dsets = ['MGH', 'NKI']
	#
	#
	# # import thresholded matrix to BCT, import partition, run WMD/PC
	# PC = np.zeros((len(np.arange(min_cost, max_cost+0.01, 0.01)), 425))
	# WMD = np.zeros((len(np.arange(min_cost, max_cost+0.01, 0.01)), 425))
	# EC = np.zeros((len(np.arange(min_cost, max_cost+0.01, 0.01)), 425))
	# GC = np.zeros((len(np.arange(min_cost, max_cost+0.01, 0.01)), 425))
	# SC = np.zeros((len(np.arange(min_cost, max_cost+0.01, 0.01)), 425))
	# ST = np.zeros((len(np.arange(min_cost, max_cost+0.01, 0.01)), 425))
	#
	# for ix, matrix in enumerate(MATS):
	# 	for i, cost in enumerate(np.arange(min_cost, max_cost, 0.01)):
	#
	# 			tmp_matrix = threshold(matrix.copy(), cost)
	#
	# 			PC[i,:] = bct.participation_coef(tmp_matrix, CI)
	# 			WMD[i,:] = bct.module_degree_zscore(tmp_matrix,CI)
	# 			EC[i,:] = bct.eigenvector_centrality_und(tmp_matrix)
	# 			GC[i,:], _ = bct.gateway_coef_sign(tmp_matrix, CI)
	# 			SC[i,:] = bct.subgraph_centrality(tmp_matrix)
	# 			ST[i,:] = bct.strengths_und(tmp_matrix)
	#
	# 	#fn = 'images/%s_PC.pscalar.nii' %dsets[ix]
	# 	#write_graph_to_pscalar(np.nanmean(PC,axis=0), fn)
	# 	fn = 'images/yeotemplate_%s_PC.nii' %dsets[ix]
	# 	write_graph_to_vol_yeo_template_nifti(np.nanmean(PC,axis=0), fn)
	# 	#fn = 'PC_%s' %dsets[ix]
	# 	#roi_df[fn] = np.nanmean(PC,axis=0)
	#
	# 	#fn = 'images/%s_WMD.pscalar.nii' %dsets[ix]
	# 	#write_graph_to_pscalar(np.nanmean(WMD,axis=0), fn)
	# 	fn = 'images/yeotemplate_%s_WMD.nii' %dsets[ix]
	# 	write_graph_to_vol_yeo_template_nifti(np.nanmean(WMD,axis=0), fn)
	# 	#fn = 'WMD_%s' %dsets[ix]
	# 	#roi_df[fn] = np.nanmean(WMD,axis=0)
	#
	# 	#fn = 'images/%s_EigenCent.pscalar.nii' %dsets[ix]
	# 	#write_graph_to_pscalar(np.nanmean(EC, axis=0), fn)
	# 	fn = 'images/yeotemplate_%s_EigenCent.nii' %dsets[ix]
	# 	write_graph_to_vol_yeo_template_nifti(np.nanmean(EC,axis=0), fn)
	# 	#fn = 'EigenCentrality_%s' %dsets[ix]
	# 	#roi_df[fn] = np.nanmean(EC,axis=0)
	#
	# 	#fn = 'images/%s_GatewayCent.pscalar.nii' %dsets[ix]
	# 	#write_graph_to_pscalar(np.nanmean(GC, axis=0), fn)
	# 	fn = 'images/yeotemplate_%s_GatewayCent.nii' %dsets[ix]
	# 	write_graph_to_vol_yeo_template_nifti(np.nanmean(GC,axis=0), fn)
	# 	#fn = 'GatewayCentrality_%s' %dsets[ix]
	# 	#roi_df[fn] = np.nanmean(GC,axis=0)
	#
	# 	#fn = 'images/%s_SubgraphCent.pscalar.nii' %dsets[ix]
	# 	#write_graph_to_pscalar(np.nanmean(SC, axis=0), fn)
	# 	fn = 'images/yeotemplate_%s_SubgraphCent.nii' %dsets[ix]
	# 	write_graph_to_vol_yeo_template_nifti(np.nanmean(SC,axis=0), fn)
	# 	#fn = 'SubgraphCentrality_%s' %dsets[ix]
	# 	#roi_df[fn] = np.nanmean(SC,axis=0)
	#
	# 	#fn = 'images/%s_WeightedDegree.pscalar.nii' %dsets[ix]
	# 	#write_graph_to_pscalar(np.nanmean(ST, axis=0), fn)
	# 	fn = 'images/yeotemplate_%s_WeightedDegree.nii' %dsets[ix]
	# 	write_graph_to_vol_yeo_template_nifti(np.nanmean(ST,axis=0), fn)
	# 	#fn = 'WeightedDegreeCentrality_%s' %dsets[ix]
	# 	#roi_df[fn] = np.nanmean(ST,axis=0)
	#
	#
	# 	#zscore version, eseentialy ranking across parcels/rois
	# 	#fn = 'images/%s_zPC.pscalar.nii' %dsets[ix]
	# 	#write_graph_to_pscalar(zscore(np.nanmean(PC,axis=0)), fn)
	# 	fn = 'images/yeotemplate_%s_zPC.nii' %dsets[ix]
	# 	write_graph_to_vol_yeo_template_nifti(zscore(np.nanmean(PC,axis=0)), fn)
	#
	# 	#fn = 'images/%s_zWMD.pscalar.nii' %dsets[ix]
	# 	#write_graph_to_pscalar(zscore(np.nanmean(WMD,axis=0)), fn)
	# 	fn = 'images/yeotemplate_%s_zWMD.nii' %dsets[ix]
	# 	write_graph_to_vol_yeo_template_nifti(zscore(np.nanmean(WMD,axis=0)), fn)
	#
	# 	#fn = 'images/%s_zEigenCent.pscalar.nii' %dsets[ix]
	# 	#write_graph_to_pscalar(zscore(np.nanmean(EC, axis=0)), fn)
	# 	fn = 'images/yeotemplate_%s_zEigenCent.nii' %dsets[ix]
	# 	write_graph_to_vol_yeo_template_nifti(zscore(np.nanmean(EC,axis=0)), fn)
	#
	# 	#fn = 'images/%s_zGatewayCent.pscalar.nii' %dsets[ix]
	# 	#write_graph_to_pscalar(zscore(np.nanmean(GC, axis=0)), fn)
	# 	fn = 'images/yeotemplate_%s_zGatewayCent.nii' %dsets[ix]
	# 	write_graph_to_vol_yeo_template_nifti(zscore(np.nanmean(GC,axis=0)), fn)
	#
	# 	#fn = 'images/%s_zSubgraphCent.pscalar.nii' %dsets[ix]
	# 	#write_graph_to_pscalar(zscore(np.nanmean(SC, axis=0)), fn)
	# 	fn = 'images/yeotemplate_%s_zSubgraphCent.nii' %dsets[ix]
	# 	write_graph_to_vol_yeo_template_nifti(zscore(np.nanmean(SC,axis=0)), fn)
	#
	# 	#fn = 'images/%s_zWeightedDegree.pscalar.nii' %dsets[ix]
	# 	#write_graph_to_pscalar(zscore(np.nanmean(ST, axis=0)), fn)
	# 	fn = 'images/yeotemplate_%s_zWeightedDegree.nii' %dsets[ix]
	# 	write_graph_to_vol_yeo_template_nifti(zscore(np.nanmean(ST,axis=0)), fn)


########################################################################
#Kitchen sink centrality loop, use Schaefer900
########################################################################
	NKI_avadj, MGH_avadj = gen_groupave_adj('Schaefer900')





# End of line
