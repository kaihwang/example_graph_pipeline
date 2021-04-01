# script to run parcel-level hub analysis on ecog patients with rest data
from graph_analysis import *
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
from nilearn import masking
import nilearn


# files under  '/data/backed_up/kahwang/ECoG_fMRI'
Subjects = [456,320,524,437,439,369,538,409,477,430,372,394,525,493,514,404,399,376,362,335,418,378,316,518,521,460,561,405,434,483,242,307,532,413,533,423,507,400,416]
masker = NiftiLabelsMasker('/home/kahwang/bsh/ROIs/Schaefer400_7network_2mm.nii.gz')

def write_graph_to_vol_sch400_template_nifti(graph_metric, fn, resolution=400):
	'''short hand to write vol based nifti file of the graph metrics
	'''

	if resolution == 400:
		vol_template = nib.load('/home/kahwang/bsh/ROIs/Schaefer400_7network_2mm.nii.gz')
		roisize = 400
	else:
		print ('Error with template')
		return

	v_data = vol_template.get_fdata()
	graph_data = np.zeros((np.shape(v_data)))

	if resolution == 'voxelwise':
		for ix, i in enumerate(vox_index):
			graph_data[v_data == i] = graph_metric[ix]
	else:
		for i in np.arange(roisize):
			#key = roi_df['KEYVALUE'][i]
			graph_data[v_data == i+1] = graph_metric[i]

	new_nii = nilearn.image.new_img_like(vol_template, graph_data)
	#new_nii = nib.Nifti1Image(graph_data, affine = vol_template.affine, header = vol_template.header)
	nib.save(new_nii, fn)


for s in Subjects:
	fn = '/data/backed_up/kahwang/ECoG_fMRI/%s/rsOut_concat/allruns_reg_res4d_normandscaled_common.nii' %s
	ffiles = nib.load(fn)

	fn = '/data/backed_up/kahwang/ECoG_fMRI/%s/infomap/%s_Infomap_to_17Network_2mm.nii' %(s,s)
	Yeo17ci = nib.load(fn)

	fn = '/data/backed_up/kahwang/ECoG_fMRI/%s/infomap/%s_Infomap_to_7Network_2mm.nii' %(s,s)
	Yeo7ci = nib.load(fn)

	Yeo7ci = np.round(masker.fit_transform(Yeo7ci))
	Yeo17ci = np.round(masker.fit_transform(Yeo17ci))
	ts = masker.fit_transform(ffiles)
	mat = np.nan_to_num(np.corrcoef(ts.T))
	min_cost = .02
	max_cost = .10
	PC_7 = np.zeros((len(np.arange(min_cost, max_cost+0.01, 0.01)), 400))
	PC_17 = np.zeros((len(np.arange(min_cost, max_cost+0.01, 0.01)), 400))
	for i, cost in enumerate(np.arange(min_cost, max_cost, 0.01)):
		tmp_matrix = threshold(mat.copy(), cost)
		PC_7[i,:] = bct.participation_coef(tmp_matrix, Yeo7ci)
		PC_17[i,:] = bct.participation_coef(tmp_matrix, Yeo17ci)

	fn = '/home/kahwang/bin/example_graph_pipeline/phub_images/%s_PC_7network.nii.gz' %s
	write_graph_to_vol_sch400_template_nifti(np.nanmean(PC_7, axis=0), fn, resolution=400)
	fn = '/home/kahwang/bin/example_graph_pipeline/phub_images/%s_PC_17network.nii.gz' %s
	write_graph_to_vol_sch400_template_nifti(np.nanmean(PC_17, axis=0), fn, resolution=400)




#end of line
