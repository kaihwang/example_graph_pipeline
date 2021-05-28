# put a seed in the thalamus, run whole-brain FC
# moving away from AFNI, doing as much as we can in nilearn/python
# built based on example from here: https://nilearn.github.io/auto_examples/03_connectivity/plot_seed_to_voxel_correlation.html
import os
import numpy as np
import pandas as pd
import nibabel as nib
import nilearn
import matplotlib.pyplot as plt
from nilearn.input_data import NiftiLabelsMasker
from nilearn import masking
from nilearn import input_data
from nilearn import plotting
import glob
import multiprocessing

def generate_correlation_mat(x, y):
	"""Correlate each n with each m.

	Parameters
	----------
	x : np.array
	  Shape N X T.

	y : np.array
	  Shape M X T.

	Returns
	-------
	np.array
	  N X M array in which each element is a correlation coefficient.

	"""
	mu_x = x.mean(1)
	mu_y = y.mean(1)
	n = x.shape[1]
	if n != y.shape[1]:
		raise ValueError('x and y must ' +
						 'have the same number of timepoints.')
	s_x = x.std(1, ddof=n - 1)
	s_y = y.std(1, ddof=n - 1)
	cov = np.dot(x,
				 y.T) - n * np.dot(mu_x[:, np.newaxis],
								  mu_y[np.newaxis, :])
	return cov / np.dot(s_x[:, np.newaxis], s_y[np.newaxis, :])

########################################
##### Step 1 #####
########################################
## here we have two "seeds" in the thalamus from Aaron that we would like to calcuate its FC with the whole brain.
# This Sprenger study got it right, with the thalamic hot spot at −14/−23/1 and −14/−23/0, https://academic.oup.com/brain/article/135/8/2536/305095
seed1_coords = [(-14, -23, 1)]
seed2_coords = [(-14, -23, 0)]

#Always double check the coordinate.
#nilearn.plotting.plot_anat(cut_coords=seed1_coords[0])
#plt.show()
#nilearn.plotting.plot_anat(cut_coords=seed2_coords[0])
#plt.show()

seed1_masker = input_data.NiftiSpheresMasker(seed1_coords, radius=5)
seed2_masker = input_data.NiftiSpheresMasker(seed2_coords, radius=5)
brain_masker = input_data.NiftiMasker()

########################################
##### Step 2 loop through functional files for MGH subjects
########################################
#get list of all subjects
subjects = pd.read_csv('/home/kahwang/bin/example_graph_pipeline/MGH_Subjects', names=['ID'])['ID'].values
#input_files = glob.glob('/data/backed_up/shared/MGH/MGH/*/MNINonLinear/rfMRI_REST.nii.gz')

# a function to run seed FC and save outputs to a folder
def run_seed_fc(s):
	try:
		file = '/data/backed_up/shared/MGH/MGH/%s/MNINonLinear/rfMRI_REST.nii.gz' %s
		########################################
		##### Step 3 extract seed time-series from the two coordinates and whole brain
		########################################
		seed1_timeseries = seed1_masker.fit_transform(file)
		seed2_timeseries = seed2_masker.fit_transform(file)
		brain_time_series = brain_masker.fit_transform(file) #this is whole brain ts data

		########################################
		##### Step 4 use dot product for faster seed-to-voxel correlation calcuation
		########################################
		seed1_to_voxel_correlations = generate_correlation_mat(seed1_timeseries.T, brain_time_series.T) #need  to be space by time, so transpose
		seed2_to_voxel_correlations = generate_correlation_mat(seed2_timeseries.T, brain_time_series.T)

		########################################
		##### Step 5 put voxel-wise r values back into brain space
		########################################
		seed1_to_voxel_correlations_img = brain_masker.inverse_transform(seed1_to_voxel_correlations)
		seed2_to_voxel_correlations_img = brain_masker.inverse_transform(seed2_to_voxel_correlations)

		# save nii
		seed1_to_voxel_correlations_img.to_filename(('/data/backed_up/shared/MGH/MGH/outputs/%s_seed1_corr.nii.gz' %s))
		seed2_to_voxel_correlations_img.to_filename(('/data/backed_up/shared/MGH/MGH/outputs/%s_seed2_corr.nii.gz' %s))

		# plot to make sure it makes sense
		#plotting.plot_stat_map(seed1_to_voxel_correlations_img, threshold=0.2, vmax=1)

	except:
		print(('subject %s failed, check' %s))


## use parallel toolbox to run multiple subjects in parallel. Here we are doing it with 8 cores. check yoru system
pool = multiprocessing.Pool(8)
results = pool.map(run_seed_fc, subjects)
pool.close()
pool.join()

########################################
##### Step 6 do group stats with whatev you like, here I am just averaging the r values.
########################################
seed1_corrs = glob.glob('/data/backed_up/shared/MGH/MGH/outputs/*_seed1_corr.nii.gz')
seed2_corrs = glob.glob('/data/backed_up/shared/MGH/MGH/outputs/*_seed2_corr.nii.gz')

r_s = np.zeros((91,109,91)) #this is the standard MNI grid, 91 by 109 by 91 voxel in 3d space
for i, file in enumerate(seed1_corrs):
	r_s = r_s + nib.load(file).get_fdata()[:,:,:,0]
ave_corr1 = r_s / len(seed1_corrs)
ave_corr1 = nilearn.image.new_img_like(nib.load(file),ave_corr1)# turn into nii object for saving and plotting.


r_s = np.zeros((91,109,91))
for i, file in enumerate(seed2_corrs):
	r_s = r_s + nib.load(file).get_fdata()[:,:,:,0]
ave_corr2 = r_s / len(seed2_corrs)
ave_corr2 = nilearn.image.new_img_like(nib.load(file),ave_corr2) # turn into nii object for saving and plotting.

# plot to look at them
plotting.plot_stat_map(ave_corr1, threshold=0.1, vmax=.3)
plt.show()
plotting.plot_stat_map(ave_corr2, threshold=0.1, vmax=.3)
plt.show()

ave_corr1.to_filename('avefc_outputs/seed1_ave_corr.nii.gz')
ave_corr2.to_filename('avefc_outputs/seed2_ave_corr.nii.gz')

# end of example.
