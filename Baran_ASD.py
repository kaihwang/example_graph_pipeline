from graph_analysis import *
import glob
import nilearn
from nilearn.connectome import ConnectivityMeasure

subjects = pd.read_csv('/home/kahwang/bsh/Baran_ASD/subj_list.txt', header = None)
subjects = subjects[0]
df = pd.DataFrame(columns=['Subject'])

for ix, s in enumerate(subjects):

    #concate two runs
    try:
        fn = '/home/kahwang/bsh/Baran_ASD/%s/*scrubbed.nii' %s
        f_files = glob.glob(fn)
        fuctional_image = nilearn.image.concat_imgs(f_files)
    except:
        fn = '/home/kahwang/bsh/Baran_ASD/%s/sw*.nii' %s
        f_files = glob.glob(fn)
        fuctional_image = nilearn.image.concat_imgs(f_files)

    # load ts
    # below is the functional atlas appraoch.
    #atlases = ['/home/kahwang/bsh/Baran_ASD/ROIs/Schaefer400+YeoThalamus_17network_3mm.nii.gz', '/home/kahwang/bsh/Baran_ASD/ROIs/Schaefer400+YeoThalamus_7network_3mm.nii.gz']
    #CI = [np.loadtxt('/home/kahwang/bin/example_graph_pipeline/SchaefferYeo17_CI'), np.loadtxt('/home/kahwang/bin/example_graph_pipeline/SchaefferYeo7_CI')]
    #June 2021, use anatomical masks.
    atlases = ['/home/kahwang/bsh/Baran_ASD/ROIs/fsl_schaefer7.nii.gz', '/home/kahwang/bsh/Baran_ASD/ROIs/fsl_schaefer17.nii.gz']
    CI = [np.loadtxt('/home/kahwang/bin/example_graph_pipeline/fsl_schaefer7CI'), np.loadtxt('/home/kahwang/bin/example_graph_pipeline/fsl_schaefer17CI')]

    df.loc[ix, 'Subject'] = s

    for ia, atlas in enumerate(atlases):
        masker = NiftiLabelsMasker(labels_img=atlas, standardize=False)
        ts = masker.fit_transform(fuctional_image)

        # fc calculation
        correlation_measure = ConnectivityMeasure(kind='correlation')
        correlation_matrix = correlation_measure.fit_transform([ts])[0]
        np.fill_diagonal(correlation_matrix, 0)

        # pc calculation, threshoulding, ave across threshold
        max_cost = .15
        min_cost = .05
        PCs = np.zeros((len(np.arange(min_cost, max_cost+0.01, 0.01)), len(correlation_matrix)))

        for i, cost in enumerate(np.arange(min_cost, max_cost, 0.01)):
            tmp_matrix = threshold(correlation_matrix.copy(), cost)
            PCs[i,:] = bct.participation_coef(tmp_matrix, CI[ia])

        #average across costs
        PC = np.mean(PCs, axis=0)
        rois = np.arange(400, len(correlation_matrix))

        if ia == 0: #17 network
            for roi in rois:
                df.loc[ix, 'PC_17Network_ThaROI' + str(roi+1)] = PC[roi]

        if ia == 1: #7 network
            for roi in rois:
                df.loc[ix, 'PC_7Network_ThaROI' + str(roi+1)] = PC[roi]


df.to_csv('Thalamus_fslROIs_PC.csv')
