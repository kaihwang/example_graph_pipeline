# Is the reviewer crazy?
import numpy as np
import bct
from datetime import datetime

# assuming 4mm resolution
size = 36070
tstep = 240

print("start: ")
print(datetime.now())
# random matrix
ts = np.random.rand(size,tstep)
matrix = np.arctanh(np.corrcoef(ts))
matrix[matrix==np.inf] = 1.0
#matrix = np.random.rand(size,size)

# random community assignment
CI = np.random.randint(1, 10, size)

max_cost = .15
min_cost = .01

# import thresholded matrix to BCT, import partition, run WMD/PC
PC = np.zeros((len(np.arange(min_cost, max_cost+0.01, 0.01)), size))
WMD = np.zeros((len(np.arange(min_cost, max_cost+0.01, 0.01)), size))
EC = np.zeros((len(np.arange(min_cost, max_cost+0.01, 0.01)), size))
GC = np.zeros((len(np.arange(min_cost, max_cost+0.01, 0.01)), size))
SC = np.zeros((len(np.arange(min_cost, max_cost+0.01, 0.01)), size))
ST = np.zeros((len(np.arange(min_cost, max_cost+0.01, 0.01)), size))

for i, cost in enumerate(np.arange(min_cost, max_cost, 0.01)):

  tmp_matrix = bct.threshold_proportional(matrix, cost, copy=True)

  # # PC slow to compute, days per threshold
  # PC[i,:] = bct.participation_coef(tmp_matrix, CI)
  # fn = 'completed PC calculation for %s at:' %cost
  # print(fn)
  # print(datetime.now())
  #
  # WMD seems relatively fast, maybe 10min per threshold
  WMD[i,:] = bct.module_degree_zscore(tmp_matrix,CI)
  fn = 'completed WMD calculation for %s at:' %cost
  print(fn)
  print(datetime.now())
  #
  # EC[i,:] = bct.eigenvector_centrality_und(tmp_matrix)
  # fn = 'completed EC calculation for %s at:' %cost
  # print(fn)
  # print(datetime.now())

  # GC[i,:], _ = bct.gateway_coef_sign(tmp_matrix, CI)
  # fn = 'completed GC calculation for %s at:' %cost
  # print(fn)
  # print(datetime.now())
  #
  # SC[i,:] = bct.subgraph_centrality(tmp_matrix)
  # fn = 'completed SC calculation for %s at:' %cost
  # print(fn)
  # print(datetime.now())
  #
  # ST is fast to compute, ~10min per threshold
  ST[i,:] = bct.strengths_und(tmp_matrix)
  fn = 'completed ST calculation for %s at:' %cost
  print(fn)
  print(datetime.now())


print("All done at: ")
print(datetime.now())
