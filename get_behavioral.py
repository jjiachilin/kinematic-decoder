import csv
import numpy as np
from scipy import io
from tempfile import TemporaryFile

infile = open('./data/EN10-cluster_df.csv')
reader = csv.DictReader(infile)

# t: timepoints = 150 per reach
# n: num features = 3 (x, y, z)
# k: num trials (reaches = 127)
# retrieve data in the form k x t x [x_smooth, y_smooth, z_smooth]
# crop data for constant t

t = 0
reach = []
out = []
removed_indices = []
prev = ''
for row in reader:
    if row['date'] != 'd1':
        if len(reach) > 250:
            out.append(reach)
        else:
            out.append([])
        break
    if row['reach'] != prev and prev != '':
        if len(reach) > 250:
            out.append(reach[:250])
        else:
            out.append([])
        reach = []
    reach.append([row['X_smooth'], row['Y_smooth'], row['Z_smooth']])
    prev = row['reach']
out = np.array(out, dtype=object)

with open('EN10_d1_behavioral.npy', 'wb') as f:
    np.save(f, out)
