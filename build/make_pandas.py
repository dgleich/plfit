import sys
from sys import path
path.append("/homes/eikmeier/plfit/src")
import numpy as np
import pandas as pd
import os
import re
import csv


def grab_vals(short_name, root):
    #Initialize Variables
    eig_x = "NaN"
    eig_a = "NaN"
    eig_p = "NaN"
    deg_x = "NaN"
    deg_a = "NaN"
    deg_p = "NaN"
    resample_x = "NaN"
    resample_a = "NaN"
    resample_p = "NaN"
    lap_x = "NaN"
    lap_a = "NaN"
    lap_p = "NaN"
    clust_x = "NaN"
    clust_a = "NaN"
    clust_p = "NaN"
    lap_file = None
    resamplelap_file = None
    clust_file = None
    orig_clust_file = None
    resamplelap_x = "NaN"
    resamplelap_a = "NaN"
    resamplelap_p = "NaN"
    max_eig = None
    orig_lap_eigs = None
    max_lap_eig = "NaN"
    size = None
    max_deg = None
    num_remaining_degs = None
    num_remaining_adjeigs = None
    num_remaining_lapeigs = "NaN"
    num_remaining_clust = "NaN"

    #Files that we will need to open
    eig_file = os.path.join(root, short_name+'.adjacency.plfit')
    deg_file = os.path.join(root,short_name+'.plfit')
    orig_degs = os.path.join(root,short_name+'.degs')
    orig_eigs = os.path.join(root, short_name+'.adjacency.eigs')
    resample_file = os.path.join(root, short_name+'.resample.adjacency.plfit')
    for root2, dirs, files in os.walk("/homes/eikmeier/plfit/data/laplacian-eigenvalues"):
        for file in files:
            if short_name in file and '.smat.laplacian.plfit' in file:
                lap_file = os.path.join(root2,file)
                orig_lap_eigs = os.path.join(root2, short_name+'.laplacian.eigs')
                resamplelap_file = os.path.join(root2,short_name+'.resample.laplacian.plfit')

    for root3, dirs, files in os.walk("/homes/eikmeier/plfit/data/graph-db-clustering-coefficients"):
        for file in files:
            if short_name in file and '.smat.ccoeffs.plfit' in file:
                clust_file = os.path.join(root3,file)
                orig_clust_file = os.path.join(root3,short_name+'.ccoeffs')
                
    for line in open(eig_file, 'r'):
        if 'xmin' in line:
            eig_x = re.sub('xmin  = ', '', line)
        elif 'alpha' in line:
            eig_a = re.sub('alpha = ', '', line)
        elif 'p     = ' in  line:
            eig_p = re.sub('p     = ', '', line)
    for line in open(deg_file, 'r'):
        if 'xmin' in line:
            deg_x = re.sub('xmin  = ', '', line)
        if 'alpha' in line:
            deg_a = re.sub('alpha = ', '', line)
        elif 'p     = ' in  line:
            deg_p = re.sub('p     = ', '', line)
    with open(orig_degs, 'r') as f:
        size = f.readline()
        degs = [float(line) for line in f]
        max_deg = max(degs)
        #count how many degrees are larger than xmin:
        num_remaining_degs = sum(float(i) > float(deg_x) for i in degs)        
    with open(orig_eigs, 'r') as f:
        eigs = [float(line) for line in f]
        max_eig = max(eigs)
        #count how many adj-eigs are larger than xmin:
        num_remaining_adjeigs = sum(float(i) > float(eig_x) for i in eigs)
    if os.path.exists(resample_file):
        for line in open(resample_file, 'r'):
            if 'xmin' in line:
                resample_x = re.sub('xmin  = ', '', line)
            elif 'alpha' in line:
                resample_a = re.sub('alpha = ', '', line)
            elif 'p     = ' in line:
                resample_p= re.sub('p     = ', '', line)

    if orig_clust_file is not None:
        with open(orig_clust_file,'r') as f:
            clustering = [float(line) for line in f]
            max_clustering = max(clustering)
            #Count how many clustering coefficients are larger than xmin:
            num_remaining_clust = sum(float(i) > float(clust_x) for i in clustering)
    
    if lap_file is not None:
        for line in open(lap_file, 'r'):
            if 'xmin' in line:
                lap_x = re.sub('xmin  = ', '', line)
            elif 'alpha' in line:
                lap_a = re.sub('alpha = ', '', line)
            elif 'p     = ' in line:
                lap_p = re.sub('p     = ', '', line)
    if orig_lap_eigs is not None:
        with open(orig_lap_eigs, 'r') as f:
            lap_eigs = [float(line) for line in f]
            max_lap_eig = max(lap_eigs)
            #count how many lap-eigs are larger than xmin:
            num_remaining_lapeigs = sum(float(i) > float(lap_x) for i in lap_eigs)
    if resamplelap_file is not None:
        if os.path.exists(resamplelap_file):
            for line in open(resamplelap_file, 'r'):
                if 'xmin' in line:
                    resamplelap_x = re.sub('xmin  = ', '', line)
                elif 'alpha' in line:
                    resamplelap_a = re.sub('alpha = ', '', line)
                elif 'p     = ' in line:
                    resamplelap_p = re.sub('p     = ', '', line)

    if clust_file is not None:
        for line in open(clust_file, 'r'):
            if 'xmin' in line:
                clust_x = re.sub('xmin  = ', '', line)
            elif 'alpha' in line:
                clust_a = re.sub('alpha = ', '', line)
            elif 'p     = ' in line:
                clust_p = re.sub('p     = ', '', line)

    #In case any values were missed, set them to Zero
    if max_eig is None:
        print("Couldn't find the maximum eigenvalue in", orig_eigs)
    if max_lap_eig is None:
        print("Couldn't find the maximum laplacian eigenvalue in", orig_lap_eigs)
    if size is None:
        print("Couldn't find the size in", orig_degs)
    if max_deg is None:
        print("Couldn't find the maximum degree in", orig_degs)


    return (int(size), float(max_eig), float(max_lap_eig), float(eig_x), float(eig_a), float(eig_p), float(deg_x), float(deg_a), float(deg_p), float(resample_x), float(resample_a), float(resample_p), float(lap_x), float(lap_a), float(lap_p), float(resamplelap_x), float(resamplelap_a), float(resamplelap_p), float(max_deg), float(num_remaining_degs), float(num_remaining_adjeigs), float(num_remaining_lapeigs), float(clust_x), float(clust_a), float(clust_p), float(num_remaining_clust))
    


#########################
## PHASE 1 
## GATHER THE INFO FROM THE PLFIT FILES
#########################

my_list = []
names_list = []
type_list = []
subtype_list = []

for root, dirs, files in os.walk("/homes/eikmeier/plfit/data/adjacency-eigenvalues-and-degrees/"):
    for file in files:
        if file.endswith(".adjacency.plfit") and not file.endswith(".resample.adjacency.plfit"):
            short_name = re.sub('.adjacency.plfit$', '', file)
            name = re.sub('.smat.adjacency.plfit$', '', file)
            deg_file = os.path.join(root,short_name+'.plfit')

            if any(c in file for c in ('copy', 'ff', 'pa-', 'rpl')):
                if os.path.exists(deg_file):
                    my_list.append(grab_vals(short_name, root))
                    names_list.append(name)
                    type_list.append('model')
                    if 'copy' in short_name:
                        subtype_list.append('Copying')
                    elif 'ff' in short_name:
                        subtype_list.append('Forest fire')
                    elif 'pa' in short_name:
                        subtype_list.append('PA')
                    elif 'rpl' in short_name:
                        subtype_list.append('RPL')
                    
            elif any(c in file for c in ('as19', 'as20', 'as-caida')):
                if os.path.exists(deg_file):
                    my_list.append(grab_vals(short_name, root))
                    names_list.append(name)
                    type_list.append('AS')
                    if 'as19' in short_name:
                        subtype_list.append('as19')
                    elif 'as20' in short_name:
                        subtype_list.append('as20')
                    elif 'as-caida' in short_name:
                        subtype_list.append('as-caida')
                        
            elif 'oregon' in file:
                if os.path.exists(deg_file):
                    my_list.append(grab_vals(short_name, root))
                    names_list.append(name)
                    type_list.append('Oregon')
                    if 'oregon1' in short_name:
                        subtype_list.append('oregon1')
                    elif 'oregon2' in short_name:
                        subtype_list.append('oregon2')

            elif 'p2p' in file:
                if os.path.exists(deg_file):
                    my_list.append(grab_vals(short_name, root))
                    names_list.append(name)
                    type_list.append('P2P')
                    subtype_list.append('')

            elif os.path.exists(deg_file):
                my_list.append(grab_vals(short_name, root))
                names_list.append(name)
                if 'porter' in root:
                    type_list.append('Facebook')
                    subtype_list.append('')
                elif 'Erdos' in short_name:
                    type_list.append('Erdos')
                    subtype_list.append('')
                else:
                    type_list.append('other')
                    if any(c in file for c in ('celegans_metabolic','dmela-cc','homo-cc','musm-cc','scere-cc','yeast-cc')):
                        subtype_list.append('Biology')
                    elif any(c in file for c in ('Kohonen-wcc', 'Lederberg-wcc', 'patents_main-wcc', 'SciMet-wcc', 'SmaGri-wcc', 'Zewail-wcc')):
                        subtype_list.append('Citation')
                    elif any(c in file for c in ('arxiv-ubc', 'astro-ph-cc', 'ca-AstroPh-cc', 'ca-CondMat-cc', 'ca-GrQc-cc', 'ca-HepPh-cc', 'ca-HepTh-cc', 'cond-mat-2003-cc', 'cond-mat-cc', 'dblp-ubc', 'hep-th-cc', 'netscience-cc', 'geom-cc')):
                        subtype_list.append('Collab.')
                    elif any(c in file for c in ('lesmis', 'marvel-chars-cc', 'marvel-comics-cc')):
                        subtype_list.append('Fiction')
                    elif any (c in file for c in ('tapir', 'usroads-cc')):
                        subtype_list.append('Low-dim')
                    elif any (c in file for c in ('CSphd-wcc', 'EVA-wcc', 'football', 'netflix-drop-4-nn-10-cc', 'pgp-cc', 'polbooks')):
                        subtype_list.append('Relational')
                    elif any (c in file for c in ('dolphins', 'email', 'email-Enron-cc', 'jazz', 'karate', 'soc-Epinions1-wcc', 'soc-Slashdot0811-wcc', 'soc-Slashdot0902-wcc', 'wiki-Vote-wcc')):
                        subtype_list.append('Social')
                    elif any (c in file for c in ('as-22july06', 'itdk0304-cc', 'power', 'USpowerGrid')):
                        subtype_list.append('Tech.')
                    elif any (c in file for c in ('California-wcc', 'EPA-wcc', 'polblogs-sym-cc', 'stanford-cs-sym', 'web-NotreDame-wcc')):
                        subtype_list.append('Web')
                    elif any (c in file for c in ('adjnoun', 'dico-sym-cc', 'dictionary28-cc', 'EAT_RS-wcc', 'FA-wcc', 'foldoc-wcc', 'ODLIS-wcc', 'Reuters911-cc', 'Roget-wcc', 'Wordnet3-wcc')):
                        subtype_list.append('Word')
                    else:
                        subtype_list.append('')


#########################
## PHASE 2
## Save the values to a pd data frame
#########################

my_array = np.transpose(np.asarray(my_list))




d = {'name' : names_list, 'types' : type_list, 'subtype' : subtype_list,
     'size' : pd.Series((my_array)[0]), 'largest_eig' : pd.Series((my_array)[1]), 'largest_lap_eig' : pd.Series((my_array)[2]), 
     'adj_xmin' : pd.Series((my_array)[3]), 'adj_alpha' : pd.Series((my_array)[4]), 'adj_p' : pd.Series((my_array)[5]),
     'deg_xmin' : pd.Series((my_array)[6]), 'deg_alpha' : pd.Series((my_array)[7]), 'deg_p' : pd.Series((my_array)[8]),
     'resample_xmin' : pd.Series((my_array)[9]), 'resample_alpha' : pd.Series((my_array)[10]), 'resample_p' : pd.Series((my_array)[11]),
     'lap_xmin' : pd.Series((my_array)[12]), 'lap_alpha' : pd.Series((my_array)[13]), 'lap_p' : pd.Series((my_array)[14]),
     'resample_lap_xmin' : pd.Series((my_array)[15]), 'resample_lap_alpha' : pd.Series((my_array)[16]), 'resample_lap_p' : pd.Series((my_array)[17]),
     'max_deg': pd.Series((my_array)[18]),
     'remaining_degs' : pd.Series((my_array)[19]), 'remaining_adj_eigs' : pd.Series((my_array)[20]), 'remaining_lap_eigs' : pd.Series((my_array)[21]),
     'clust_xmin' : pd.Series((my_array)[22]), 'clust_alpha' : pd.Series((my_array)[23]), 'clust_p' : pd.Series((my_array)[24])}


df = pd.DataFrame(d)
df.drop_duplicates(subset='name')



df.to_csv('Matrix_Info')


#pd.DataFrame.from_csv(


