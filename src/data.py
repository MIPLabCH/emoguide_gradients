"""
@author: Chun Hei Michael Chan
@copyright: Private Copyright
@credits: [Chun Hei Michael Chan]
@maintainer: Chun Hei Michael Chan
@email: miki998chan@gmail.com
"""

import json
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.constants import *

def df_to_timeseries(df, filename):
    """
    Information:
    ------------
    Read our formatted dataframes to obtain timeseries 
    in (time,voxels) format of a specific acquisition

    Parameters
    ----------
    df      ::[DataFrame]
        DataFrame containing mri info
    
    filename::[string]
        Name of the acquisition we want to single out


    Returns
    -------
    series  ::[2darray<float>]
        timeseries that we are interested in 
    cur_file::[DatFrame]
        DataFrame of info about the filename we gave as input
    """
    
    cur_file = df[df.filename == filename]
    nbv, nbt = int(cur_file.vindex.max()) + 1, len(cur_file[cur_file.vindex==0])
    series   = np.zeros((nbt,nbv))

    for v in range(nbv):
        series[:,v] = np.array(cur_file[cur_file.vindex==v]['score'])
    return series, cur_file


def format_bids(path2data='./data/emoFiles/EmoBIDS/'):
    """
    Information:
    ------------
    Merge the different emotion data in BIDS csv format and arrange them
    WARNING: very specific function to my usage (filenames are hard-coded)

    Parameters
    ----------
    path2data::[String]

    Returns
    -------
    emo_df::[pandas Dataframe]
    """

    movienames = list(FILM2DURATION.keys())
    record = {'filename': [], 'item': [], 'score': []}
    for mname in movienames:
        # Wrote this quick fix because of mismatch between files for the name "BetweenViewing"/(s)
        if mname == 'BetweenViewing':
            mname = 'BetweenViewings'

        filenamejson = 'Annot13_{}_stim.json'.format(mname)
        filenametsv  = 'Annot13_{}_stim.tsv'.format(mname)
        
        # We load the emotions' index in the other file
        with open(path2data+'{}'.format(filenamejson)) as emoindex:
            tmp = json.load(emoindex)
        
        # We load the scores for each movies
        double_array = np.loadtxt(path2data+'{}'.format(filenametsv), delimiter='\t')
        nbt, nbitem  = double_array.shape
        
        for item_idx in range(nbitem):
            record['score'] += list(double_array[:,item_idx])
            record['item']  += ([tmp['Columns'][item_idx]] * nbt)
            
        if mname == 'BetweenViewings':
            record['filename'] += (['W_{}13.csv'.format(mname[:-1])] * nbt * nbitem)
        else:
            record['filename'] += (['W_{}13.csv'.format(mname)] * nbt * nbitem)

    emo_df = pd.DataFrame.from_dict(record)

    return emo_df



def format_timecourse_cortical(path2mrifolder, path2label):
    """
    Information:
    ------------
    Merge the different fmri timecourse data and arrange them including only cortical 400 regions
    WARNING: very specific function to my usage (filenames are hard-coded)

    Parameters
    ----------
    path2mrifolder::[String]
        path to fmri timecourse
    path2label    ::[String]
        path to file for 7 yeo network index labels

    Returns
    -------
    list_mri_df::[list<pandas Dataframe>]
    names      ::[list<String>]
    """

    labels = pd.read_csv(path2label)
    files = os.listdir(path2mrifolder)
    names = set([f.split('_')[4] for f in files])
    names = [n.strip('.csv') for n in names]

    list_mri_df = []
    for idx in tqdm(range(len(names))):
        files_specific = [f for f in files if names[idx] in f]

        # we keep a dictionary with each film and their respective scores for each item in a double array
        record = {'filename': [], 'vindex': [],'full_region': [],
                    'partial_region': [],'score': [], 'movie_name':[], 'parcel': []
                    ,'id': [], 'session': []}
        for filename in tqdm(files_specific):
            _, parcel, _id, session, moviename = filename.split('_')
            moviename    = moviename.strip('.csv')
            double_array = np.loadtxt(path2mrifolder + '/' + filename, delimiter=',')
            nbt, nbvoxel = double_array.shape
            
            for voxel_idx in range(nbvoxel):
                record['score'] += list(double_array[:,voxel_idx])
                record['vindex'] += [voxel_idx] * nbt
                
                region_name  = labels[labels['ROI Label'] == voxel_idx + 1]['ROI Name'].item()
                partial_name = region_name.split('_')[2]
                record['full_region']    += ([region_name] * nbt)
                record['partial_region'] += ([partial_name] * nbt)


            record['filename']   += ([filename] * nbt * nbvoxel)
            record['movie_name'] += ([moviename] * nbt * nbvoxel)
            record['parcel']     += ([parcel] * nbt * nbvoxel)
            record['id']         += ([_id] * nbt * nbvoxel)
            record['session']    += ([session] * nbt * nbvoxel)

        mri_df = pd.DataFrame.from_dict(record)
        list_mri_df.append(mri_df)

    return list_mri_df, names


def combine_timecourse_subcortical(path2mri400, path2labelsub):
    """
    Information:
    ------------
    Merge the different fmri timecourse data and arrange them 
    combining 400 regions with subcortical 14 regions

    WARNING: very specific function to my usage (filenames are hard-coded)

    Parameters
    ----------
    path2mri400::[String]
        path to fmri compiled version generated by function "format_timecourse_cortical"
    path2labelsub::[String]
        path to file for subcortical index labels

    Returns
    -------
    mri_df::[pandas Dataframe]
    """

    with open(path2labelsub, 'r') as f:
        text = f.readlines()
        text = [t.strip() for t in text]
    compiled_list = os.listdir(path2mri400)

    list_mri_df = []
    for cfile in tqdm(compiled_list):
        mri_df = pd.read_csv(path2mri400+'{}'.format(cfile))

        subcort = mri_df[mri_df.parcel == 14]
        tmp_df  = mri_df.drop(subcort.index)

        # We reset the index the regions, adding the 14 subcortical regions to the 400 parcellations
        # And we then call the subcortical area to be the "Sub" network
        subcort['full_region']    = np.array(text)[np.array(subcort.vindex)]
        subcort['vindex']         = 400 + subcort['vindex'] 
        subcort['partial_region'] = 'Sub'

        res_df = pd.concat([tmp_df,subcort])

        rename = lambda x: 'TC_414_'+'_'.join(np.array(x.split('_'))[[2,4]])
        res_df['filename'] = res_df['filename'].apply(rename)

        list_mri_df.append(res_df)

    return list_mri_df, compiled_list