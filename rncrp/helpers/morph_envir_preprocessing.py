import numpy as np
import scipy as sp
import scipy.io as spio
import scipy.interpolate
import sqlite3 as sql
import pandas as pd
import pickle
from datetime import datetime
from glob import glob
import os.path
from astropy.convolution import convolve, Gaussian1DKernel
from rncrp.helpers import morph_envir_utilities as u

######################################################################
### Preprocessing functions for loading morph environment dataset

def rep_dist(S_trial_mat,morphs,metric='cosine', pop=False,ftest = False):

    if pop:
        dist = np.zeros([1,morphs.shape[0]])
        S_trial_mat = S_trial_mat.reshape(S_trial_mat.shape[0],-1)
        S_trial_mat = S_trial_mat[:,:,np.newaxis]
        print(S_trial_mat.shape)
    else:
        dist = np.zeros((S_trial_mat.shape[-1],morphs.shape[0]))
    
    if metric in ["cosine","cos_llr"]:
            S_tmat = S_trial_mat/(np.linalg.norm(S_trial_mat,axis=1,ord=2,keepdims=True)+1E-8)
        
    for trial in range(morphs.shape[0]):
        mask0, mask1 = morphs==0, morphs==1
        if morphs[trial]==0:
            mask0[trial]=False
        elif morphs[trial]==1:
            mask1[trial]=False

        if metric == 'cosine':
            # calculate centroids
                
            centroid0, centroid1 = np.nanmean(S_trial_mat[mask0,:,:],axis=0,keepdims=True), np.nanmean(S_trial_mat[mask1,:,:],axis=0,keepdims=True)
            centroid0=centroid0/(np.linalg.norm(centroid0,ord=2,axis=1,keepdims=True)+1E-8)
            centroid1=centroid1/(np.linalg.norm(centroid1,ord=2,axis=1,keepdims=True)+1E-8)

            # similarity to two centroids
            if pop:
#                 print(np.linalg.norm(centroid0))
                angle0,angle1 = np.dot(S_tmat[trial,:,:].T,centroid0.ravel()),np.dot(S_tmat[trial,:,:].T,centroid1.ravel())
#                 print(angle0,angle1)
            else:
                angle0,angle1 = np.diagonal(np.matmul(S_tmat[trial,:,:].T,np.squeeze(centroid0))),np.diagonal(np.matmul(S_tmat[trial,:,:].T,np.squeeze(centroid1)))
    #         # whole trial similarity fraction
            dist[:,trial] = np.squeeze(angle1/(angle0+angle1+1E-8))
    #         cd[:,trial]= np.diagonal(np.matmul(S_trial_mat[trial,:,:].T,cd_))
            
        elif metric == 'cos_llr':
             # calculate centroids
            centroid0, centroid1 = np.nanmean(S_trial_mat[mask0,:,:],axis=0,keepdims=True), np.nanmean(S_trial_mat[mask1,:,:],axis=0,keepdims=True)
            centroid0=centroid0/(np.linalg.norm(centroid0,ord=2,axis=1,keepdims=True)+1E-8)
            centroid1=centroid1/(np.linalg.norm(centroid1,ord=2,axis=1,keepdims=True)+1E-8)
            
            # similarity to two centroids
            angle0,angle1 = np.diagonal(np.matmul(S_tmat[trial,:,:].T,centroid0)),np.diagonal(np.matmul(S_tmat[trial,:,:].T,centroid1))
            dist[:,trial]= np.log(angle1)-np.log(angle0)
        elif metric == "cd":
            centroid0, centroid1 = np.nanmean(S_trial_mat[mask0,:,:],axis=0,keepdims=True), np.nanmean(S_trial_mat[mask1,:,:],axis=0,keepdims=True)
            
            cd = centroid0 - centroid1
            cd = cd/np.linalg.norm(cd,ord=2,keepdims=True,axis=1)
            dist[:,trial]= np.diagonal(np.matmul(S_trial_mat[trial,:,:].T,cd[0,:,:]))
        elif metric == "euclidean":
            centroid0, centroid1 = np.nanmean(S_trial_mat[mask0,:,:],axis=0), np.nanmean(S_trial_mat[mask1,:,:],axis=0)
            
            dist0 = np.mean((S_trial_mat[trial,:,:]-centroid0)**2,axis=0)
            dist1 = np.mean((S_trial_mat[trial,:,:]-centroid1)**2,axis=0)
            
            dist[:,trial] = dist1/(dist0+dist1)
        elif metric == "euc_llr":
            centroid0, centroid1 = np.nanmean(S_trial_mat[mask0,:,:],axis=0), np.nanmean(S_trial_mat[mask1,:,:],axis=0)
            
            dist0 = np.mean((S_trial_mat[trial,:,:]-centroid0)**2,axis=0)
            dist1 = np.mean((S_trial_mat[trial,:,:]-centroid1)**2,axis=0)
            dist[:,trial] = np.log(dist1)-np.log(dist0)
            
    if pop:  
        return dist
    else:
        
        if ftest:
            grandmean = np.nanmean(S_trial_mat,axis=0,keepdims=True)
            gm_dists = []
            for c in [0,.25,.5,.75,1]:
                gm_dists.append(np.nanmean((S_trial_mat[morphs==c,:,:]-grandmean)**2, axis=1))
            fvals = np.zeros((S_trial_mat.shape[-1],))
            pvals = np.zeros((S_trial_mat.shape[-1],))
            for cell in range(S_trial_mat.shape[-1]):   
                _f,_p = sp.stats.f_oneway(gm_dists[0][:,cell],gm_dists[1][:,cell],gm_dists[2][:,cell],gm_dists[3][:,cell],gm_dists[4][:,cell])
                fvals[cell] = _f
                pvals[cell]= _p
            return dist,fvals, pvals
        else:                
            mask0, mask1 = morphs==0, morphs==1
            centroid0, centroid1 = np.nanmean(S_trial_mat[mask0,:,:],axis=0), np.nanmean(S_trial_mat[mask1,:,:],axis=0)
            return dist, np.mean((centroid1-centroid0)**2,axis=0)
    
    
def single_sess_dist(sess,metric = "euclidean"):
    with open(os.path.join("D:\\Suite2P_Data\\",sess["MouseName"],"%s_%s_%i.pkl" % (sess["Track"],sess["DateFolder"],sess["SessionNumber"])),'rb') as f:
            data = pickle.load(f)
    VRDat,S = data['VRDat'],data['S']
    S[np.isnan(S)]=0
    S = S/np.percentile(S,95,axis=0)
    trial_info,S_trial_mat= data['trial_info'],data['S_trial_mat']
    S_trial_mat[np.isnan(S_trial_mat)]=0
    dist,centroid_diff = rep_dist(S_trial_mat,trial_info['morphs'],metric=metric,pop=False)
    
    return dist, centroid_diff, S_trial_mat, trial_info


def centroid_diff_perm_test(centroid_diff,S_trial_mat,trial_info,nperms=1000):
    centroid_diff_shuff = np.zeros([S_trial_mat.shape[-1],nperms])
    extreme_mask = (trial_info['morphs']==0) + (trial_info['morphs']==1)
    S_trial_mat_ext = S_trial_mat[extreme_mask,:,:]
    print(S_trial_mat_ext.shape)
    morphs_ext = trial_info['morphs'][extreme_mask]
    for p in range(nperms):
        _morph = morphs_ext[np.random.permutation(morphs_ext.shape[0])]
        centroid_diff_shuff[:,p] = np.mean((S_trial_mat_ext[_morph==0,:,:].mean(axis=0)-S_trial_mat_ext[_morph==1,:,:].mean(axis=0))**2,axis=0)
        
    return np.array(centroid_diff[:,np.newaxis]<centroid_diff_shuff,dtype=np.float).mean(axis=1)


def regress_distance(dist,morphs,x = np.linspace(-.1,1.1,num=50)) :
    dist_reg = np.zeros([dist.shape[0],x.shape[0]])
    for j in range(dist.shape[0]):
        model = sk.neighbors.KNeighborsRegressor(n_neighbors=10)
        model.fit(morphs[:,np.newaxis],dist[j,:])
        dist_reg[j,:]=model.predict(x[:,np.newaxis])
        
    return dist_reg


######################################################################

def loadmat_sbx(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    info = _check_keys(data)['info']
    # Defining number of channels/size factor
    if info['channels'] == 1:
        info['nChan'] = 2; factor = 1
    elif info['channels'] == 2:
        info['nChan'] = 1; factor = 2
    elif info['channels'] == 3:
        info['nChan'] = 1; factor = 2

    if info['scanmode'] == 0:
        info['recordsPerBuffer'] *= 2

     # Determine number of frames in whole file
    info['max_idx'] = int(os.path.getsize(filename[:-4] + '.sbx')/info['recordsPerBuffer']/info['sz'][1]*factor/4-1)
    info['fr']=info['resfreq']/info['config']['lines']*(2-info['scanmode'])
    return info


def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''

    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''

    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def load_scan_sess(sess,plane=0,fneu_coeff=.7):
    '''loads imaging aligned behavioral data and neural data for a single session
        inputs:
            sess: row from pandas array of session metadata
                    can also be a dictionary as long as the following fields are present and valid
                    'data file' - raw VR *.sqlite data file path
                    'scanmat' - Neurolabware .mat file path
                    's2pfolder' - Suite2P output path
            plane: which imaging plane to load - zero indexed
            fneu_coeff: coefficient to multiply neuropil coefficient by for dF/F calculation
        outputs:
            VRDat: imaging aligned behavioral data as pandas array (timepoints x number of variables)
            C: dF/F (timepoints x neurons)
            S: deconcolved activity rate (timepoints x neurons)
            F_: raw extracted fluorescence (timepoints x neurons)'''

    # load aligned VR Data
    VRDat = behavior_dataframe(sess['data file'],scanmats=sess['scanmat'],concat=False)

    # load imaging
    info = loadmat_sbx(sess['scanmat'])

    # suite2p output folder
    folder = os.path.join(sess['s2pfolder'],'plane%i' % plane)

    F= np.load(os.path.join(folder,'F.npy')) # raw extracted fluorescence
    Fneu = np.load(os.path.join(folder,'Fneu.npy')) # neuropil fluorescence
    iscell =  np.load(os.path.join(folder,'iscell.npy')) # mask for whether ROI is a cell or not
    S = np.load(os.path.join(folder,'spks.npy')) # deconvolved activity rate
    F_ = F-fneu_coeff*Fneu #neuropil correction
    C=F_[iscell[:,0]>0,:].T
    C = u.dff(C) # dF/F
    S=S[iscell[:,0 ]>0,:].T
    frame_diff = VRDat.shape[0]-C.shape[0] # make sure that VR data and imaging data are the same length
    print('frame diff',frame_diff)
    assert (frame_diff==0), "something is wrong with aligning VR and calcium data"
    return VRDat,C,S,F_[iscell[:,0]>0,:].T


def load_session_db(dir = "G:\\My Drive\\",scandir="TwoTower"):
    '''open the sessions sqlite database and add some columns
        inputs:
            dir: base directory for VR data

        outputs:
            df: pandas array 'dataframe' which contains metadata for all sessions '''

    # find sqlite file that contains metadata
    vr_fname = os.path.join(dir,"VR_Data","TwoTower","behavior.sqlite")

    # open a connections to pandas dataframe
    conn = sql.connect(vr_fname)
    df = pd.read_sql("SELECT MouseName, DateFolder, SessionNumber,Track, RewardCount, Imaging, ImagingRegion FROM sessions",conn)
    sdir = os.path.join(dir,"VR_Data","TwoTower")

    # convert file name to date time
    df['DateTime'] = [datetime.strptime(s,'%d_%m_%Y') for s in df['DateFolder']]

    # build data file name
    df['data file'] = [ build_VR_filename(df['MouseName'].iloc[i],
                                           df['DateFolder'].iloc[i],
                                           df['Track'].iloc[i],
                                           df['SessionNumber'].iloc[i],serverDir=sdir) for i in range(df.shape[0])]

    twop_dir = os.path.join(dir,"2P_Data",scandir)

    # build folders for 2P data locations
    # .mat
    df['scanmat'] = [build_2P_filename(df.iloc[i],serverDir=twop_dir) for i in range(df.shape[0])]
    # add s2p filefolder
    df['s2pfolder']=[build_s2p_folder(df.iloc[i],serverDir=twop_dir) for i in range(df.shape[0])]

    # close connection to sqlite file
    conn.close()
    return df

def build_s2p_folder(df,serverDir="G:\\My Drive\\2P_Data\\TwoTower\\"):
    '''build Suite2P results folder from metadata information. called internally from load_session_db
        inputs:
            df: single row from metadata array
        outputs:
            Suite2p results path'''

    # build folder name
    res_folder = os.path.join(serverDir,df['MouseName'],df['DateFolder'],df['Track'],"%s_*%s_*" % (df['Track'],df['SessionNumber']),'suite2p')
    # check for potential matches
    match= glob(res_folder)
    assert len(match)<2, "Suite2P multiple matching subfolders"
    if len(match)<1:
        return None
    else:
        return match[0]



def build_2P_filename(df,serverDir = "G:\\My Drive\\2P_Data\\TwoTower\\"):
    ''' use sessions database inputs to build appropriate filenames for 2P data
    called internally from load_session_db
    inputs: same as build_s2p_folder
    outputs: path to Neurolabware *.mat file'''

    mouse,date,scene,sess = df["MouseName"],df["DateFolder"],df["Track"],df["SessionNumber"]
    info_fname = os.path.join(serverDir,mouse,date,scene,"%s_*%s_*[0-9].mat" % (scene,sess))
    info_file = glob(info_fname)
    if len(info_file)>0:
        match= glob(info_file[0])
        assert len(match)<2, "2P .mat: multiple matching files"
        if len(match)==0:
            return None
        else:
            return info_file[0]
    else:
        return None

def build_VR_filename(mouse,date,scene,session,serverDir = "G:\\My Drive\\VR_Data\\TwoTower\\",verbose = False):
    '''use sessions database to build filenames for behavioral data (also a
    sqlite database)
    called internally from load_session_db
    inputs: same as build_s2p_folder
    outputs: path to Unity created .sqlite beahvioral data
    '''
    fname = os.path.join(serverDir,mouse,date,"%s_%s.sqlite" % (scene,session))
    file=glob(fname)

    if len(file)==1:
        return file[0]
    elif len(file)>1:
        if verbose:
            print("%s\\%s\\%s\\%s_%s.sqlite" % (serverDir, mouse, date, scene, session))
            print("file doesn't exist or multiples, errors to come!!!")
    else:
        if verbose:
            print("%s\\%s\\%s\\%s_%s.sqlite" % (serverDir, mouse, date, scene, session))
            print("file doesn't exist or multiples, errors to come!!!")

def behavior_dataframe(filenames,scanmats=None,concat = True,fix_teleports=True):
    '''Load a list of vr sessions given filenames. Capable of concatenating for
    averaging data across sessions. If scanmats is not None, aligns VR data to
    imaging data.
    inputs:
        filenames - string or list of strings with paths to VR sqlite files
        scanmats- string or list of strings with paths to .mat files from 2P data
        concat- bool, whether or not to concatenate sessions
    outpus:
        df/frames - [aligned][concatenated] VR dataframe/list of VR dataframes
    '''
    # print(filenames)
    # if there is imaging data
    if scanmats is None:
        # if multiple sessions
        if isinstance(filenames,list):
            frames = [_VR_interp(_get_frame(f)) for f in filenames] # load data and interpolate onto even time grid
            df = pd.concat(frames,ignore_index=True) # concatenate data
        else: #load single session
            df = _VR_interp(_get_frame(filenames,fix_teleports=fix_teleports))
        df['trial number'] = np.cumsum(df['teleport']) # fix trial numbers

        if isinstance(filenames,list):
            if concat:
                return df
            else:
                return frames
        else:
            return df

    else:
        if isinstance(filenames,list):
            # check to make sure number of scanmats and sqlite files is the same
            if len(filenames)!=len(scanmats):
                raise Exception("behavior and scanfile lists must be of the same length")
            else:
                # load and align all VR data
                frames = [_VR_align_to_2P(_get_frame(f,fix_teleports=fix_teleports),s) for (f,s) in zip(filenames,scanmats)]
                df = pd.concat(frames,ignore_index=True)
        else:
            df = _VR_align_to_2P(_get_frame(filenames,fix_teleports=fix_teleports),scanmats)

        df['trial number'] = np.cumsum(df['teleport'])

        if isinstance(filenames,list):
            if concat:
                return df
            else:
                return frames
        else:
            return df

def _VR_align_to_2P(vr_dframe,infofile, n_imaging_planes = 1,n_lines = 512.):
    '''align VR behavior data to 2P sample times using splines. called internally
    from behavior_dataframe if scanmat exists
    inputs:
        vr_dframe- VR pandas dataframe loaded directly from .sqlite file
        infofile- path
        n_imaging_planes- number of imaging planes (not implemented)
        n_lines - number of lines collected during each frame (default 512.)
    outputs:
        ca_df - calcium imaging aligned VR data frame (pandas dataframe)
    '''

    info = loadmat_sbx(infofile) # load .mat file with ttl times
    fr = info['fr'] # frame rate

    print('frame_rate',fr)
    lr = fr*n_lines # line rate

    ## on Feb 6, 2019 noticed that AA's new National Instruments board
    ## created a floating ground on my TTL circuit. This caused a bunch of extra TTLs
    ## due to unexpected grounding of the signal.


    orig_ttl_times = info['frame']/fr + info['line']/lr # including error ttls
    dt_ttl = np.diff(np.insert(orig_ttl_times,0,0)) # insert zero at beginning and calculate delta ttl time
    tmp = np.zeros(dt_ttl.shape)
    tmp[dt_ttl<.005] = 1 # find ttls faster than 200 Hz (unrealistically fast - probably a ttl which bounced to ground)
    # ensured outside of this script that this finds the true start ttl on every scan
    mask = np.insert(np.diff(tmp),0,0) # find first ttl in string that were too fast
    mask[mask<0] = 0
    print('num aberrant ttls',tmp.sum())

    frames = info['frame'][mask==0] # should be the original ttls up to a 1 VR frame error
    lines = info['line'][mask==0]

    ##
    ##

    # times of each ttl (VR frame)
    ttl_times = frames/fr + lines/lr
    numVRFrames = frames.shape[0]

    # create empty pandas dataframe to store calcium aligned data
    ca_df = pd.DataFrame(columns = vr_dframe.columns, index = np.arange(info['max_idx']))
    ca_time = np.arange(0,1/fr*info['max_idx'],1/fr) # time on this even grid
    if (ca_time.shape[0]-ca_df.shape[0])==1: # occaionally a 1 frame correction due to
                                            # scan stopping mid frame
        print('one frame correction')
        ca_time = ca_time[:-1]


    ca_df.loc[:,'time'] = ca_time
    mask = ca_time>=ttl_times[0] # mask for when ttls have started on imaging clock
                                # (i.e. imaging started and stabilized, ~10s)

    # take VR frames for which there are valid TTLs
    vr_dframe = vr_dframe.iloc[-numVRFrames:]
    # print(ttl_times.shape,vr_dframe.shape)

    # linear interpolation of position
    print(ttl_times[0],ttl_times[-1])
    print(ca_time[mask][0],ca_time[mask][-1])
    f_mean = sp.interpolate.interp1d(ttl_times,vr_dframe['pos']._values,axis=0,kind='slinear')
    ca_df.loc[mask,'pos'] = f_mean(ca_time[mask])
    ca_df.loc[~mask,'pos']=-500.

    # nearest frame interpolation
    near_list = ['morph','towerJitter','wallJitter','bckgndJitter']
    f_nearest = sp.interpolate.interp1d(ttl_times,vr_dframe[near_list]._values,axis=0,kind='nearest')
    ca_df.loc[mask,near_list] = f_nearest(ca_time[mask])
    ca_df.fillna(method='ffill',inplace=True)
    ca_df.loc[~mask,near_list]=-1.

    # integrate, interpolate and then take difference, to make sure data is not lost
    cumsum_list = ['dz','lick','reward','tstart','teleport','rzone']
    f_cumsum = sp.interpolate.interp1d(ttl_times,np.cumsum(vr_dframe[cumsum_list]._values,axis=0),axis=0,kind='slinear')
    ca_cumsum = np.round(np.insert(f_cumsum(ca_time[mask]),0,[0,0, 0,0 ,0,0],axis=0))
    if ca_cumsum[-1,-2]<ca_cumsum[-1,-3]:
        ca_cumsum[-1,-2]+=1


    ca_df.loc[mask,cumsum_list] = np.diff(ca_cumsum,axis=0)
    ca_df.loc[~mask,cumsum_list] = 0.

    # fill na here
    ca_df.loc[np.isnan(ca_df['teleport']._values),'teleport']=0
    ca_df.loc[np.isnan(ca_df['tstart']._values),'tstart']=0


    # smooth instantaneous speed
    k = Gaussian1DKernel(5)
    cum_dz = convolve(np.cumsum(ca_df['dz']._values),k,boundary='extend')
    ca_df['dz'] = np.ediff1d(cum_dz,to_end=0)


    ca_df['speed'].interpolate(method='linear',inplace=True)
    ca_df['speed']=np.array(np.divide(ca_df['dz'],np.ediff1d(ca_df['time'],to_begin=1./fr)))
    ca_df['speed'].iloc[0]=0

    # calculate and smooth lick rate
    ca_df['lick rate'] = np.array(np.divide(ca_df['lick'],np.ediff1d(ca_df['time'],to_begin=1./fr)))
    ca_df['lick rate'] = convolve(ca_df['lick rate']._values,k,boundary='extend')

    # replace nans with 0s
    ca_df[['reward','tstart','teleport','lick','towerJitter','wallJitter','bckgndJitter']].fillna(value=0,inplace=True)
    return ca_df

def _VR_align_to_2P_FlashLED(vr_dframe,infofile, n_imaging_planes = 1,n_lines = 512.):
    '''align VR behavior data to 2P sample times using splines. called internally
    from behavior_dataframe if scanmat exists
    inputs:
        vr_dframe- VR pandas dataframe loaded directly from .sqlite file
        infofile- path
        n_imaging_planes- number of imaging planes (not implemented)
        n_lines - number of lines collected during each frame (default 512.)
    outputs:
        ca_df - calcium imaging aligned VR data frame (pandas dataframe)
    '''

    info = loadmat_sbx(infofile) # load .mat file with ttl times
    fr = info['fr'] # frame rate
    lr = fr*n_lines # line rate


    orig_ttl_times = info['frame']/fr + info['line']/lr # including error ttls
    dt_ttl = np.diff(np.insert(orig_ttl_times,0,0)) # insert zero at beginning and calculate delta ttl time
    tmp = np.zeros(dt_ttl.shape)
    tmp[dt_ttl<.005] = 1 # find ttls faster than 200 Hz (unrealistically fast - probably a ttl which bounced to ground)
    # ensured outside of this script that this finds the true start ttl on every scan
    mask = np.insert(np.diff(tmp),0,0) # find first ttl in string that were too fast
    mask[mask<0] = 0
    print('num aberrant ttls',tmp.sum())

    frames = info['frame'][mask==0] # should be the original ttls up to a 1 VR frame error
    lines = info['line'][mask==0]

    ##
    ##

    # times of each ttl (VR frame)
    ttl_times = frames/fr + lines/lr
    numVRFrames = frames.shape[0]

    # create empty pandas dataframe to store calcium aligned data
    ca_df = pd.DataFrame(columns = vr_dframe.columns, index = np.arange(info['max_idx']))
    ca_time = np.arange(0,1/fr*info['max_idx'],1/fr) # time on this even grid
    if (ca_time.shape[0]-ca_df.shape[0])==1: # occaionally a 1 frame correction due to
                                            # scan stopping mid frame
        print('one frame correction')
        ca_time = ca_time[:-1]


    ca_df.loc[:,'time'] = ca_time
    mask = ca_time>=ttl_times[0] # mask for when ttls have started on imaging clock
                                # (i.e. imaging started and stabilized, ~10s)

    # take VR frames for which there are valid TTLs
    vr_dframe = vr_dframe.iloc[-numVRFrames:]
    # print(ttl_times.shape,vr_dframe.shape)

    # linear interpolation of position
    print(ttl_times[0],ttl_times[-1])
    print(ca_time[mask][0],ca_time[mask][-1])

    near_list = ['LEDCue']
    f_nearest = sp.interpolate.interp1d(ttl_times,vr_dframe[near_list]._values,axis=0,kind='nearest')
    ca_df.loc[mask,near_list] = f_nearest(ca_time[mask])
    ca_df.fillna(method='ffill',inplace=True)
    ca_df.loc[~mask,near_list]=-1.

    # integrate, interpolate and then take difference, to make sure data is not lost
    cumsum_list = ['dz','lick','reward','gng','manrewards']
    f_cumsum = sp.interpolate.interp1d(ttl_times,np.cumsum(vr_dframe[cumsum_list]._values,axis=0),axis=0,kind='slinear')
    ca_cumsum = np.round(np.insert(f_cumsum(ca_time[mask]),0,[0, 0,0 ,0,0],axis=0))
    if ca_cumsum[-1,-2]<ca_cumsum[-1,-3]:
        ca_cumsum[-1,-2]+=1


    ca_df.loc[mask,cumsum_list] = np.diff(ca_cumsum,axis=0)
    ca_df.loc[~mask,cumsum_list] = 0.



    # smooth instantaneous speed
    k = Gaussian1DKernel(5)
    cum_dz = convolve(np.cumsum(ca_df['dz']._values),k,boundary='extend')
    ca_df['dz'] = np.ediff1d(cum_dz,to_end=0)


    ca_df['speed'].interpolate(method='linear',inplace=True)
    ca_df['speed']=np.array(np.divide(ca_df['dz'],np.ediff1d(ca_df['time'],to_begin=1./fr)))
    ca_df['speed'].iloc[0]=0

    # calculate and smooth lick rate
    ca_df['lick rate'] = np.array(np.divide(ca_df['lick'],np.ediff1d(ca_df['time'],to_begin=1./fr)))
    ca_df['lick rate'] = convolve(ca_df['lick rate']._values,k,boundary='extend')

    # replace nans with 0s
    ca_df[['reward','lick']].fillna(value=0,inplace=True)
    return ca_df


def _VR_interp(frame):
    '''if 2P data doesn't exist interpolates behavioral data onto an even grid
    see _VR_align_to_2P for details'''
    fr = 30 # frame rate

    vr_time = frame['time']._values
    vr_time = vr_time - vr_time[0]
    ca_time = np.arange(0,vr_time[-1],1/fr)
    ca_df = pd.DataFrame(columns = frame.columns,index=np.arange(ca_time.shape[0]))

    ca_df['time'] = ca_time

    f_mean = sp.interpolate.interp1d(vr_time,frame['pos']._values,axis=0,kind='slinear')
    ca_df['pos'] = f_mean(ca_time)

    near_list = ['morph','clickOn','towerJitter','wallJitter','bckgndJitter']
    f_nearest = sp.interpolate.interp1d(vr_time,frame[near_list]._values,axis=0,kind='nearest')
    ca_df[near_list] = f_nearest(ca_time)

    cumsum_list = ['dz','lick','reward','tstart','teleport','rzone']

    f_cumsum = sp.interpolate.interp1d(vr_time,np.cumsum(frame[cumsum_list]._values,axis=0),axis=0,kind='slinear')
    ca_cumsum = np.round(np.insert(f_cumsum(ca_time),0,[0,0, 0 ,0,0,0],axis=0))
    if ca_cumsum[-1,-1]<ca_cumsum[-1,-2]:
        ca_cumsum[-1,-1]+=1


    ca_df[cumsum_list] = np.diff(ca_cumsum,axis=0)

    ca_df.fillna(method='ffill',inplace=True)
    k = Gaussian1DKernel(5)
    cum_dz = convolve(np.cumsum(ca_df['dz']._values),k,boundary='extend')
    ca_df['dz'] = np.ediff1d(cum_dz,to_end=0)


    ca_df['speed'].interpolate(method='linear',inplace=True)
    ca_df['speed']=np.array(np.divide(ca_df['dz'],np.ediff1d(ca_df['time'],to_begin=1./fr)))
    ca_df['speed'].iloc[0]=0


    ca_df['lick rate'] = np.array(np.divide(ca_df['lick'],np.ediff1d(ca_df['time'],to_begin=1./fr)))
    ca_df['lick rate'] = convolve(ca_df['lick rate']._values,k,boundary='extend')
    ca_df[['reward','tstart','teleport','lick','clickOn','towerJitter','wallJitter','bckgndJitter']].fillna(value=0,inplace=True)

    return ca_df

def _get_frame(f,fix_teleports=True):
    '''load a single session's sqlite database for behavior
    inputs: f - path to file
            fix_teleports - bool, whether or not to redo teleport values (Recommended)
    outputs: frame - pandas dataframe with raw VR data'''

    # connect to sqlite file
    sess_conn = sql.connect(f)

    # load all columns
    frame = pd.read_sql('''SELECT * FROM data''',sess_conn)


    frame['speed']=np.array(np.divide(frame['dz'],np.ediff1d(frame['time'],to_begin=.001)))
    frame['lick rate'] = np.array(np.divide(frame['lick'],np.ediff1d(frame['time'],to_begin=.001)))

    if fix_teleports:
        tstart_inds_vec,teleport_inds_vec = np.zeros([frame.shape[0],]), np.zeros([frame.shape[0],])
        pos = frame['pos']._values
        pos[pos<-50] = -50
        teleport_inds = np.where(np.ediff1d(pos,to_end=0)<=-50)[0]
        tstart_inds = np.append([0],teleport_inds[:-1]+1)

        for ind in range(tstart_inds.shape[0]):  # for teleports
            while (pos[tstart_inds[ind]]<0) : # while position is negative
                if tstart_inds[ind] < pos.shape[0]-1: # if you haven't exceeded the vector length
                    tstart_inds[ind]=tstart_inds[ind]+ 1 # go up one index
                else: # otherwise you should be the last teleport and delete this index
                    print("deleting last index from trial start")
                    tstart_inds=np.delete(tstart_inds,ind)
                    break

        tstart_inds_vec = np.zeros([frame.shape[0],])
        # print('fix teleports',frame.shape,tstart_inds.shape,teleport_inds.shape)
        tstart_inds_vec[tstart_inds] = 1

        teleport_inds_vec = np.zeros([frame.shape[0],])
        teleport_inds_vec[teleport_inds] = 1
        #print('fix teleports post sub',teleport_inds_vec.sum(),tstart_inds_vec.sum())
        frame['teleport']=teleport_inds_vec
        frame['tstart']=tstart_inds_vec
        #print('frame fill',frame.teleport.sum(),frame.tstart.sum())

    return frame