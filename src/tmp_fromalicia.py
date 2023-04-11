def get_dir(onset_dir, mask_dir, fmri_dir, films, nb_sub):
    “”"
    plus tard
    “”"
    sub_ID=[‘%0*d’ % (2, i+1)for i in np.arange(nb_sub)]
    #sub_ID.remove(‘12’)
    #sub_ID.remove(‘18’)
    delay = 4
    mask = compute_brain_mask(nib.load(os.path.join(mask_dir, ‘gray_matter.nii.gz’)), threshold=.3)
    mean_vox=[]
    durations = []
    for f in films :
        vox_film=[]
        for ID in sub_ID:
            try :
                o_f = os.path.join(onset_dir,f”sub-S{ID}/**/*{f}_events.tsv*“)
                o_f=glob.glob(o_f, recursive=True)
                p_s = os.path.join(fmri_dir,f”sub-S{ID}/**/*{f}.feat*“)
                p_s=glob.glob(p_s, recursive=True)
            except:
                    print(“Something went wrong with file reading “)
            o_f=pd.read_csv(o_f[0], sep=‘\t’)
            onset=int(np.round(o_f[o_f[‘trial_type’]==“film”][‘onset’])+delay)
            duration = int(np.round(o_f[o_f[‘trial_type’]==“film”][‘duration’]/1.3))
            for file in sorted(os.listdir(p_s[0])):
                if file.endswith(‘MNI.nii’):
                    map_ = nib.load(os.path.join(p_s[0], file))
                    x=get_samples(clean_img(map_, standardize=False, ensure_finite=True), mask)
                    #Onset Removed
                    x = x[onset:onset+duration-1]
                    #Scrubbing
                    vox_film.append(scrubbing(p_s[0], onset, duration, x, True, 0.5))
        #Average among subject
        mean_vox.append(np.nanmean(vox_film, axis=0))
        durations.append(duration)
    mean_vox=np.vstack(mean_vox)
    X=pd.DataFrame(np.array(mean_vox).reshape(-1, np.array(mean_vox).shape[-1]))
    return X, durations





