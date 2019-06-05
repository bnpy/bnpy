import pandas as pd
import numpy as np
import scipy.io
import bnpy

def read_list_of_str_from_mat_struct(struct_var):
    return np.asarray([str(np.squeeze(s)) for s in np.squeeze(struct_var)], dtype='str')

if __name__ == '__main__':
    Q = scipy.io.loadmat('/Users/mhughes/git/mocap6dataset/mocap6.mat')
    
    file_names = read_list_of_str_from_mat_struct([Q['DataBySeq'][ii][0][3][0,0] for ii in range(6)])
    print(file_names)

    channel_names = read_list_of_str_from_mat_struct(Q['ChannelNames'])
    print(channel_names)

    action_names = read_list_of_str_from_mat_struct(Q['ActionNames'])
    print(action_names)

    Q = Q['DataBySeq']
    X_list = list()
    X_prev_list = list()
    T_list = list()
    Z_list = list()
    for doc in xrange(6):
        doc_X = np.asarray(Q[doc]['X'][0], dtype=np.float64).copy()
        doc_Xprev = np.asarray(Q[doc]['Xprev'][0], dtype=np.float64).copy()
        doc_Z = np.squeeze(
            np.asarray(Q[doc]['TrueZ'][0], dtype=np.float64).copy())
        X_list.append(doc_X)
        X_prev_list.append(doc_Xprev)
        T_list.append(doc_X.shape[0])
        Z_list.append(doc_Z)

    X = np.vstack(X_list)
    Xprev = np.vstack(X_prev_list)
    doc_range = np.hstack([0, np.cumsum(T_list)])
    TrueZ = np.hstack(Z_list).astype(np.int32)

    seq_id_arr = []
    tstep_id_arr = []
    for seq_id in range(6):
        seq_id_arr = np.hstack([seq_id_arr, [seq_id for _ in range(T_list[seq_id])]])
        tstep_id_arr = np.hstack([tstep_id_arr, [t for t in range(T_list[seq_id])]])
    seq_id_arr = seq_id_arr.astype(np.int32)
    tstep_id_arr = tstep_id_arr.astype(np.int32)

    x_df = pd.DataFrame(X, columns=channel_names)
    x_df['seq_id'] = seq_id_arr
    x_df['tstep_id'] = tstep_id_arr
    new_col_order= ['seq_id', 'tstep_id'] + x_df.columns[:12].values.tolist()
    x_df = x_df[new_col_order]

    xprev_df = pd.DataFrame(Xprev, columns=['prev_' + c for c in channel_names])
    xprev_df['seq_id'] = seq_id_arr
    xprev_df['tstep_id'] = tstep_id_arr
    new_col_order = ['seq_id', 'tstep_id'] + xprev_df.columns[:12].values.tolist()
    xprev_df = xprev_df[new_col_order]

    z_df = pd.DataFrame()
    z_df['action_name'] = [action_names[TrueZ[t] - 1] for t in range(TrueZ.size)]
    z_df['seq_id'] = seq_id_arr
    z_df['tstep_id'] = tstep_id_arr
    z_df = z_df[['seq_id','tstep_id', 'action_name']]

    z_df.to_csv('actions_per_tstep.csv', index=False)
    x_df.to_csv('sensor_data_per_tstep.csv', index=False)
    xprev_df.to_csv('prev_sensor_data_per_tstep.csv', index=False)

    subj_uid = np.asarray([f.split('_')[0] for f in file_names], dtype=np.int32)
    trial_id = np.asarray([f.split('_')[1] for f in file_names], dtype=np.int32)
    seq_df = pd.DataFrame(file_names, columns=['file_name'])
    seq_df['seq_id'] = np.arange(6, dtype=np.int32)
    seq_df['trial_id'] = trial_id
    seq_df['subj_uid'] = subj_uid
    seq_df['url'] = ["http://mocap.cs.cmu.edu/search.php?subjectnumber=%d" % uid for uid in subj_uid]
    seq_df.to_csv('metadata_per_seq.csv', index=False, columns=['seq_id', 'subj_uid', 'trial_id', 'file_name', 'url']) 


    np.savez('dataset.npz',
        **dict(X=X, Xprev=Xprev, doc_range=doc_range, TrueZ=TrueZ, true_state_names=action_names, column_names=channel_names))
