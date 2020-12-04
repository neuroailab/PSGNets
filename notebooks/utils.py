import numpy as np
import pickle as cPickle
import gridfs
import scipy.signal as signal
import pymongo as pm
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import host_subplot
import mpl_toolkits.axisartist as AA
import matplotlib.cm as cm
import sys
import tensorflow as tf
import copy
import os

def return_metrics(conn, dbname='test_phys', collname='po_net', exp_id='run2', valid_key='valid0', loss_keys='particle_only_loss'):
    if '.files' not in collname:
        coll = conn[dbname][collname+'.files']
    else:
        coll = conn[dbname][collname]

    if not isinstance(loss_keys, list):
        loss_keys = [loss_keys]

    if loss_keys == ['val_loss']:
        val_loss = [r['validation_results'][valid_key]['val_loss'] for r in coll.find({'exp_id': exp_id,
                                                         'validation_results': {'$exists': True}}).sort('step')]
        return {'val_loss': val_loss}
    elif len(loss_keys) == 1:
        val_loss = [r['validation_results'][valid_key]['loss'] for r in coll.find({'exp_id': exp_id,
                                                         'validation_results': {'$exists': True}}).sort('step')]

        particle_only_loss = [r['validation_results'][valid_key][loss_keys[0]] for r in coll.find({'exp_id': exp_id,
                                                         'validation_results': {'$exists': True}}).sort('step')]


        return val_loss, particle_only_loss
    else:
        loss_dict = {}
        # if 'loss' not in loss_keys:
        #     loss_dict['loss'] = [r['validation_results'][valid_key]['loss'] for r in coll.find({'exp_id': exp_id,
        #                                                  'validation_results': {'$exists': True}}).sort('step')]
        for k in loss_keys:
            loss_dict[k] = [r['validation_results'][valid_key][k] for r in coll.find({'exp_id': exp_id,
                                                         'validation_results': {'$exists': True}}).sort('step')]

        return loss_dict

def load_from_gridfs(conn, dbname='test_phys', collname='po_net', exp_id='run2', expid_prefix='eval_part', load_step=190000):
#     print('Loading results for file ' + str(file))
    if load_step is not None:
        suffix = '_'+str(load_step)
    else:
        suffix = ''
    expidstr = expid_prefix+exp_id+suffix
    print(expidstr)
    r = conn[dbname][collname+'.files'].find_one({'exp_id': expidstr})
    _id = r['_id']
    fn = str(_id) + '_fileitems'
    fsys = gridfs.GridFS(conn[dbname], collname)
    fh = fsys.get_last_version(fn)
    fstr = fh.read()
    fh.close()
    obj = cPickle.loads(fstr, encoding='latin1') # python3
#     obj = pickle.loads(fstr)
    targets = obj['validation_results'][expid_prefix]
    # targets = obj['validation_results']
    return targets

def plot_smooth_trainloss(conn, dbname, collname, exp_id, N=1, ylim=None, xlim=None, nanfilter=True, loss_keys=['loss', 'learning_rate'], loss_scales={}):
    coll = conn[dbname][collname+'.files']
    train_loss = np.concatenate([[[_r[lkey] for lkey in loss_keys] for _r in r['train_results']]
                            for r in coll.find(
                                       {'exp_id': exp_id, 'train_results': {'$exists': True}},
                                        projection=['train_results'])])

    for l, lk in enumerate(loss_keys):
        if lk == 'learning_rate':
            continue
        this_loss = train_loss[:,l:l+1]
        if nanfilter:
            this_loss = this_loss[~np.isnan(this_loss[:,0]),:]
        smooth_this_loss = np.convolve(this_loss[:,0], (1./N)*np.ones(N), 'valid')
        mult = loss_scales.get(lk, 1.0)
        plt.plot(mult * smooth_this_loss, label=lk)
    if ylim is not None:
        plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)
    plt.legend(loc='upper right')
    plt.xlabel('training steps', fontsize=16)
    plt.ylabel('training loss', fontsize=16)
    plt.title('training losses for ' + exp_id)
    plt.show()

    return train_loss

def plot_val_losses(conn, dbname, collname, exp_id, valid_key='object_metrics', save_valid_freq=5000, plot_ticks_freq=5000, get_losses=False, start=0, end=None, validate_first=False,
                    loss_keys=['mIoU_matched_t0'], maxes=False,
                    transform_y=None, ylabel_prefix="", colors=['tab:blue', 'tab:red', 'tab:green', 'tab:purple', 'tab:orange', 'tab:cyan', 'magenta', 'gold', 'gray']):

    loss_dict = return_metrics(conn=conn, dbname=dbname, collname=collname, exp_id=exp_id,
                               valid_key=valid_key, loss_keys=loss_keys)

    # return loss_dict
    num_iters = save_valid_freq * len(loss_dict[loss_keys[0]])
    x = np.arange(0, num_iters, save_valid_freq)
    if not validate_first:
        x = [_x + x[1] for _x in x]
    if end is None:
        end = len(x)
    val_loss = np.array(loss_dict[loss_keys[0]])
    val_loss[np.isnan(val_loss)] = np.max(val_loss[~np.isnan(val_loss)])

    host = host_subplot(111, axes_class=AA.Axes)
    plt.subplots_adjust(right=0.75)

    transform_y = transform_y or (lambda x: x)
    host.plot(x[start:end], transform_y(val_loss)[start:end], color=colors[0])
    host.set_xlabel("val step (10^3)")
    ptf = plot_ticks_freq // save_valid_freq
    host.set_xticks(x[start:end:ptf])
    host.set_xticklabels(["%.0f" % (xi/1000) for xi in x[start:end:ptf]])
    host.set_ylabel((ylabel_prefix+"  %s") % (loss_keys[0]))
    host.axis["left"].label.set_color(colors[0])

    pars = []
    if maxes:
        loss_mins = {loss_keys[0]: ((start+np.argmax(val_loss[start:end])+(1-int(validate_first)))*save_valid_freq, np.max(val_loss[start:end]))}
    else:
        loss_mins = {loss_keys[0]: ((start+np.argmin(val_loss[start:end])+(1-int(validate_first)))*save_valid_freq, np.min(val_loss[start:end]))}
    val_losses = {loss_keys[0]: val_loss}
    n_axes = len(loss_keys) - 1
    offset=60
    if n_axes:
        for i in range(n_axes):
            par = host.twinx()
            if i>0:
                new_axis = par.get_grid_helper().new_fixed_axis
                par.axis["right"] = new_axis(loc="right", axes=par, offset=((i)*offset, 0))
            par.axis["right"].toggle(all=True)
            val_loss = loss_dict[loss_keys[i+1]]
            val_losses[loss_keys[i+1]] = val_loss
            par.plot(x[start:end], transform_y(val_loss)[start:end], color=colors[i+1])
            par.set_ylabel("%s %s" % (ylabel_prefix, loss_keys[i+1]))
            par.axis["right"].label.set_color(colors[i+1])

            if maxes:
                loss_mins[loss_keys[i+1]] = ((start+np.argmax(val_loss[start:end])+(1-int(validate_first)))*save_valid_freq, np.max(val_loss[start:end]))
            else:
                loss_mins[loss_keys[i+1]] = ((start+np.argmin(val_loss[start:end])+(1-int(validate_first)))*save_valid_freq, np.min(val_loss[start:end]))

    plt.draw()
    plt.title('val losses for ' + exp_id)
    plt.show()

    if get_losses:
        return val_losses
    else:
        return loss_mins

def get_val_results_from_db(conn, dbname, collname, load_exp_id, suffix, group='val', step=None, metrics=['object_metrics'], python=3):
    exp_id = load_exp_id + '_' + group + '_' + suffix + '_' + ('last' if step is None else str(step))
    raw_data = multi_load_from_gridfs(conn, dbname, collname, exp_id, expid_prefix=None, load_step=None, datasets=metrics, python=python)
    return raw_data

def get_results_from_db(conn, dbname, collname, exp_id, expid_prefix, load_step, datasets, group='val', return_raw=False, python=3):
    exp_id += group
    raw_data = multi_load_from_gridfs(conn, dbname, collname, exp_id, expid_prefix, load_step, datasets=datasets, python=python)
    if return_raw:
        return raw_data
    else:
        combine_data = combine_val_files(raw_data, expid_prefix, datasets)
        return combine_data

def multi_load_from_gridfs(conn, dbname='test_phys', collname='po_net', exp_id='run2', expid_prefix='eval_part',
                           load_step=190000, datasets=[], python=3):
#     print('Loading results for file ' + str(file))
    # expidstr = expid_prefix+exp_id+'_'+str(load_step)
    expidstr = exp_id
    if expid_prefix is not None:
        expidstr = expid_prefix + expidstr
    if load_step is not None:
        expidstr += '_' + str(load_step)
    print(expidstr)
    r = conn[dbname][collname+'.files'].find_one({'exp_id': expidstr})
    _id = r['_id']
    fn = str(_id) + '_fileitems'
    fsys = gridfs.GridFS(conn[dbname], collname)
    fh = fsys.get_last_version(fn)
    fstr = fh.read()
    fh.close()

    if python==3:
        obj = cPickle.loads(fstr, encoding='latin1')
    elif python==2:
        obj = cPickle.loads(fstr)
    targets = {}
    for d in datasets:
        try:
            targets[expid_prefix+'_'+d] = obj['validation_results'][expid_prefix+'_'+d]
        except KeyError:
            print(obj['validation_results'].keys())
            targets[expid_prefix+'_'+d] = obj['validation_results'][d]
        except TypeError:
            targets[d] = obj['validation_results'][d]
    return targets
