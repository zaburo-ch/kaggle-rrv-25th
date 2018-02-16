import numpy as np
import pandas as pd

import base
import os
import argparse
import json



parser = argparse.ArgumentParser()
parser.add_argument('--target_dir', type=str, default='data/output/seq_run/')
parser.add_argument('--without_valid', action='store_true')
parser.add_argument('--description', type=str, default='wavenet ensemble')
args = parser.parse_args()

args.target_results = []
for name in os.listdir(args.target_dir):
    with open(args.target_dir + name + '/params.json') as fp:
        params = json.load(fp)
    if params['without_valid'] == args.without_valid:
        args.target_results.append(args.target_dir + name + '/')

om = base.OutputManager(vars(args))
args.save_path = om.get_path()


def base_averaging(suf=''):
    pred = base.load_df(args.target_results[0] + 'pred' + suf + '.h5').values
    for name in args.target_results[1:]:
        pred += base.load_df(name + 'pred' + suf + '.h5').values
    pred /= len(args.target_results)
    pred[pred < np.log1p(1)] = np.log1p(1)
    return pred


def save_as_csv(pred, filepath):
    air_visit_data = pd.read_csv(base.INPUT_DIR + 'air_visit_data.csv', parse_dates=[1])
    air_visit_data = air_visit_data.pivot(index='air_store_id', columns='visit_date', values='visitors')
    id_test = pd.read_csv(base.INPUT_DIR + 'sample_submission.csv')['id']
    pred = np.expm1(pred)
    pred = pd.DataFrame(pred, index=air_visit_data.index, columns=pd.date_range('2017-04-23', periods=39)).reset_index()
    pred = pred.melt(id_vars='air_store_id', var_name='date', value_name='visitors')
    pred['id'] = pred['air_store_id'] + pred['date'].map(lambda x: x.strftime('_%Y-%m-%d'))
    pred.loc[pred['visitors'] < 1, 'visitors'] = 1
    pred.set_index('id').loc[id_test, 'visitors'].reset_index().to_csv(filepath, index=False)


if args.without_valid:
    base_ave_pred = base_averaging()
    save_as_csv(base_ave_pred, args.save_path + 'base_ave_pred.csv')
else:

    def rmse(y_true, y_pred, mask):
        se = (y_true - y_pred) ** 2
        mse = np.sum(se * mask) / np.sum(mask)
        return np.sqrt(mse)

    def print_score(y_true, y_pred, mask, name):
        om.print('validation score ({})'.format(name))
        om.print('total score:   {:.8f}'.format(rmse(y_true, y_pred, mask)))
        om.print('public score:  {:.8f}'.format(rmse(y_true[:, :6], y_pred[:, :6], mask[:, :6])))
        om.print('private score: {:.8f}'.format(rmse(y_true[:, 6:], y_pred[:, 6:], mask[:, 6:])))

    log_x_1yago = base.load_npy(base.WORKING_DIR + 'log_x_1yago.npy')
    mask_1yago = base.load_npy(base.WORKING_DIR + 'mask_1yago.npy')
    log_x_latest = base.load_npy(base.WORKING_DIR + 'log_x_latest.npy')
    mask_latest = base.load_npy(base.WORKING_DIR + 'mask_latest.npy')

    print_score(log_x_1yago, base.load_df(args.target_results[0] + 'pred_valid_1yago.h5').values, mask_1yago, 'one of base 1yago')
    print_score(log_x_latest, base.load_df(args.target_results[0] + 'pred_valid_latest.h5').values, mask_latest, 'one of base latest')

    base_ave_pred = base_averaging()
    base_ave_1yago = base_averaging('_valid_1yago')
    base_ave_latest = base_averaging('_valid_latest')
    print_score(log_x_1yago, base_ave_1yago, mask_1yago, 'base_ave 1yago')
    print_score(log_x_latest, base_ave_latest, mask_latest, 'base_ave latest')

    save_as_csv(base_ave_pred, args.save_path + 'base_ave_pred.csv')
