import pandas as pd
from scipy.sparse import csr_matrix
import numpy as np

date_end = 1914 + 28 * 1

file_pass = './'
sw_df = pd.read_pickle(file_pass + 'sw_df_{}_{}.pkl'.format(date_end - 28, date_end))
S = sw_df.s.values
W = sw_df.w.values
SW = sw_df.sw.values

roll_mat_df = pd.read_pickle(file_pass + 'roll_mat_df_{}_{}.pkl'.format(date_end - 28, date_end))
roll_index = roll_mat_df.index
roll_mat_csr = csr_matrix(roll_mat_df.values)


# 评分函数
# 得到聚合的结果
def rollup(v):
    '''
    '''
    return (v.T * roll_mat_csr.T).T


# 计算 WRMSSE 评估指标
def wrmsse(preds, y_true, score_only=False, s=S, w=W, sw=SW):
    '''
    preds - Predictions: pd.DataFrame of size (30490 rows, N day columns)
    y_true - True values: pd.DataFrame of size (30490 rows, N day columns)
    sequence_length - np.array of size (42840,)
    sales_weight - sales weights based on last 28 days: np.array (42840,)
    '''

    if score_only:
        return np.sum(
            np.sqrt(
                np.mean(
                    np.square(rollup(preds.values - y_true.values))
                    , axis=1)) * sw * 12)
    else:
        # score_matrix = (np.square(rollup(preds.values - y_true.values)) * np.square(w)[:, None]) * 12 / s[:, None]
        # score = np.sum(np.sqrt(np.mean(score_matrix, axis=1)))
        # score_matrix = (np.square(rollup(preds.values - y_true.values)) * np.square(w)[:, None]) / s[:, None]
        rmsse = np.sqrt(np.mean(np.square(rollup(preds.values - y_true.values)), axis=1))
        rmsse = rmsse / s
        score = rmsse * w
        # score = np.sum(np.sqrt(np.mean(score_matrix, axis=1)))
        # score = np.sum(np.sqrt(np.mean(score_matrix, axis=1)))
        return score, rmsse


numcols = [f"d_{day}" for day in range(date_end-28, date_end)]
pred_cols = ["F{}".format(day+1) for day in range(28)]
true_data = pd.read_csv('./input/sales_train_evaluation.csv', dtype="float32", usecols=numcols)
preds = pd.read_csv('submissionV3.csv', nrows=30490, usecols=pred_cols)
result = wrmsse(preds, true_data, False)

