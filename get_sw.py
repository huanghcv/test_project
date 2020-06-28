import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.sparse import csr_matrix
import gc


# 转换数据类型，减少内存占用空间
def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (
                start_mem - end_mem) / start_mem))
    return df


# 加载数据
# data_pass = './m5-forecasting-accuracy/'
data_pass = './input/'

# sale数据
sales = pd.read_csv(data_pass + 'sales_train_evaluation.csv')

# 日期数据
calendar = pd.read_csv(data_pass + 'calendar.csv')
calendar = reduce_mem_usage(calendar)

# 价格数据
sell_prices = pd.read_csv(data_pass + 'sell_prices.csv')
sell_prices = reduce_mem_usage(sell_prices)

# 计算价格
# 按照定义，只需要计算最近的 28 天售卖量（售卖数*价格），通过这个可以得到 weight
# 可以不是 1914
date_end = 1914+28*1
cols = ["d_{}".format(i) for i in range(date_end - 28, date_end)]
data = sales[["id", 'store_id', 'item_id'] + cols]

# 从横表改为纵表
data = data.melt(id_vars=["id", 'store_id', 'item_id'],
                 var_name="d", value_name="sale")

# 和日期数据做关联
data = pd.merge(data, calendar, how='left',
                left_on=['d'], right_on=['d'])

data = data[["id", 'store_id', 'item_id', "sale", "d", "wm_yr_wk"]]

# 和价格数据关联
data = data.merge(sell_prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
data.drop(columns=['wm_yr_wk'], inplace=True)

# 计算售卖量
data['sale_usd'] = data['sale'] * data['sell_price']

# 得到聚合矩阵
# 30490 -> 42840
# 需要聚合的维度明细计算出来
dummies_list = [sales.state_id, sales.store_id,
                sales.cat_id, sales.dept_id,
                sales.state_id + sales.cat_id, sales.state_id + sales.dept_id,
                sales.store_id + sales.cat_id, sales.store_id + sales.dept_id,
                sales.item_id, sales.state_id + sales.item_id, sales.id]

# 全部聚合为一个， 最高 level
dummies_df_list = [pd.DataFrame(np.ones(sales.shape[0]).astype(np.int8),
                                index=sales.index, columns=['all']).T]

# 挨个计算其他 level 等级聚合
for i, cats in enumerate(dummies_list):
    dummies_df_list += [pd.get_dummies(cats, drop_first=False, dtype=np.int8).T]

# 得到聚合矩阵
roll_mat_df = pd.concat(dummies_df_list, keys=list(range(12)),
                        names=['level', 'id'])  # .astype(np.int8, copy=False)

# 保存聚合矩阵
roll_index = roll_mat_df.index
roll_mat_csr = csr_matrix(roll_mat_df.values)
roll_mat_df.to_pickle('roll_mat_df_{}_{}.pkl'.format(date_end-28, date_end))

# 释放内存
del dummies_df_list, roll_mat_df
gc.collect()


# 按照定义，计算每条时间序列 RMSSE 的权重:
def get_s(drop_days=0):
    """
    drop_days: int, equals 0 by default, so S is calculated on all data.
               If equals 28, last 28 days won't be used in calculating S.
    """

    # 要计算的时间序列长度
    d_name = ['d_' + str(i + 1) for i in range(date_end-1 - drop_days)]
    # 得到聚合结果
    sales_train_val = roll_mat_csr * sales[d_name].values

    # 按照定义，前面连续为 0 的不参与计算
    start_no = np.argmax(sales_train_val > 0, axis=1)

    # 这些连续为 0 的设置为 nan
    flag = np.dot(np.diag(1 / (start_no + 1)), np.tile(np.arange(1, date_end - drop_days), (roll_mat_csr.shape[0], 1))) < 1
    sales_train_val = np.where(flag, np.nan, sales_train_val)

    # 根据公式计算每条时间序列 rmsse的权重
    weight1 = np.nansum(np.diff(sales_train_val, axis=1) ** 2, axis=1) / (date_end-1 - start_no - 1)

    return weight1


S = get_s(drop_days=0)


# 根据定义计算 WRMSSE 的权重，这里指 w
def get_w(sale_usd):
    """
    """
    # 得到最细维度的每条时间序列的权重
    total_sales_usd = sale_usd.groupby(
        ['id'], sort=False)['sale_usd'].apply(np.sum).values

    # 通过聚合矩阵得到不同聚合下的权重
    weight2 = roll_mat_csr * total_sales_usd

    # return 12 * weight2 / np.sum(weight2)
    return 1 / 12 * weight2 / np.sum(total_sales_usd)


W = get_w(data[['id', 'sale_usd']])

SW = W / np.sqrt(S)

sw_df = pd.DataFrame(np.stack((S, W, SW), axis=-1), index=roll_index, columns=['s', 'w', 'sw'])
sw_df.to_pickle('sw_df_{}_{}.pkl'.format(date_end-28, date_end))





# 加载前面预先计算好的各个权重
file_pass = './'
sw_df = pd.read_pickle(file_pass + 'sw_df_{}_{}.pkl'.format(date_end-28, date_end))
S = sw_df.s.values
W = sw_df.w.values
SW = sw_df.sw.values

roll_mat_df = pd.read_pickle(file_pass + 'roll_mat_df_{}_{}.pkl'.format(date_end-28, date_end))
roll_index = roll_mat_df.index
roll_mat_csr = csr_matrix(roll_mat_df.values)

print(sw_df.loc[(11,slice(None))].sw)
