import numpy as np
import pandas as pd
import lightgbm as lgb
 
IDIR = '../input/'
 

#Load Data

 
print('loading prior')
priors = pd.read_csv(IDIR + 'order_products__prior.csv', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8})
 
print('loading train')
train = pd.read_csv(IDIR + 'order_products__train.csv', dtype={
            'order_id': np.int32,
            'product_id': np.uint16,
            'add_to_cart_order': np.int16,
            'reordered': np.int8})
 
print('loading orders')
orders = pd.read_csv(IDIR + 'orders.csv', dtype={
        'order_id': np.int32,
        'user_id': np.int32,
        'eval_set': 'category',
        'order_number': np.int16,
        'order_dow': np.int8,
        'order_hour_of_day': np.int8,
        'days_since_prior_order': np.float32})
 
print('loading products')
products = pd.read_csv(IDIR + 'products.csv', dtype={
        'product_id': np.uint16,
        'order_id': np.int32,
        'aisle_id': np.uint8,
        'department_id': np.uint8},
        usecols=['product_id', 'aisle_id', 'department_id'])
 
print('priors {}: {}'.format(priors.shape, ', '.join(priors.columns)))
print('orders {}: {}'.format(orders.shape, ', '.join(orders.columns)))
print('train {}: {}'.format(train.shape, ', '.join(train.columns)))
 
 

#Product-level Features

 
print('computing product features')
prods = pd.DataFrame()
prods['orders'] = priors.groupby(priors.product_id).size().astype(np.int32)
prods['reorders'] = priors['reordered'].groupby(priors.product_id).sum().astype(np.float32)
prods['reorder_rate'] = (prods.reorders / prods.orders).astype(np.float32)
products = products.join(prods, on='product_id')
products.set_index('product_id', drop=False, inplace=True)
del prods
 
print('add order info to priors')
orders.set_index('order_id', inplace=True, drop=False)
priors = priors.join(orders, on='order_id', rsuffix='_')
priors.drop('order_id_', inplace=True, axis=1)



# User-level Features

print('computing user features')
usr = pd.DataFrame()
usr['average_days_between_orders'] = orders.groupby('user_id')['days_since_prior_order'].mean().astype(np.float32)
usr['nb_orders'] = orders.groupby('user_id').size().astype(np.int16)
 
users = pd.DataFrame()
users['total_items'] = priors.groupby('user_id').size().astype(np.int16)
users['all_products'] = priors.groupby('user_id')['product_id'].apply(set)
users['total_distinct_items'] = (users.all_products.map(len)).astype(np.int16)
 
users = users.join(usr)
del usr
users['average_basket'] = (users.total_items / users.nb_orders).astype(np.float32)
print('user features shape:', users.shape)




#User x Product Features (original)

print('computing user x product features...')
priors['user_product'] = priors.product_id + priors.user_id * 100000
 
d = dict()
for row in priors.itertuples():
    z = row.user_product
    if z not in d:
        d[z] = (1,
                (row.order_number, row.order_id),
                row.add_to_cart_order)
    else:
        d[z] = (d[z][0] + 1,
                max(d[z][1], (row.order_number, row.order_id)),
                d[z][2] + row.add_to_cart_order)
 
print('converting to dataframe...')
userXproduct = pd.DataFrame.from_dict(d, orient='index')
del d
userXproduct.columns = ['nb_orders', 'last_order_id', 'sum_pos_in_cart']
userXproduct.nb_orders = userXproduct.nb_orders.astype(np.int16)
userXproduct.last_order_id = userXproduct.last_order_id.map(lambda x: x[1]).astype(np.int32)
userXproduct.sum_pos_in_cart = userXproduct.sum_pos_in_cart.astype(np.int16)
print('user x product features count:', len(userXproduct))





#SECTION 5: Temporal Features (NEW)
# Builds per user-product interval stats to answer
# "what is the probability of reorder on a given day?"

 
def build_absolute_timeline(orders):
    """
    Converts relative days_since_prior_order into an absolute day
    counter per user, so we can compute real intervals between purchases.
    """
    print('building absolute day timeline...')
    orders_sorted = orders.sort_values(['user_id', 'order_number']).copy()
    orders_sorted['days_since_prior_order'] = orders_sorted['days_since_prior_order'].fillna(0)
    orders_sorted['abs_day'] = orders_sorted.groupby('user_id')['days_since_prior_order'].cumsum()
    return orders_sorted
 
 
def build_userproduct_temporal_features(priors, orders_sorted):
    """
    For each user-product pair computes:
    - avg/std/min/max days between purchases
    - last purchase absolute day
    - total purchase count
    These are the core features for day-level reorder probability.
    """
    print('building temporal user-product features...')
 
    # merge absolute day into priors
    priors_with_day = priors.merge(
        orders_sorted[['order_id', 'abs_day']],
        on='order_id',
        how='left'
    )
 
    # sort to get purchase sequence per user-product
    priors_with_day = priors_with_day.sort_values(['user_id', 'product_id', 'order_number'])
 
    # interval between consecutive purchases of same product by same user
    priors_with_day['prev_abs_day'] = priors_with_day.groupby(
        ['user_id', 'product_id']
    )['abs_day'].shift(1)
 
    priors_with_day['purchase_interval'] = (
        priors_with_day['abs_day'] - priors_with_day['prev_abs_day']
    )
 
    # aggregate per user-product
    up_temporal = priors_with_day.groupby(['user_id', 'product_id']).agg(
        UP_avg_interval   = ('purchase_interval', 'mean'),
        UP_std_interval   = ('purchase_interval', 'std'),
        UP_min_interval   = ('purchase_interval', 'min'),
        UP_max_interval   = ('purchase_interval', 'max'),
        UP_last_abs_day   = ('abs_day', 'max'),
        UP_purchase_count = ('product_id', 'count'),
    ).reset_index()
 
    up_temporal['UP_std_interval'] = up_temporal['UP_std_interval'].fillna(0)
 
    return up_temporal
 
 
def build_user_timeline(orders_sorted):
    """
    For each user, get their last known absolute day and overall stats.
    This acts as the reference point when predicting for a target date.
    """
    user_timeline = orders_sorted.groupby('user_id').agg(
        user_last_abs_day            = ('abs_day', 'max'),
        user_total_orders            = ('order_number', 'max'),
        user_avg_days_between_orders = ('days_since_prior_order', 'mean'),
    ).reset_index()
    return user_timeline
 
 
def build_prediction_features(user_ids, target_day_offset,
                               up_temporal, user_timeline,
                               users, products):
    """
    Builds the feature dataframe for prediction.
 
    Parameters
    ----------
    user_ids         : list of user IDs to predict for
    target_day_offset: int — days from user's last known order
                       e.g. 0 = today, 7 = one week from now
    """
    # filter to relevant users
    up = up_temporal[up_temporal['user_id'].isin(user_ids)].copy()
    ut = user_timeline[user_timeline['user_id'].isin(user_ids)].copy()
 
    df = up.merge(ut, on='user_id', how='left')
 
    # target absolute day
    df['target_abs_day'] = df['user_last_abs_day'] + target_day_offset
 
    # ── core temporal signals ──────────────────────────────────────────────
 
    # raw days since this product was last bought
    df['UP_days_since_last'] = df['target_abs_day'] - df['UP_last_abs_day']
 
    # how many days past the expected reorder point are we?
    # positive = overdue, negative = too early
    df['UP_days_overdue'] = df['UP_days_since_last'] - df['UP_avg_interval']
 
    # normalized overdue relative to user's consistency for this product
    df['UP_overdue_zscore'] = df['UP_days_overdue'] / (df['UP_std_interval'] + 1e-5)
 
    # how far through the expected cycle are we?
    # 0.5 = halfway, 1.0 = right on time, 2.0 = twice overdue
    df['UP_interval_progress'] = df['UP_days_since_last'] / (df['UP_avg_interval'] + 1e-5)
 
    # ── product-level features ─────────────────────────────────────────────
    df = df.merge(
        products[['product_id', 'aisle_id', 'department_id',
                   'orders', 'reorders', 'reorder_rate']],
        on='product_id', how='left'
    )
 
    # ── user-level features ────────────────────────────────────────────────
    users_reset = users.reset_index()
    df = df.merge(
        users_reset[['user_id', 'total_items', 'total_distinct_items',
                     'average_days_between_orders', 'average_basket', 'nb_orders']],
        on='user_id', how='left'
    )
 
    # target day context — lets model learn day-specific patterns
    df['target_day_offset'] = target_day_offset
 
    return df
 
 
def build_training_data(train_orders, train,
                         up_temporal, user_timeline,
                         users, products, orders_sorted):
    """
    Builds training data by treating each train order as the 'target day'.
    Label = 1 if product was reordered in that order, else 0.
    """
    print('building training data...')
 
    # attach absolute day to each train order
    train_orders = train_orders.merge(
        orders_sorted[['order_id', 'abs_day']],
        on='order_id', how='left'
    )
 
    all_dfs = []
 
    for i, row in enumerate(train_orders.itertuples()):
        if i % 5000 == 0:
            print(f'  processing train order {i}...')
 
        user_id      = row.user_id
        order_id     = row.order_id
        target_abs_day = row.abs_day
 
        # user's last abs day strictly before this order
        user_history = orders_sorted[
            (orders_sorted['user_id'] == user_id) &
            (orders_sorted['abs_day'] < target_abs_day)
        ]
        if user_history.empty:
            continue
 
        user_last_day = user_history['abs_day'].max()
        day_offset    = int(target_abs_day - user_last_day)
 
        # build features as if predicting for this day
        ut_user = user_timeline[user_timeline['user_id'] == user_id].copy()
        # override last abs day to simulate "just before this order"
        ut_user['user_last_abs_day'] = user_last_day
 
        df = build_prediction_features(
            user_ids=[user_id],
            target_day_offset=day_offset,
            up_temporal=up_temporal,
            user_timeline=ut_user,
            users=users,
            products=products
        )
 
        # label: did this product appear in the train order?
        ordered_products = set(
            train[train['order_id'] == order_id]['product_id'].tolist()
        )
        df['label'] = df['product_id'].isin(ordered_products).astype(np.int8)
        all_dfs.append(df)
 
    return pd.concat(all_dfs, ignore_index=True)
 