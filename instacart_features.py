import numpy as np
import pandas as pd
import lightgbm as lgb

# ── NLP imports (lightweight only — no transformers or large embeddings) ──────
import re
from sklearn.feature_extraction.text import TfidfVectorizer
# ─────────────────────────────────────────────────────────────────────────────

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

# ── NLP INSERT 1: Load extra text sources needed for NLP features ─────────────
# We load product_name and aisle separately so we never touch the products
# DataFrame the rest of the pipeline already depends on (same usecols, same
# dtypes above are left entirely unchanged).

print('loading product names for NLP features')
# Re-read products.csv to get product_name only — kept in a separate frame
# so the existing products load contract (usecols) is not broken.
products_text = pd.read_csv(
    IDIR + 'products.csv',
    usecols=['product_id', 'product_name'],
    dtype={'product_id': np.uint16, 'product_name': str}
)

print('loading aisles for NLP features')
aisles = pd.read_csv(
    IDIR + 'aisles.csv',
    dtype={'aisle_id': np.uint8, 'aisle': str}
)
# ─────────────────────────────────────────────────────────────────────────────

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

# ── NLP INSERT 2: Compute NLP features and merge into products ────────────────
#
# All three features are computed once over the product/aisle catalogue and
# joined by product_id.  Zero per-row inference cost at prediction time.
#
print('computing NLP features')

# ── NLP Feature 1: product_name_word_count ────────────────────────────────────
# Hypothesis: Longer, more descriptive product names (e.g. "Organic Free-Range
# Large Brown Eggs 12 ct") signal specialty or niche items.  Buyers of such
# items tend to be brand-loyal and repeat-purchase at higher rates than buyers
# of generic, short-named commodities ("Eggs").  Word count is a free, O(N)
# proxy for product specificity with no vocabulary assumption.
products_text['product_name_clean'] = (
    products_text['product_name']
    .fillna('')
    .str.lower()
    .str.strip()
)
products_text['product_name_word_count'] = (
    products_text['product_name_clean']
    .apply(lambda x: len(x.split()))          # simple whitespace tokenisation
    .astype(np.int16)
)

# ── NLP Feature 2: is_health_organic ─────────────────────────────────────────
# Hypothesis: Health-conscious and organic-oriented shoppers are among the most
# loyal repeat buyers on Instacart (widely observed in public competition
# kernels). A binary flag for health/organic keywords captures this buyer
# segment directly without any embedding overhead.  The keyword list covers the
# most predictive tokens while staying fast and interpretable.
HEALTH_KEYWORDS = re.compile(
    r'\b(organic|natural|gluten.free|vegan|non.gmo|free.range|'
    r'whole.grain|raw|plant.based|dairy.free|sugar.free|'
    r'hormone.free|antibiotic.free|sprouted|probiotic)\b',
    re.IGNORECASE
)
products_text['is_health_organic'] = (
    products_text['product_name_clean']
    .str.contains(HEALTH_KEYWORDS)
    .astype(np.int8)
)

# ── NLP Feature 3: aisle_tfidf_score ─────────────────────────────────────────
# Hypothesis: Aisles whose names contain rare, distinctive tokens
# (e.g. "kombucha", "refrigerated", "specialty") describe niche categories
# with high repeat-buyer affinity; generic aisles ("beverages", "snacks") have
# lower category loyalty.  We fit a TF-IDF on the 134 Instacart aisle names
# and take each aisle's max TF-IDF weight as its "vocabulary distinctiveness"
# score — a single scalar per aisle, no per-row overhead.
tfidf = TfidfVectorizer(
    analyzer='word',
    ngram_range=(1, 2),   # uni+bigrams capture compound aisle names
    sublinear_tf=True,    # dampen frequency explosions
    min_df=1
)
aisle_tfidf_matrix = tfidf.fit_transform(aisles['aisle'].fillna(''))
# max TF-IDF weight across all tokens → one scalar per aisle
aisles['aisle_tfidf_score'] = (
    np.asarray(aisle_tfidf_matrix.max(axis=1)).flatten().astype(np.float32)
)

# Join aisle_tfidf_score onto products_text via aisle_id
# We use the main products frame (which already has aisle_id) as the bridge.
products_text = products_text.merge(
    products[['product_id', 'aisle_id']].reset_index(drop=True),
    on='product_id',
    how='left'
)
products_text = products_text.merge(
    aisles[['aisle_id', 'aisle_tfidf_score']],
    on='aisle_id',
    how='left'
)

# Combine the three NLP features into one lightweight frame keyed on product_id
nlp_features = products_text[[
    'product_id',
    'product_name_word_count',   # Feature 1
    'is_health_organic',         # Feature 2
    'aisle_tfidf_score',         # Feature 3
]].set_index('product_id')

# Join NLP features into the main products DataFrame (products is already
# indexed by product_id, so a direct join works without disrupting any
# downstream code that reads products by index).
products = products.join(nlp_features, on='product_id')

del products_text, aisles, nlp_features, tfidf, aisle_tfidf_matrix
print('NLP features added to products:', ['product_name_word_count',
                                           'is_health_organic',
                                           'aisle_tfidf_score'])
# ─────────────────────────────────────────────────────────────────────────────

print('add order info to priors')
# orders.set_index('order_id', inplace=True, drop=False)
orders.set_index('order_id', inplace=True, drop=True)
orders['order_id'] = orders.index
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
    # orders_sorted = orders.sort_values(['user_id', 'order_number']).copy()
    orders_sorted = orders.reset_index(drop=True).sort_values(['user_id', 'order_number']).copy()
    orders_sorted['order_id'] = orders_sorted.index
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
    # priors_with_day = priors.merge(
    #     orders_sorted[['order_id', 'abs_day']],
    #     on='order_id',
    #     how='left'
    # )

    priors_with_day = priors.merge(
    orders_sorted[['abs_day']],
    left_on='order_id',
    right_index=True,
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
    # df = df.merge(
    #     products[['product_id', 'aisle_id', 'department_id',
    #                'orders', 'reorders', 'reorder_rate']],
    #     on='product_id', how='left'
    # )
    products_reset = products.reset_index(drop=True)
    df = df.merge(
    products_reset[['product_id', 'aisle_id', 'department_id',
                    'orders', 'reorders', 'reorder_rate']],
    on='product_id', how='left')
 
    # ── user-level features ────────────────────────────────────────────────
    # users_reset = users.reset_index()
    # df = df.merge(
    #     users_reset[['user_id', 'total_items', 'total_distinct_items',
    #                  'average_days_between_orders', 'average_basket', 'nb_orders']],
    #     on='user_id', how='left'
    # )
    users.index.name = 'user_id'
    users_reset = users.reset_index()
    df = df.merge(
    users_reset[['user_id', 'total_items', 'total_distinct_items',
                 'average_days_between_orders', 'average_basket', 'nb_orders']],
    on='user_id', how='left')
 
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
    # train_orders = train_orders.merge(
    #     orders_sorted[['order_id', 'abs_day']],
    #     on='order_id', how='left'
    # )
    train_orders = train_orders.merge(
    orders_sorted[['abs_day']],
    left_on='order_id',
    right_index=True,
    how='left')
 
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
 




 # SECTION 6: Features List

 
FEATURES = [
    # ── temporal signals (core of day-level prediction) ──
    'UP_days_since_last',       # raw days since last bought
    'UP_days_overdue',          # how late past expected interval
    'UP_overdue_zscore',        # normalized overdue (accounts for habit consistency)
    'UP_interval_progress',     # 0.5=halfway, 1.0=due, 2.0=very overdue
    'UP_avg_interval',          # user's average reorder cycle for this product
    'UP_std_interval',          # consistency of buying habit
    'UP_min_interval',
    'UP_max_interval',
    'UP_purchase_count',        # total times ever purchased
 
    # ── user signals ──
    'nb_orders',
    'average_days_between_orders',
    'average_basket',
    'total_distinct_items',
 
    # ── product signals ──
    'reorder_rate',
    'aisle_id',
    'department_id',
 
    # ── target day context ──
    'target_day_offset',        # days from now being predicted
]
 
 

#Train / Test Split

 
print('splitting orders: train, test')
test_orders  = orders[orders.eval_set == 'test']
train_orders = orders[orders.eval_set == 'train']
 
train.set_index(['order_id', 'product_id'], inplace=True, drop=False)
 
 

# SECTION 8: Build Temporal Features

 
orders_sorted = build_absolute_timeline(orders)
up_temporal   = build_userproduct_temporal_features(priors, orders_sorted)
user_timeline = build_user_timeline(orders_sorted)
 
 

#Build Training Data & Train Model

 
df_train = build_training_data(
    train_orders, train,
    up_temporal, user_timeline,
    users, products, orders_sorted
)
 
print('training LightGBM model...')
X = df_train[FEATURES]
y = df_train['label']
 
d_train = lgb.Dataset(
    X, label=y,
    categorical_feature=['aisle_id', 'department_id']
)
 
params = {
    'boosting_type'   : 'gbdt',
    'objective'       : 'binary',
    'metric'          : 'binary_logloss',
    'num_leaves'      : 64,
    'max_depth'       : 8,
    'learning_rate'   : 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.9,
    'bagging_freq'    : 5,
    'min_child_samples': 20,
    'verbose'         : -1
}
 
bst = lgb.train(params, d_train, num_boost_round=200)
del d_train
 
 

#Predict — User ID + Target Day → Reorder Probability

 
def predict_for_user_on_day(user_id, target_day_offset, bst,
                             up_temporal, user_timeline,
                             users, products, top_k=10):
    """
    Given a user_id and target_day_offset (days from their last known order),
    returns all their products ranked by reorder probability.
 
    Example
    -------
    predict_for_user_on_day(user_id=1, target_day_offset=7, ...)
    → "What will user 1 likely reorder 7 days from now?"
    """
    df = build_prediction_features(
        user_ids=[user_id],
        target_day_offset=target_day_offset,
        up_temporal=up_temporal,
        user_timeline=user_timeline,
        users=users,
        products=products
    )
 
    df['reorder_probability'] = bst.predict(df[FEATURES])
 
    result = df[['user_id', 'product_id', 'reorder_probability',
                  'UP_days_since_last', 'UP_avg_interval',
                  'UP_interval_progress']].copy()
    result = result.sort_values('reorder_probability', ascending=False)
 
    print(f"\nTop {top_k} products for user {user_id} on day +{target_day_offset}:")
    print(result.head(top_k).to_string(index=False))
    return result
 
 

#Generate Test Submission
#(predict for all test users, 0 days offset = their next order day)

 
print('generating predictions for test users...')
THRESHOLD = 0.22
 
test_user_ids = test_orders['user_id'].unique().tolist()
 
df_test = build_prediction_features(
    user_ids=test_user_ids,
    target_day_offset=0,        # predicting for their immediate next order
    up_temporal=up_temporal,
    user_timeline=user_timeline,
    users=users,
    products=products
)
 
df_test = df_test.merge(
    test_orders[['user_id', 'order_id']],
    on='user_id', how='left'
)
 
print('predicting...')
df_test['reorder_probability'] = bst.predict(df_test[FEATURES])
 
# build submission
d = dict()
for row in df_test.itertuples():
    if row.reorder_probability > THRESHOLD:
        try:
            d[row.order_id] += ' ' + str(row.product_id)
        except KeyError:
            d[row.order_id] = str(row.product_id)
 
# for order in test_orders.order_id:
for order in test_orders.index: 
    if order not in d:
        d[order] = 'None'
 
sub = pd.DataFrame.from_dict(d, orient='index')
sub.reset_index(inplace=True)
sub.columns = ['order_id', 'products']
sub.to_csv('sub.csv', index=False)
print('submission saved to sub.csv')
 
 

#Example Usage

 

results = predict_for_user_on_day(
    user_id=1,
    target_day_offset=7,        
    bst=bst,
    up_temporal=up_temporal,
    user_timeline=user_timeline,
    users=users,
    products=products,
    top_k=10
)