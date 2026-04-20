
import gc
import os
from multiprocessing import Pool, cpu_count
from functools import partial
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


class DataLoader:
    """Handles loading and initial processing of Instacart data."""

    def __init__(self, data_dir, batch_size=100000, use_gpu=True):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.n_workers = min(cpu_count(), 8)

    def load_data_in_batches(self, filepath, dtype_dict, **kwargs):
        """Load large CSV files in batches to reduce memory usage."""
        chunks = []
        for chunk in pd.read_csv(
        filepath, dtype=dtype_dict, chunksize=self.batch_size, **kwargs
        ):
            chunks.append(chunk)
            if len(chunks) % 10 == 0:
                print(f'Loaded {len(chunks)} chunks...')
            gc.collect()
        return pd.concat(chunks, ignore_index=True)

    def load_all_data(self):
        """Load all required datasets."""
        print('Loading prior orders...')
        priors = self.load_data_in_batches(
        self.data_dir + 'order_products__prior.csv',
        {
        'order_id': np.int32,
        'product_id': np.uint16,
        'add_to_cart_order': np.int16,
        'reordered': np.int8
        }
        )

        print('Loading train orders...')
        train = self.load_data_in_batches(
        self.data_dir + 'order_products__train.csv',
        {
        'order_id': np.int32,
        'product_id': np.uint16,
        'add_to_cart_order': np.int16,
        'reordered': np.int8
        }
        )

        print('Loading orders...')
        orders = self.load_data_in_batches(
        self.data_dir + 'orders.csv',
        {
        'order_id': np.int32,
        'user_id': np.int32,
        'eval_set': 'category',
        'order_number': np.int16,
        'order_dow': np.int8,
        'order_hour_of_day': np.int8,
        'days_since_prior_order': np.float32
        }
        )

        print('Loading products...')
        products = self.load_data_in_batches(
        self.data_dir + 'products.csv',
        {
        'product_id': np.uint16,
        'order_id': np.int32,
        'aisle_id': np.uint8,
        'department_id': np.uint8
        },
        usecols=['product_id', 'aisle_id', 'department_id']
        )

        self._print_data_info(priors, orders, train)
        return priors, train, orders, products

    def _print_data_info(self, priors, orders, train):
        """Print dataset information."""
        print(f'priors {priors.shape}: {", ".join(priors.columns)}')
        print(f'orders {orders.shape}: {", ".join(orders.columns)}')
        print(f'train {train.shape}: {", ".join(train.columns)}')


class FeatureEngineer:
    """Engineers features for the recommendation model."""

    def __init__(self, batch_size=100000):
        self.batch_size = batch_size

    def create_product_features(self, priors, products):
        """Create product-level features."""
        print('Computing product features...')
        prods = pd.DataFrame()
        prods['orders'] = priors.groupby(priors.product_id).size().astype(np.int32)
        prods['reorders'] = priors['reordered'].groupby(priors.product_id).sum().astype(np.float32)
        prods['reorder_rate'] = (prods.reorders / prods.orders).astype(np.float32)
        products = products.join(prods, on='product_id')
        products.set_index('product_id', drop=False, inplace=True)
        del prods
        return products

    def create_user_features(self, priors, orders):
        """Create user-level features."""
        print('Computing user features...')
        usr = pd.DataFrame()
        usr['average_days_between_orders'] = orders.groupby('user_id')['days_since_prior_order'].mean().astype(np.float32)
        usr['nb_orders'] = orders.groupby('user_id').size().astype(np.int16)

        users = pd.DataFrame()
        users['total_items'] = priors.groupby('user_id').size().astype(np.int16)
        users['all_products'] = priors.groupby('user_id')['product_id'].apply(set)
        users['total_distinct_items'] = users.all_products.map(len).astype(np.int16)

        users = users.join(usr)
        del usr
        users['average_basket'] = (users.total_items / users.nb_orders).astype(np.float32)
        print('User features shape:', users.shape)
        return users

    def create_user_product_features(self, priors):
        """Create user x product features with batch processing."""
        print('Computing user x product features with batch processing...')
        priors['user_product'] = priors.product_id + priors.user_id * 100000

        batch_dictionaries = []
        n_batches = (len(priors) + self.batch_size - 1) // self.batch_size

        for i in range(n_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, len(priors))
            batch = priors.iloc[start_idx:end_idx]

            if i % 10 == 0:
                print(f'Processing batch {i+1}/{n_batches}...')

            batch_dict = self._process_user_product_batch(batch)
            batch_dictionaries.append(batch_dict)

            del batch
            if i % 5 == 0:
                gc.collect()

        print('Merging batch results...')
        d = {}
        for batch_dict in batch_dictionaries:
            for key, value in batch_dict.items():
                if key not in d:
                    d[key] = value
                else:
                    d[key] = (
                    d[key][0] + value[0],
                    max(d[key][1], value[1]),
                    d[key][2] + value[2]
                    )

        del batch_dictionaries
        gc.collect()

        print('Converting to dataframe...')
        user_x_product = pd.DataFrame.from_dict(d, orient='index')
        del d
        user_x_product.columns = ['nb_orders', 'last_order_id', 'sum_pos_in_cart']
        user_x_product.nb_orders = user_x_product.nb_orders.astype(np.int16)
        user_x_product.last_order_id = user_x_product.last_order_id.map(lambda x: x[1]).astype(np.int32)
        user_x_product.sum_pos_in_cart = user_x_product.sum_pos_in_cart.astype(np.int16)
        print('User x product features count:', len(user_x_product))
        return user_x_product

    def _process_user_product_batch(self, batch_df):
        """Process a batch of user-product pairs efficiently."""
        d = {}
        for row in batch_df.itertuples():
            z = row.user_product
            if z not in d:
                d[z] = (
                1,
                (row.order_number, row.order_id),
                row.add_to_cart_order
                )
            else:
                d[z] = (
                d[z][0] + 1,
                max(d[z][1], (row.order_number, row.order_id)),
                d[z][2] + row.add_to_cart_order
                )
        return d


class TemporalFeatureEngineer:
    """Builds temporal features for day-level prediction."""

    def __init__(self, batch_size=100000):
        self.batch_size = batch_size

    def build_absolute_timeline(self, orders):
        """Convert relative days to absolute day counter per user."""
        print('Building absolute day timeline...')
        orders_sorted = orders.reset_index(drop=True).sort_values(['user_id', 'order_number']).copy()
        orders_sorted['days_since_prior_order'] = orders_sorted['days_since_prior_order'].fillna(0)
        orders_sorted['abs_day'] = orders_sorted.groupby('user_id')['days_since_prior_order'].cumsum()
        return orders_sorted

    def build_userproduct_temporal_features(self, priors, orders_sorted):
        """Build temporal features for user-product pairs."""
        print('Building temporal user-product features with batch processing...')

        priors_with_day = priors.merge(
        orders_sorted[['abs_day']],
        left_on='order_id',
        right_index=True,
        how='left'
        )

        priors_with_day = priors_with_day.sort_values(['user_id', 'product_id', 'order_number'])

        priors_with_day['prev_abs_day'] = priors_with_day.groupby(
        ['user_id', 'product_id']
        )['abs_day'].shift(1)

        priors_with_day['purchase_interval'] = (
        priors_with_day['abs_day'] - priors_with_day['prev_abs_day']
        )

        print('Processing temporal features in batches...')
        unique_user_products = priors_with_day[['user_id', 'product_id']].drop_duplicates()
        n_batches = (len(unique_user_products) + self.batch_size - 1) // self.batch_size

        temporal_results = []

        for i in range(n_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, len(unique_user_products))
            batch_user_products = unique_user_products.iloc[start_idx:end_idx]

            if i % 10 == 0:
                print(f'Processing temporal batch {i+1}/{n_batches}...')

            batch_mask = priors_with_day.set_index(['user_id', 'product_id']).index.isin(
            batch_user_products.set_index(['user_id', 'product_id']).index
            )
            batch_priors = priors_with_day[batch_mask]

            batch_temporal = batch_priors.groupby(['user_id', 'product_id']).agg(
            UP_avg_interval=('purchase_interval', 'mean'),
            UP_std_interval=('purchase_interval', 'std'),
            UP_min_interval=('purchase_interval', 'min'),
            UP_max_interval=('purchase_interval', 'max'),
            UP_last_abs_day=('abs_day', 'max'),
            UP_purchase_count=('product_id', 'count'),
            ).reset_index()

            temporal_results.append(batch_temporal)

            del batch_priors, batch_temporal
            if i % 5 == 0:
                gc.collect()

        print('Combining temporal batch results...')
        up_temporal = pd.concat(temporal_results, ignore_index=True)
        del temporal_results
        gc.collect()

        up_temporal['UP_std_interval'] = up_temporal['UP_std_interval'].fillna(0)
        return up_temporal

    def build_user_timeline(self, orders_sorted):
        """Build user timeline features."""
        user_timeline = orders_sorted.groupby('user_id').agg(
        user_last_abs_day=('abs_day', 'max'),
        user_total_orders=('order_number', 'max'),
        user_avg_days_between_orders=('days_since_prior_order', 'mean'),
        ).reset_index()
        return user_timeline

    def build_prediction_features(self, user_ids, target_day_offset, up_temporal,
    user_timeline, users, products):
        """Build features for prediction."""
        up = up_temporal[up_temporal['user_id'].isin(user_ids)].copy()
        ut = user_timeline[user_timeline['user_id'].isin(user_ids)].copy()

        df = up.merge(ut, on='user_id', how='left')
        df['target_abs_day'] = df['user_last_abs_day'] + target_day_offset

        df['UP_days_since_last'] = df['target_abs_day'] - df['UP_last_abs_day']
        df['UP_days_overdue'] = df['UP_days_since_last'] - df['UP_avg_interval']
        df['UP_overdue_zscore'] = df['UP_days_overdue'] / (df['UP_std_interval'] + 1e-5)
        df['UP_interval_progress'] = df['UP_days_since_last'] / (df['UP_avg_interval'] + 1e-5)

        products_reset = products.reset_index(drop=True)
        df = df.merge(
        products_reset[['product_id', 'aisle_id', 'department_id',
        'orders', 'reorders', 'reorder_rate']],
        on='product_id', how='left'
        )

        users.index.name = 'user_id'
        users_reset = users.reset_index()
        df = df.merge(
        users_reset[['user_id', 'total_items', 'total_distinct_items',
        'average_days_between_orders', 'average_basket', 'nb_orders']],
        on='user_id', how='left'
        )

        df['target_day_offset'] = target_day_offset
        return df

    def build_training_data(self, train_orders, train, up_temporal, user_timeline,
    users, products, orders_sorted):
        """Build training data with batch processing."""
        print('Building training data with batch processing...')

        train_orders = train_orders.merge(
        orders_sorted[['abs_day']],
        left_on='order_id',
        right_index=True,
        how='left'
        )

        n_train_batches = (len(train_orders) + self.batch_size - 1) // self.batch_size
        all_dfs = []

        for batch_idx in range(n_train_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min((batch_idx + 1) * self.batch_size, len(train_orders))
            batch_train_orders = train_orders.iloc[start_idx:end_idx]

            if batch_idx % 5 == 0:
                print(f' Processing training batch {batch_idx+1}/{n_train_batches} ({len(batch_train_orders)} orders)...')

            batch_dfs = []

            for i, row in enumerate(batch_train_orders.itertuples()):
                if i % 1000 == 0 and i > 0:
                    print(f' Processed {i}/{len(batch_train_orders)} orders in current batch...')

                user_id = row.user_id
                order_id = row.Index
                target_abs_day = row.abs_day

                user_history = orders_sorted[
                (orders_sorted['user_id'] == user_id) &
                (orders_sorted['abs_day'] < target_abs_day)
                ]
                if user_history.empty:
                    continue

                user_last_day = user_history['abs_day'].max()
                day_offset = int(target_abs_day - user_last_day)

                ut_user = user_timeline[user_timeline['user_id'] == user_id].copy()
                ut_user['user_last_abs_day'] = user_last_day

                df = self.build_prediction_features(
                user_ids=[user_id],
                target_day_offset=day_offset,
                up_temporal=up_temporal,
                user_timeline=ut_user,
                users=users,
                products=products
                )

                ordered_products = set(
                train[train['order_id'] == order_id]['product_id'].tolist()
                )
                df['label'] = df['product_id'].isin(ordered_products).astype(np.int8)
                batch_dfs.append(df)

            if batch_dfs:
                batch_result = pd.concat(batch_dfs, ignore_index=True)
                all_dfs.append(batch_result)

            del batch_train_orders, batch_dfs
            if batch_idx % 2 == 0:
                gc.collect()

        print('Combining all training batches...')
        return pd.concat(all_dfs, ignore_index=True)


class ModelTrainer:
    """Handles model training and prediction."""

    FEATURES = [
    'UP_days_since_last', 'UP_days_overdue', 'UP_overdue_zscore',
    'UP_interval_progress', 'UP_avg_interval', 'UP_std_interval',
    'UP_min_interval', 'UP_max_interval', 'UP_purchase_count',
    'nb_orders', 'average_days_between_orders', 'average_basket',
    'total_distinct_items', 'reorder_rate', 'aisle_id',
    'department_id', 'target_day_offset'
    ]

    def __init__(self, use_gpu=True, n_workers=8):
        self.use_gpu = use_gpu
        self.n_workers = n_workers
        self.model = None

    def train_model(self, df_train):
        """Train LightGBM model with GPU optimization."""
        print('Training LightGBM model with GPU optimization...')
        X = df_train[self.FEATURES]
        y = df_train['label']

        params = {
        'boosting_type': 'gbdt',
        'objective': 'binary',
        'metric': 'binary_logloss',
        'device': 'gpu' if self.use_gpu else 'cpu',
        'gpu_platform_id': 0,
        'gpu_device_id': 0,
        'gpu_use_dp': False,
        'num_leaves': 64,
        'max_depth': 8,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.9,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'verbose': -1,
        'n_jobs': self.n_workers,
        'force_row_wise': True,
        'force_col_wise': False
        }

        d_train = lgb.Dataset(
        X, label=y,
        categorical_feature=['aisle_id', 'department_id'],
        free_raw_data=False
        )

        print(f'Using GPU: {self.use_gpu}, Workers: {self.n_workers}')
        print(f'Training data shape: {X.shape}')

        self.model = lgb.train(
        params,
        d_train,
        num_boost_round=200,
        callbacks=[lgb.log_evaluation(period=50)]
        )

        del d_train, X, y
        gc.collect()
        print('Model training completed')
        return self.model

    def predict_for_user_on_day(self, user_id, target_day_offset, up_temporal,
    user_timeline, users, products, top_k=10):
        """Predict reorder probability for a user on a specific day."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")

        temporal_engineer = TemporalFeatureEngineer()
        df = temporal_engineer.build_prediction_features(
        user_ids=[user_id],
        target_day_offset=target_day_offset,
        up_temporal=up_temporal,
        user_timeline=user_timeline,
        users=users,
        products=products
        )

        df['reorder_probability'] = self.model.predict(df[self.FEATURES])

        result = df[[
        'user_id', 'product_id', 'reorder_probability',
        'UP_days_since_last', 'UP_avg_interval', 'UP_interval_progress'
        ]].copy()
        result = result.sort_values('reorder_probability', ascending=False)

        print(f"\nTop {top_k} products for user {user_id} on day +{target_day_offset}:")
        print(result.head(top_k).to_string(index=False))
        return result

    def generate_submission(self, df_test, test_orders, threshold=0.22):
        """Generate submission file."""
        print('Generating predictions for test users...')
        print(f'Predicting for {len(df_test)} test instances')
        df_test['reorder_probability'] = self.model.predict(df_test[self.FEATURES])

        print('Building submission file...')
        d = dict()
        submission_batches = []
        n_submission_batches = (len(df_test) + 100000 - 1) // 100000

        for batch_idx in range(n_submission_batches):
            start_idx = batch_idx * 100000
            end_idx = min((batch_idx + 1) * 100000, len(df_test))
            batch_df = df_test.iloc[start_idx:end_idx]

            if batch_idx % 10 == 0:
                print(f'Processing submission batch {batch_idx+1}/{n_submission_batches}...')

            batch_dict = {}
            for row in batch_df.itertuples():
                if row.reorder_probability > threshold:
                    try:
                        batch_dict[row.order_id] += ' ' + str(row.product_id)
                    except KeyError:
                        batch_dict[row.order_id] = str(row.product_id)

            submission_batches.append(batch_dict)
            del batch_dict, batch_df
            if batch_idx % 5 == 0:
                gc.collect()

        print('Merging submission batches...')
        for batch_dict in submission_batches:
            d.update(batch_dict)

        del submission_batches
        gc.collect()

        for order in test_orders.index:
            if order not in d:
                d[order] = 'None'

        sub = pd.DataFrame.from_dict(d, orient='index')
        sub.reset_index(inplace=True)
        sub.columns = ['order_id', 'products']
        sub.to_csv('sub.csv', index=False)
        print('Submission saved to sub.csv')
        print(f'Submission shape: {sub.shape}')
        return sub


class InstacartRecommender:
    """Main class for the Instacart recommendation system."""

    def __init__(self, data_dir, batch_size=100000, use_gpu=True):
        self.data_loader = DataLoader(data_dir, batch_size, use_gpu)
        self.feature_engineer = FeatureEngineer(batch_size)
        self.temporal_engineer = TemporalFeatureEngineer(batch_size)
        self.model_trainer = ModelTrainer(use_gpu, min(cpu_count(), 8))

        self.priors = None
        self.train = None
        self.orders = None
        self.products = None
        self.users = None
        self.user_x_product = None
        self.orders_sorted = None
        self.up_temporal = None
        self.user_timeline = None

    def run_pipeline(self):
        """Run the complete pipeline."""
        # Check if feature files already exist
        train_features_path = 'features/train_features.csv'
        test_features_path = 'features/test_features.csv'

        if os.path.exists(train_features_path) and os.path.exists(test_features_path):
            print('Loading existing feature files...')
            df_train = pd.read_csv(train_features_path)
            df_test = pd.read_csv(test_features_path)
            print(f'Loaded training features: {df_train.shape}')
            print(f'Loaded test features: {df_test.shape}')

            # Still need to load orders for test_orders and submission
            _, _, self.orders, _ = self.data_loader.load_all_data()
            self.orders.set_index('order_id', inplace=True, drop=True)
            test_orders = self.orders[self.orders.eval_set == 'test']
        else:
            print('Feature files not found. Processing features...')
            self.priors, self.train, self.orders, self.products = self.data_loader.load_all_data()

            self.orders.set_index('order_id', inplace=True, drop=True)
            self.priors = self.priors.join(self.orders, on='order_id')

            self.products = self.feature_engineer.create_product_features(self.priors, self.products)
            self.users = self.feature_engineer.create_user_features(self.priors, self.orders)
            self.user_x_product = self.feature_engineer.create_user_product_features(self.priors)

            self.orders_sorted = self.temporal_engineer.build_absolute_timeline(self.orders)
            self.up_temporal = self.temporal_engineer.build_userproduct_temporal_features(
            self.priors, self.orders_sorted
            )
            self.user_timeline = self.temporal_engineer.build_user_timeline(self.orders_sorted)

            test_orders = self.orders[self.orders.eval_set == 'test']
            train_orders = self.orders[self.orders.eval_set == 'train']

            self.train.set_index(['order_id', 'product_id'], inplace=True, drop=False)

            df_train = self.temporal_engineer.build_training_data(
            train_orders, self.train, self.up_temporal, self.user_timeline,
            self.users, self.products, self.orders_sorted
            )

            print('Saving training data to CSV...')
            df_train.to_csv(train_features_path, index=False)
            print(f'Training data saved: {df_train.shape}')

            print('Building test features...')
            df_test = self.temporal_engineer.build_prediction_features(
            user_ids=test_orders['user_id'].unique().tolist(),
            target_day_offset=0,
            up_temporal=self.up_temporal,
            user_timeline=self.user_timeline,
            users=self.users,
            products=self.products
            )

            df_test = df_test.merge(
            test_orders[['user_id']].reset_index(),
            on='user_id', how='left'
            )

            print('Saving test data to CSV...')
            df_test.to_csv(test_features_path, index=False)
            print(f'Test data saved: {df_test.shape}')

        self.model_trainer.train_model(df_train)
        submission = self.model_trainer.generate_submission(df_test, test_orders)

        # For prediction, we need the temporal features and user data
        # Only load if not already loaded
        if not hasattr(self, 'up_temporal') or self.up_temporal is None:
            print('Loading additional data for predictions...')
            self.priors, self.train, self.orders, self.products = self.data_loader.load_all_data()
            self.orders.set_index('order_id', inplace=True, drop=True)
            self.priors = self.priors.join(self.orders, on='order_id')

            self.products = self.feature_engineer.create_product_features(self.priors, self.products)
            self.users = self.feature_engineer.create_user_features(self.priors, self.orders)
            self.user_x_product = self.feature_engineer.create_user_product_features(self.priors)

            self.orders_sorted = self.temporal_engineer.build_absolute_timeline(self.orders)
            self.up_temporal = self.temporal_engineer.build_userproduct_temporal_features(
            self.priors, self.orders_sorted
            )
            self.user_timeline = self.temporal_engineer.build_user_timeline(self.orders_sorted)

        results = self.model_trainer.predict_for_user_on_day(
        user_id=1,
        target_day_offset=7,
        up_temporal=self.up_temporal,
        user_timeline=self.user_timeline,
        users=self.users,
        products=self.products,
        top_k=10
        )

        return results, submission


if __name__ == "__main__":
    DATA_DIR = '/home/bryson/.cache/kagglehub/datasets/yasserh/instacart-online-grocery-basket-analysis-dataset/versions/1/'

    recommender = InstacartRecommender(DATA_DIR)
    results, submission = recommender.run_pipeline()