import pandas as pd
from MF_core import MF_core

class MF:
    def __init__(self, train_file, test_file):
        self.train_file = train_file
        self.test_file = test_file
        self.read_data()
        
    def read_data(self):
        self.train_set = pd.read_csv(self.train_file)[['user_id', 'anime_id', 'rating']]
        self.test_set = pd.read_csv(self.test_file)[['user_id', 'anime_id', 'rating']]
        self.user_ids = list(set(self.train_set['user_id'].values))
        self.rate_train = self.train_set.to_numpy().copy()
        self.rate_test = self.test_set.to_numpy().copy()
        
        # indices start from 0
        self.rate_train[:, 2] -= 1
        self.rate_test[:, 2] -= 1
        
    def build(self, K = 10, lam = .1, print_every = 10, 
            learning_rate = 0.75, max_iter = 100, print_time=False,
            user_based = 0 
            ):
        self.MF_obj = MF_core(self.rate_train, K=K, lam = lam, 
            print_every = print_every, learning_rate = learning_rate, 
            max_iter = max_iter, user_based = user_based, print_time=print_time)
        
    def fit(self):
        self.MF_obj.fit()

    def predict(self, user_id):
        if user_id not in self.user_ids: 
            return None
        pred = self.MF_obj.pred_for_user(user_id)
        pred = sorted(pred, key=lambda x: x[1], reverse=True)
        return pred
        
    def update_rating(self, user_id, movie_id, rating, out_file=None):
        flag = False
        for i in range(len(self.train_set)):
            if self.train_set['user_id'][i]==user_id and \
                self.train_set['anime_id'][i]==movie_id:
                self.train_set['rating'][i]=rating
                flag = True
                break
        if not flag:
            new_df = pd.DataFrame(
                {'user_id': [user_id], 'anime_id': [movie_id], 'rating': [rating]})
            self.train_set = self.train_set.append(new_df, ignore_index=True)
        self.train_set = self.train_set.sort_values(by=['user_id', 'anime_id'])
        self.train_set = self.train_set.reset_index().drop(columns=['index'])
        
        if out_file is not None:
            self.train_set.to_csv(out_file)
        else:
            self.train_set.to_csv(self.train_file)
        
    def eval(self):
        # evaluate on test data
        RMSE = self.MF_obj.evaluate_RMSE(self.rate_test)
        print('\nRMSE =', RMSE)
    
    def split_train_test(file, test_size, train_dir, test_dir):
        df = pd.read_csv(file)[['user_id', 'anime_id', 'rating']]
        from sklearn.model_selection import train_test_split
        df_train, df_test, _, _ = train_test_split(df, [0]*len(df), test_size=test_size)
        df_train = df_train.reset_index().drop(columns=['index'])
        df_test = df_test.reset_index().drop(columns=['index'])
        
        swap_idx=[]
        for i in range(len(df_test)):
            if df_test['user_id'][i] not in df_train['user_id'].values:
                swap_idx += [i]
                
        df_train = df_train.append(df_test.iloc[swap_idx, :])
        df_test = df_test.drop(swap_idx)
        
        df_train = df_train.sort_values(by=['user_id', 'anime_id'])
        df_train = df_train.reset_index().drop(columns=['index'])
        df_test = df_test.sort_values(by=['user_id', 'anime_id'])
        df_test = df_test.reset_index().drop(columns=['index'])
        
        df_train.to_csv(train_dir)
        df_test.to_csv(test_dir)
                                    
    

