from MF import MF

### create train, test
MF.split_train_test('rating_100000.csv', test_size=0.2, 
    train_dir='data/train.csv', test_dir='data/test.csv')

### create model and train
model = MF('data/train.csv', 'data/test.csv')

# K is embedding dim, lam is lambda for regularization
model.build(K = 2, lam = 0.1, print_every = 1, print_time=True,
    learning_rate = 2, max_iter = 1, user_based = 0)

model.fit()


### predict 
# return None if not existed
# return [(anime_id, pred_rating)] (of unrated anime)
pred = model.predict(user_id=0)
print(pred[:5])


### add user rating
# if out_file=None then will update straight to train_file
model.update_rating(user_id=0, movie_id=0, rating=10, out_file='alo.csv')