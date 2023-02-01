def create_XGB(max_depth=8, learning_rate=0.025, subsample=0.85,
               colsample_bytree=0.35, eval_metric='logloss', objective='binary:logistic',
               tree_method='gpu_hist', seed=1):
    return xgBoost(max_depth=8, learning_rate=0.025, subsample=0.85,
                   colsample_bytree=0.35, eval_metric='logloss', objective='binary:logistic',
                   tree_method='gpu_hist', seed=1)


class xgBoost():
    def __init__(self, max_depth=8, learning_rate=0.025, subsample=0.85,
                 colsample_bytree=0.35, eval_metric='logloss', objective='binary:logistic',
                 tree_method='gpu_hist', seed=1):
        self.max_depth = 8
        self.learning_rate = 0.025
        self.subsample = 0.85
        self.colsample_bytree = 0.35
        self.eval_metric = 'logloss'
        self.objective = 'binary:logistic'
        self.tree_method = 'gpu_hist'
        self.seed = 1

    def __getparams__(self):
        dict_ = {
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': self.subsample,
            'colsample_bytree': self.subsample,
            'eval_metric': self.eval_metric,
            'objective': self.objective,
            'tree_method': self.tree_method,
            'seed': self.seed,
        }
        return dict_
