import pandas as pd
import random
import sys

test_path = sys.argv[1]
y_test_save_path = sys.argv[2]
y_rand_pred_save_path = sys.argv[3]

test = pd.read_csv(test_path)

test = test.loc[:500]
labels = test['Label']
y_test = [label for label in labels]
y_rand = [random.choice([0, 1])] * len(y_test)

y_test = pd.DataFrame({'Label': y_test})
y_test.to_csv(y_test_save_path, index=False)

y_rand_pred = pd.DataFrame({'Label': y_rand})
y_rand_pred.to_csv(y_rand_pred_save_path, index=False)
