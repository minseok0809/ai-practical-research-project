import pandas as pd
import sys

y_test_path = sys.argv[1]
y_pred_path = sys.argv[2]
score_save_path = sys.argv[3]

y_test = pd.read_csv(y_test_path)
y_pred = pd.read_csv(y_pred_path)

y_test = y_test['Label'].values.tolist()
y_pred = y_pred['Label'].values.tolist()

sum = 0
for i, j in zip(y_test, y_pred):
    if i == j:
        sum += 1
score = sum / len(y_test)

with open(score_save_path, 'w') as file:
    file.write(str(score))
