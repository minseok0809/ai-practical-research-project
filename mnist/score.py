import random
import argparse
import pandas as pd


def main():
    
    parer = argparse.ArgumentParser()
    parer.add_argument('--train_dataset', type=str, default='data/train.csv')
    parer.add_argument('--test_dataset', type=str, default='data/t10k.csv')
    parer.add_argument('--label', type=list, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    args = parer.parse_args('')   

    X_train = pd.read_csv(args.train_dataset)
    X_train = X_train.rename(columns={'5':'label'})
    X_test = pd.read_csv(args.test_dataset)
    X_test = X_test.rename(columns={'7':'label'})
    y_train = X_train['label'] 
    x_train = X_train.drop(['label'], axis=1)
    y_test = X_test['label'] 
    x_test = X_test.drop(['label'], axis=1)

    train_x = (x_train.values) / 255
    train_y = y_train.values.flatten() 

    num = len(y_test)
    y_rand_list = []

    for i in range(0, num):
        n = random.randint(args.label[0], args.label[-1])
        y_rand_list.append(n)

    y_rand_df = pd.DataFrame({"Label":y_rand_list})
    y_rand_df.to_csv("log/y_rand.csv", index = False)
    # y_rand_df.to_excel("log/y_rand.xlsx", index = False)
    y_rand = y_rand_df["Label"]
    y_test.to_csv("log/y_test.csv", index = False)
    # y_test.to_excel("log/y_test.xlsx", index = False)   

    def evaluate(y_test, y_rand):
        sum = 0
        for i, j in zip(y_test, y_rand):
            if i == j:
                sum += 1
        accuracy = sum / len(y_test)
        return accuracy 
    
    accuracy = evaluate(y_test, y_rand)
    print('Accuracy: {:3.2f} %'.format(accuracy*100))

if __name__ == '__main__':
    main()    
