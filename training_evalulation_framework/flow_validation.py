import os
import yaml
import time

# config.yaml에서 환경 설정을 읽기
with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

# 삭제하지 않을 파일 리스트를 생성합니다.
excluded_files = [
    config['train_filepath'],
    config['test_filepath']
]

# config에서 나머지 파일들을 순회하며 삭제합니다.
for key, file_path in config.items():
    if file_path not in excluded_files and os.path.exists(file_path):
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

print("Files cleanup completed.")

def execute_logging(cmd):
    print(f"Start '{cmd}'")
    start_time = time.time()
    os.system(cmd)
    end_time = time.time()
    duration = end_time - start_time
    print(f"Done {duration:.2f} seconds")

def run_test(test_func, config_key, *args):
    global final_score
    try:
        result = test_func(*args)
        if result:
            final_score += score_per_item
            print(f"{config_key} check passed! (+{score_per_item} points)")
        else:
            print(f"{config_key} check failed!")
    except Exception as e:
        print(f"Error during {config_key} check: {e}")
        
commands = [
    f"python random_inference.py {config['test_filepath']} {config['y_test_filepath']} {config['y_rand_pred_filepath']}",
    f"python score.py {config['y_test_filepath']} {config['y_rand_pred_filepath']} {config['random_score_filepath']}",
    f"python train.py {config['train_filepath']} {config['model_filepath']}",
    f"python model_inference.py {config['model_filepath']} {config['test_filepath']} {config['y_pred_filepath']}",
    f"python score.py {config['y_test_filepath']} {config['y_pred_filepath']} {config['model_score_filepath']}"
]

for cmd in commands:
    try:
        execute_logging(cmd)
    except Exception as e:
        print(f"Error during command '{cmd}': {e}")

# 제대로된 테스트를 진행
def check_file_exists(filepath):
    return os.path.exists(filepath)

def check_float_content(filepath):
    try:
        with open(filepath, 'r') as file:
            float(file.read())
        return True
    except:
        return False

def score_validation(random_score_path, model_score_path, score_order):
    with open(random_score_path, 'r') as file:
        random_score = float(file.read())
    with open(model_score_path, 'r') as file:
        model_score = float(file.read())
    
    if score_order == 'high':
        return model_score > random_score
    elif score_order == 'low':
        return model_score < random_score
    else:
        print(f"Unknown score_order: {score_order}")
        return False

test_functions = {
    'y_rand_pred_filepath': lambda: check_file_exists(config['y_rand_pred_filepath']),
    'random_score_filepath': lambda: check_file_exists(config['random_score_filepath']) and check_float_content(config['random_score_filepath']),
    'model_filepath': lambda: check_file_exists(config['model_filepath']),
    'y_pred_filepath': lambda: check_file_exists(config['y_pred_filepath']),
    'model_score_filepath': lambda: check_file_exists(config['model_score_filepath']) and check_float_content(config['model_score_filepath']),
    'score_validation': lambda: score_validation(config['random_score_filepath'], config['model_score_filepath'], config['score_order'])
}

test_results = {}

for key, test_func in test_functions.items():
    try:
        test_results[key] = test_func()
    except Exception as e:
        print(f"Error checking '{key}': {e}")
        test_results[key] = False

base_score = 40
score_per_item = 10
final_score = base_score

for key, result in test_results.items():
    if result:
        final_score += score_per_item
        print(f"{key} check passed! (+{score_per_item} points)")
    else:
        print(f"{key} check failed!")

print(f"Total score: {final_score} / 100")