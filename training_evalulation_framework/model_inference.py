import sys
import os
import glob
import datetime
import regex as re
import argparse
import pandas as pd 
from tqdm import tqdm

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import AutoTokenizer, AutoModelForSequenceClassification


def main():

    model_path = sys.argv[1]
    test_path = sys.argv[2]
    y_pred_save_path = sys.argv[3]

    class ClickbaitDetectionDataset(Dataset):
    
        def __init__(self, dataset):
            self.dataset = dataset
            self.tokenizer = AutoTokenizer.from_pretrained('klue/roberta-large')
        
        def __len__(self):
            return len(self.dataset)
        
        def __getitem__(self, idx):
            row = self.dataset.iloc[idx, 0:2].values
            text = row[0]
            y = row[1]

            inputs = self.tokenizer(
                text, 
                return_tensors='pt',
                truncation=True,
                max_length=256,
                pad_to_max_length=True,
                add_special_tokens=True
                )
            
            input_ids = inputs['input_ids'][0]
            attention_mask = inputs['attention_mask'][0]

            return input_ids, attention_mask, y    


    if torch.cuda.is_available() == True:
        device = torch.device("cuda")
        model = AutoModelForSequenceClassification.from_pretrained('klue/roberta-small').to(device)
    elif torch.cuda.is_available() == False:
        model = AutoModelForSequenceClassification.from_pretrained('klue/roberta-small')

    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)
    model.eval()

    test_data = pd.read_csv(test_path)
    test_data = test_data.loc[:500]
    test_dataset = ClickbaitDetectionDataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    y_pred_index = [num for num in range(1, len(test_data)+1)]
    
    prediction = []
    test_time = []
    test_accuracy = []

    for i in range(1):
        correct = 0
        total = 0
        batches = 0
        num = 0
        
        with tqdm(test_loader) as pbar:
            pbar.set_description("Epoch " + str(i + 1))        
            for input_ids_batch, attention_masks_batch, y_batch in pbar:
                # optimizer.zero_grad()

                if torch.cuda.is_available() == True:
                    y_batch = y_batch.to(device)
                    y_pred = model(input_ids_batch.to(device), attention_mask=attention_masks_batch.to(device))[0]
               
                elif torch.cuda.is_available() == False:
                    y_batch = y_batch
                    y_pred = model(input_ids_batch, attention_mask=attention_masks_batch)[0]

                _, predicted = torch.max(y_pred, 1)
                correct += (predicted == y_batch).sum()
                total += len(y_batch)
                    
                batches += 1

                for i in predicted.tolist():
                    prediction.append(i)   
                # if batches % 100 == 0:
                # print("Batch Loss:", total_loss, "Accuracy:", correct.float() / total)
            elapsed = pbar.format_dict['elapsed']
            elapsed_str = pbar.format_interval(elapsed)
                
        y_pred_save = pd.DataFrame({'Label':prediction}) 
        y_pred_save.to_csv(y_pred_save_path, index=False)
        
        if len(elapsed_str) == 5:
            elapsed_str = "00:" + elapsed_str
        elapsed_str = str(datetime.datetime.strptime(elapsed_str, '%H:%M:%S').time())    
        
        pbar.close()  
        test_time.append(elapsed_str) 
        accuracy = round((correct.float() / total).item(), 4)
        test_accuracy.append(accuracy)
        print("Test Time",  elapsed_str, "  ", "Test Accuracy:", accuracy)  

if __name__ == '__main__':
    main()    

