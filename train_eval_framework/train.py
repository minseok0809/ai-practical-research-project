import sys
import os
import datetime
import regex as re
import argparse
import pandas as pd 
from tqdm import tqdm

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from transformers import AutoTokenizer, RobertaForSequenceClassification, AdamW



def main():

    train_path = sys.argv[1]
    model_save_path = sys.argv[2]

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
        model = RobertaForSequenceClassification.from_pretrained('klue/roberta-small').to(device)
    elif torch.cuda.is_available() == False:
        model = RobertaForSequenceClassification.from_pretrained('klue/roberta-small')

    train_data = pd.read_csv(train_path)
    train_data = train_data.loc[:4000]

    train_dataset = ClickbaitDetectionDataset(train_data)
    
    optimizer = AdamW(model.parameters(), lr=1e-5)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    train_time = []
    train_loss = []
    train_accuracy = []

    for i in range(5):
        total_loss = 0.0
        correct = 0
        total = 0
        batches = 0

        model.train()
        
        with tqdm(train_loader) as pbar:
            pbar.set_description("Epoch " + str(i + 1))        
            for input_ids_batch, attention_masks_batch, y_batch in pbar:
                optimizer.zero_grad()

                if torch.cuda.is_available() == True:
                    y_batch = y_batch.to(device)
                    y_pred = model(input_ids_batch.to(device), attention_mask=attention_masks_batch.to(device))[0]
               
                elif torch.cuda.is_available() == False:
                    y_batch = y_batch
                    y_pred = model(input_ids_batch, attention_mask=attention_masks_batch)[0]
                                
        
                one_loss = F.cross_entropy(y_pred, y_batch)
                one_loss.backward()
                optimizer.step()

                total_loss += one_loss.item()

                _, predicted = torch.max(y_pred, 1)
                correct += (predicted == y_batch).sum()
                total += len(y_batch)

                batches += 1
                # if batches % 100 == 0:
                # print("Batch Loss:", total_loss, "Accuracy:", correct.float() / total)

                elapsed = pbar.format_dict['elapsed']
                elapsed_str = pbar.format_interval(elapsed)
                
        

        if len(elapsed_str) == 5:
            elapsed_str = "00:" + elapsed_str
        elapsed_str = str(datetime.datetime.strptime(elapsed_str, '%H:%M:%S').time())    
        
        pbar.close()  
        train_time.append(elapsed_str)
        total_loss = round(total_loss, 4)                             
        train_loss.append(total_loss)
        accuracy = round((correct.float() / total).item(), 4)
        train_accuracy.append(accuracy)
        print("Train Time",  elapsed_str, "  ", "Train Loss:", total_loss,  "  ",  "Train Accuracy:", accuracy)    
    
    torch.save(model.state_dict(), model_save_path)

if __name__ == '__main__':
    main()    
