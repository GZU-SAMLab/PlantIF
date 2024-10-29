import torch
import torch.nn as nn
import numpy as np
from config import get_config
from sklearn.metrics import accuracy_score
import models
from utils import to_gpu, time_desc_decorator, DiffLoss, MSE, SIMSE, CMD
from utils.custom_dataset import CustomDataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import BertTokenizer
from random import random

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def eval(test_config,test_data_loader,model_name,criterion):
    

    y_true, y_pred = [], []
    eval_loss, eval_loss_diff = [], []

    dataloader = test_data_loader

    model = getattr(models, 'MISA')(test_config)

    model.load_state_dict(torch.load(
                f'checkpoints/model_{model_name}.std'))
    model.cuda()
    model.eval()

    with torch.no_grad():

        for batch in dataloader:
            
            vision, y, text_input = batch
            input_ids, attention_masks, token_type_ids = text_input['input_ids'].squeeze(1), text_input[
                'attention_mask'].squeeze(1), text_input['token_type_ids'].squeeze(1)

            batch_size = input_ids.size(0)
            #input_ids = to_gpu(input_ids)
            #attention_masks = to_gpu(attention_masks)
            #token_type_ids = to_gpu(token_type_ids)
            #vision = to_gpu(vision)
            #y = to_gpu(y)
            input_ids = input_ids.to(device)
            attention_masks = attention_masks.to(device)
            token_type_ids =token_type_ids.to(device)
            vision =vision.to(device)
            y = y.to(device)
            y_tilde = model(vision, input_ids, attention_masks, token_type_ids)

            if test_config.data == "ur_funny":
                y = y.squeeze()
            cls_loss = criterion(y_tilde, y)
            loss = cls_loss

            eval_loss.append(loss.item())
            y_pred.extend(torch.argmax(y_tilde, dim=-1).detach().cpu().numpy())
            y_true.extend(y.detach().cpu().numpy())

    eval_loss = np.mean(eval_loss)
    # y_true = np.concatenate(y_true, axis=0).squeeze()
    # y_pred = np.concatenate(y_pred, axis=0).squeeze()

    # accuracy = self.calc_metrics(y_true, y_pred, mode, to_print)
    accuracy = accuracy_score(y_true, y_pred)

    print("test acc is %f, valid loss is %f"%(accuracy, eval_loss))


if __name__ == '__main__':
    
    # Setting random seed
    random_name = str(random())
    random_seed = 336   
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)

    rgb_mean = [0.5, 0.5, 0.5]
    rgb_std = [0.5, 0.5, 0.5]

    transform_val = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(rgb_mean, rgb_std),
    ])
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    train_dataset = CustomDataset("../leaf_diseases/train.csv", "../leaf_diseases/", transform_val, tokenizer)
    dev_dataset = CustomDataset("../leaf_diseases/test.csv", "../leaf_diseases/", transform_val, tokenizer)
    
    #train_dataset = CustomDataset('Plant_leaf_disease/train.csv', 'Plant_leaf_disease', transform_val, tokenizer)
    #dev_dataset = CustomDataset('Plant_leaf_disease/test.csv', 'Plant_leaf_disease', transform_val, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    dev_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)

    model_name="2023-07-26_13:37:45"
    test_config = get_config(mode='train')
    criterion = nn.CrossEntropyLoss(reduction="mean")
    eval(test_config,dev_loader,model_name,criterion)