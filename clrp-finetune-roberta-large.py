# coding: utf-8
import os
from pathlib import Path
from scripts.imports import *
from scripts.config import *
from scripts.dataset import *
from scripts.model import *

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONASSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(seed=Config.seed)

 
kfold_df = pd.read_csv('/home/fcq/Li/nlp_learn/datasets/kaggle_CLPR/new_kfold.csv')
aux_kfold_df = pd.read_csv('/home/fcq/Li/nlp_learn/datasets/kaggle_CLPR/kfold_parsed_annotated_by_all_models.csv')

def create_optimizer(model):
    named_parameters = list(model.named_parameters())    
    
    roberta_parameters = named_parameters[:389]    
    attention_parameters = named_parameters[391:395]
    regressor_parameters = named_parameters[395:]
        
    attention_group = [params for (name, params) in attention_parameters]
    regressor_group = [params for (name, params) in regressor_parameters]
    
    parameters = []
    parameters.append({"params": attention_group})
    parameters.append({"params": regressor_group})
    # increase lr every second layer
    increase_lr_every_k_layer = 1
    lrs = np.linspace(1, 5, 24 // increase_lr_every_k_layer)
    for layer_num, (name, params) in enumerate(roberta_parameters):
        weight_decay = 0.0 if "bias" in name else 0.01
        splitted_name = name.split('.')
        lr = Config.lr
        if len(splitted_name) >= 4 and str.isdigit(splitted_name[3]):
            layer_num = int(splitted_name[3])
            lr = lrs[layer_num // increase_lr_every_k_layer] * Config.lr 

        parameters.append({"params": params,
                           "weight_decay": weight_decay,
                           "lr": lr})

    return optim.AdamW(parameters)

 
class DynamicPadCollate:
 def __call__(self,batch):
             
     out = {'input_ids' :[],
            'attention_mask':[],
             'label':[]
     }
     
     for i in batch:
         for k,v in i.items():
             out[k].append(v)
             
     max_pad =0

     for p in out['input_ids']:
         if max_pad < len(p):
             max_pad = len(p)
                 

     for i in range(len(batch)):
         
         input_id = out['input_ids'][i]
         att_mask = out['attention_mask'][i]
         text_len = len(input_id)
         
         out['input_ids'][i] = (out['input_ids'][i].tolist() + [1] * (max_pad - text_len))[:max_pad]
         out['attention_mask'][i] = (out['attention_mask'][i].tolist() + [0] * (max_pad - text_len))[:max_pad]
     
     out['input_ids'] = torch.tensor(out['input_ids'],dtype=torch.long)
     out['attention_mask'] = torch.tensor(out['attention_mask'],dtype=torch.long)
     out['label'] = torch.tensor(out['label'],dtype=torch.float)
     
     return out


class AvgCounter:
 def __init__(self):
     self.reset()
     
 def update(self, loss, n_samples):
     self.loss += loss * n_samples
     self.n_samples += n_samples
     
 def avg(self):
     return self.loss / self.n_samples
 
 def reset(self):
     self.loss = 0
     self.n_samples = 0

class EvaluationScheduler:
 def __init__(self, evaluation_schedule, penalize_factor=1, max_penalty=8):
     self.evaluation_schedule = evaluation_schedule
     self.evaluation_interval = self.evaluation_schedule[0][1]
     self.last_evaluation_step = 0
     self.prev_loss = float('inf')
     self.penalize_factor = penalize_factor
     self.penalty = 0
     self.prev_interval = -1
     self.max_penalty = max_penalty

 def step(self, step):
     # should we to make evaluation right now
     if step >= self.last_evaluation_step + self.evaluation_interval:
         self.last_evaluation_step = step
         return True
     else:
         return False
     
         
 def update_evaluation_interval(self, last_loss):
     # set up evaluation_interval depending on loss value
     cur_interval = -1
     for i, (loss, interval) in enumerate(self.evaluation_schedule[:-1]):
         if self.evaluation_schedule[i+1][0] < last_loss < loss:
             self.evaluation_interval = interval
             cur_interval = i
             break
#         if last_loss > self.prev_loss and self.prev_interval == cur_interval:
#             self.penalty += self.penalize_factor
#             self.penalty = min(self.penalty, self.max_penalty)
#             self.evaluation_interval += self.penalty
#         else:
#             self.penalty = 0
         
     self.prev_loss = last_loss
     self.prev_interval = cur_interval
     
       
     
def make_dataloader(data, tokenizer, is_train=True):
 dataset = CLRPDataset(data, tokenizer=tokenizer, max_len=Config.max_len)
 if is_train:
     sampler = RandomSampler(dataset)
 else:
     sampler = SequentialSampler(dataset)

 batch_dataloader = DataLoader(dataset, sampler=sampler, batch_size=Config.batch_size, pin_memory=True, collate_fn=DynamicPadCollate())
 return batch_dataloader
                
         
class Trainer:
 def __init__(self, train_dl, val_dl, model, optimizer, scheduler, scaler, criterion, model_num):
     self.train_dl = train_dl
     self.val_dl = val_dl
     self.model = model
     self.optimizer = optimizer
     self.scheduler = scheduler
     self.device = Config.device
     self.batches_per_epoch = len(self.train_dl)
     self.total_batch_steps = self.batches_per_epoch * Config.epochs
     self.criterion = criterion
     self.model_num = model_num
     
     self.scaler = scaler
             
 def run(self):
     patience = 15
     record_info = {
         'train_loss': [],
         'val_loss': [],
     }
     
     best_val_loss = float('inf')
     evaluation_scheduler = EvaluationScheduler(Config.eval_schedule)
     train_loss_counter = AvgCounter()
     step = 0
     
     model_updated_step = 0
     evaluation_step = 0
     
     for epoch in range(Config.epochs):
         
         print(f'{r_}Epoch: {epoch+1}/{Config.epochs}{sr_}')
         start_epoch_time = time()
         
         for batch_num, batch in enumerate(self.train_dl):
             train_loss = self.train(batch, step)
#                 print(f'{epoch+1}#[{step+1}/{len(self.train_dl)}]: train loss - {train_loss.item()}')

             train_loss_counter.update(train_loss, len(batch))
             record_info['train_loss'].append((step, train_loss.item()))

             if evaluation_scheduler.step(step):
                 val_loss = self.evaluate()
                 
                 record_info['val_loss'].append((step, val_loss.item()))        
                 print(f'\t\t{bb_}{r_}[{evaluation_step-model_updated_step}] {sr_}{epoch+1}#[{batch_num+1}/{self.batches_per_epoch}]: train loss - {train_loss_counter.avg()} | val loss - {val_loss}',)
                 train_loss_counter.reset()

                 if val_loss < best_val_loss:
                     
                     best_val_loss = val_loss.item()
                     print(f"\t\t{g_}Val loss decreased from {best_val_loss} to {val_loss}{sr_}")
                     torch.save(self.model, models_dir / f'best_model_{self.model_num}.pt')
                     model_updated_step = evaluation_step
                     
                     
                 evaluation_scheduler.update_evaluation_interval(val_loss.item())
                 evaluation_step += 1
                 if evaluation_step - model_updated_step > patience:
                     print(f'\t{bb_}{r_}Model does"t converge. Stop training...{sr_}')
                     break         
             
             step += 1

             
         end_epoch_time = time()
         print(f'{bb_}{y_}The epoch took {end_epoch_time - start_epoch_time} sec..{sr_}')

     return record_info, best_val_loss
         

 def train(self, batch, batch_step):
     self.model.train()
     sent_id, mask, labels = batch['input_ids'].to(self.device), batch['attention_mask'].to(self.device), batch['label'].to(self.device)
     with autocast():
         preds = self.model(sent_id, mask)
         train_loss = self.criterion(preds, labels.unsqueeze(1))
     
     self.scaler.scale(train_loss).backward()
#         train_loss.backward()
     
     if (batch_step + 1) % Config.gradient_accumulation or batch_step+1 == self.total_batch_steps:
         self.scaler.step(self.optimizer)
         self.scaler.update()
#             self.optimizer.step()
         self.model.zero_grad() 
     self.scheduler.step()
     return torch.sqrt(train_loss)

 def evaluate(self):
     self.model.eval()
     val_loss_counter = AvgCounter()

     for step,batch in enumerate(self.val_dl):
         sent_id, mask, labels = batch['input_ids'].to(self.device), batch['attention_mask'].to(self.device), batch['label'].to(self.device)
         with torch.no_grad():
             with autocast():
                 preds = self.model(sent_id, mask)
                 loss = self.criterion(preds,labels.unsqueeze(1))
             val_loss_counter.update(torch.sqrt(loss), len(labels))
     return val_loss_counter.avg()
 
 
def mse_loss(y_true,y_pred):
 return nn.functional.mse_loss(y_true,y_pred)


best_scores = []

for model_num in range(5): 
    print(f'{bb_}{w_}  Model#{model_num+1}  {sr_}')

    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
    config = AutoConfig.from_pretrained(Config.model_name)
    config.update({
            "hidden_dropout_prob": 0.0,
            "layer_norm_eps": 1e-7,
            "output_hidden_states": True
            }) 
    
    l_train_fold = kfold_df[kfold_df.fold!=model_num]
    r_train_fold = aux_kfold_df[aux_kfold_df.fold!=model_num]
    train_df = pd.concat([l_train_fold, r_train_fold])
    
    train_dl = make_dataloader(train_df, tokenizer)
    val_dl = make_dataloader(kfold_df[kfold_df.fold==model_num], tokenizer, is_train=False)

#     train_dl = make_dataloader(train_df, tokenizer)
#     val_dl = make_dataloader(val_df, tokenizer, is_train=False)

    transformer = AutoModel.from_pretrained(Config.model_name, config=config)  

    model = CLRPModel(transformer, config)
    
    model = model.to(Config.device)
    optimizer = create_optimizer(model)
    scaler = GradScaler()
#     optimizer = optim.AdamW(model.parameters(), lr=Config.lr)
    scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_training_steps=Config.epochs * len(train_dl),
            num_warmup_steps=len(train_dl) * Config.epochs * 0.11)  

    criterion = mse_loss

    trainer = Trainer(train_dl, val_dl, model, optimizer, scheduler, scaler, criterion, model_num)
    record_info, best_val_loss = trainer.run()
    best_scores.append(best_val_loss)    
    
    steps, train_losses = list(zip(*record_info['train_loss']))
    plt.plot(steps, train_losses, label='train_loss')
    steps, val_losses = list(zip(*record_info['val_loss']))
    plt.plot(steps, val_losses, label='val_loss')
    plt.legend()
    plt.show()
    
print('Best val losses:', best_scores)
print('Avg val loss:', np.array(best_scores).mean())
