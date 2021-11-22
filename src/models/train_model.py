import torch
from datetime import date, datetime
import numpy as np
from src.models.metrics import sensitivity, specificity, f1_score


class model_train():
    '''
    Class for training pytorch model
    '''
    def __init__(self, model, optimizer, loss_fn, writer = None, scheduler = None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.loss_fn.to(self.device)
        self.writer = writer


    def train(self,
              train_loader,
              val_loader, 
              epochs):
        '''
        Train model
        '''

        train_loss = torch.zeros(epochs)
        val_loss = torch.zeros(epochs)
        
        for epoch in range(epochs):
            time = datetime.now()
            running_train_loss = 0
            running_val_loss = 0

            num_batch = 1
            print('Epoch', epoch + 1, 'out of', epochs)
            self.model.train()
            for batch in train_loader:
                x = batch[0].float().to(self.device)
                y = batch[1].long().to(self.device)
                self.optimizer.zero_grad()
                out = self.model(x, training = True)
                loss = self.loss_fn(out, y)
                loss.backward()
                self.optimizer.step()

                running_train_loss += loss.detach().cpu()
                num_batch += 1

            if self.scheduler is not None:
                self.scheduler.step()

            train_loss[epoch] = running_train_loss/num_batch
            if self.writer is not None:
                self.writer.add_scalar('train/loss', train_loss[epoch], epoch)
            print('Training loss:', train_loss[epoch])

            num_batch = 1
            self.model.eval()
            for batch in val_loader:
                x = batch[0].float().to(self.device)
                y = batch[1].long().to(self.device)
                out = self.model(x)
                loss = self.loss_fn(out, y)

                running_val_loss += loss.detach().cpu()
                if num_batch == 1:
                    y_true = y.detach().cpu().numpy()
                    y_pred = torch.argmax(out, axis = -1).detach().cpu().numpy()
                else:
                    y_true = np.append(y_true, y.detach().cpu().numpy(), axis = 0)
                    y_pred = np.append(y_pred, torch.argmax(out, axis = -1).detach().cpu().numpy(), axis = 0)
                num_batch += 1
            
            sens = sensitivity(y_true, y_pred)
            spec = specificity(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            if self.writer is not None:
                self.writer.add_scalar('val/sens', sens, epoch)
                self.writer.add_scalar('val/spec', spec, epoch)
                self.writer.add_scalar('val/f1', f1, epoch)

            val_loss[epoch] = running_val_loss/num_batch
            if self.writer is not None:
                self.writer.add_scalar('val/loss', val_loss[epoch], epoch)
            print('Validation loss:', val_loss[epoch])
            epoch_time = (datetime.now()-time).total_seconds()
            print('Epoch time', epoch_time)
            if self.writer is not None:
                self.writer.add_scalar('Loss/epoch_time', epoch_time, epoch)
        
        self.writer.flush()
        return train_loss, val_loss
    
    def eval(self, data_loader):
        y_pred = None

        self.model.eval()
        for batch in data_loader:
            x = batch[0].float().to(self.device)
            out = self.model(x)
            y_class = torch.argmax(out, axis = -1).cpu().numpy()

            if y_pred is None:
                y_pred = y_class
            else:
                y_pred = np.append(y_pred, y_class, axis = 0)
        
        return y_pred







