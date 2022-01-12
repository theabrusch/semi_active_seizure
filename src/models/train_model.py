import torch
from datetime import date, datetime
import numpy as np
from pathlib import Path
from src.models.metrics import sensitivity, specificity
from sklearn.metrics import f1_score, confusion_matrix, precision_score
import optuna


class model_train():
    '''
    Class for training pytorch model
    '''
    def __init__(self, model, optimizer, loss_fn, 
                 choose_best = True, writer = None, scheduler = None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.loss_fn.to(self.device)
        self.writer = writer
        self.choose_best = choose_best

    def train(self,
              train_loader,
              val_loader, 
              safe_best_model = False,
              test_loader = None,
              trial = None,
              early_stopping = False,
              epochs = 10):
        '''
        Train model
        '''

        train_loss = torch.zeros(epochs)
        val_loss = torch.zeros(epochs)
        f1_scores = torch.zeros(epochs)
        checkpoint_path = 'models/checkpoints/' + str(datetime.now())        
        p = Path(checkpoint_path)
        p.mkdir(parents=True, exist_ok=True)
        
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
            
            if self.choose_best:
                model_check = checkpoint_path + '/epoch_' + str(epoch) + '.pt'
                torch.save({'model_state_dict': self.model.state_dict()},
                            model_check)

            train_loss[epoch] = running_train_loss/num_batch
            if self.writer is not None:
                self.writer.add_scalar('train/loss', train_loss[epoch], epoch)
            print('Training loss:', train_loss[epoch])

            # Compute validation loss and metrics
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
            prec = precision_score(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred, normalize = 'true')
            tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]

            f1_scores[epoch] = f1

            if trial is not None:
                trial.report(f1, epoch)
                if trial.should_prune():
                    trial.set_user_attr('sens', sens)
                    trial.set_user_attr('spec', spec)
                    raise optuna.exceptions.TrialPruned()

            if self.writer is not None:
                self.writer.add_scalar('val/sens', sens, epoch)
                self.writer.add_scalar('val/spec', spec, epoch)
                self.writer.add_scalar('val/f1', f1, epoch)
                self.writer.add_scalar('val/precision', prec, epoch)
                self.writer.add_scalar('val_raw/true_pos', tp, epoch)
                self.writer.add_scalar('val_raw/false_neg', fn, epoch)
                self.writer.add_scalar('val_raw/false_pos', fp, epoch)
                self.writer.add_scalar('val_raw/true_neg', tn, epoch)

            val_loss[epoch] = running_val_loss/num_batch
            if self.writer is not None:
                self.writer.add_scalar('val/loss', val_loss[epoch], epoch)
            print('Validation loss:', val_loss[epoch])

            #Compute test loss and metrics
            if test_loader is not None:
                num_batch = 1
                running_test_loss = 0 
                self.model.eval()
                for batch in test_loader:
                    x = batch[0].float().to(self.device)
                    y = batch[1].long().to(self.device)
                    out = self.model(x)
                    loss = self.loss_fn(out, y)

                    running_test_loss += loss.detach().cpu()
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
                prec = precision_score(y_true, y_pred)
                test_loss = running_test_loss/num_batch

                if self.writer is not None:
                    self.writer.add_scalar('test/sens', sens, epoch)
                    self.writer.add_scalar('test/spec', spec, epoch)
                    self.writer.add_scalar('test/f1', f1, epoch)
                    self.writer.add_scalar('test/precision', prec, epoch)
                    self.writer.add_scalar('test/loss', test_loss, epoch)

            epoch_time = (datetime.now()-time).total_seconds()
            print('Epoch time', epoch_time)
            if self.writer is not None:
                self.writer.add_scalar('Loss/epoch_time', epoch_time, epoch)

            if early_stopping and epoch > 0:
                if abs(train_loss[epoch]-train_loss[epoch-1]) < 1e-4:
                    break
        if self.choose_best and epochs>0:
            best_epoch = torch.argmax(f1_scores).item()
            best_model_path = checkpoint_path + '/epoch_' + str(best_epoch) + '.pt'
            checkpoint = torch.load(best_model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
        elif safe_best_model:
            model_check = checkpoint_path + '/final_model' + '.pt'
            torch.save({'model_state_dict': self.model.state_dict()},
                        model_check)
        if self.writer is not None:
            self.writer.flush()
        if trial is not None:
            return f1_scores[-1], sens, spec
        else:
            return train_loss, val_loss
    
    def eval(self, data_loader, return_seiz_type = False):
        y_pred = None

        self.model.eval()
        i = 1
        for batch in data_loader:
            print('Batch', i, 'out of',  data_loader.__len__())
            i+=1
            x = batch[0].float().to(self.device)
            out = self.model(x)
            y_class = torch.argmax(out, axis = -1).cpu().numpy()

            if y_pred is None:
                y_pred = y_class
                y_true = batch[1]
                if len(batch) > 2:
                    seiz_type = batch[2]
            else:
                y_pred = np.append(y_pred, y_class, axis = 0)
                y_true = np.append(y_true, batch[1], axis = 0)
                if len(batch) > 2:
                    seiz_type = np.append(seiz_type, batch[2], axis = 0)
        if return_seiz_type:
            return y_pred, y_true, seiz_type
        else:
            return y_pred, y_true








