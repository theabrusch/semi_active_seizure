from pyexpat import features
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
    def __init__(self, model, optimizer, loss_fn, val_loss = None,
                 choose_best = True, writer = None, scheduler = None):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        if self.loss_fn is not None:
            self.loss_fn.to(self.device)
        self.writer = writer
        self.choose_best = choose_best
        if val_loss is not None:
            self.val_loss = val_loss.to(self.device)
        else:
            self.val_loss = None

    def train(self,
              train_loader,
              val_loader, 
              safe_best_model = False,
              test_loader = None,
              trial = None,
              job_name = None,
              early_stopping = False,
              transfer_subj = None,
              epochs = 10):
        '''
        Train model
        '''

        train_loss = torch.zeros(epochs)
        val_loss = torch.zeros(epochs)
        f1_scores = torch.zeros(epochs)

        if safe_best_model or self.choose_best:    
            if safe_best_model:
                checkpoint_path = 'models/final_models/' + str(datetime.now())  + str(job_name)
            else:
                checkpoint_path = 'models/checkpoints/' + str(datetime.now())  + str(job_name)
            p = Path(checkpoint_path)
            p.mkdir(parents=True, exist_ok=True)

        f1_val_old = 0 
        sensspec_old = 0
        sens_old = 0
        spec_old = 0
        
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
                if trial is None and transfer_subj is None:
                    run = ''
                elif trial is not None:
                    run = str(trial.number)
                elif transfer_subj is not None:
                    run = transfer_subj

                self.writer.add_scalar('train/loss'+run, train_loss[epoch], epoch)
            print('Training loss:', train_loss[epoch])

            # Compute validation loss and metrics
            num_batch = 1
            self.model.eval()
            for batch in val_loader:
                x = batch[0].float().to(self.device)
                y = batch[1].long().to(self.device)
                out = self.model(x)
                if self.val_loss is None:
                    loss = self.loss_fn(out, y)
                else:
                    loss = self.val_loss(out, y)

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
            f1_val_new = f1_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred)
            cm = confusion_matrix(y_true, y_pred, normalize = 'true')
            tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]

            f1_scores[epoch] = f1_val_new
            f1_val = (f1_val_new + f1_val_old)/2
            f1_val_old = f1_val_new

            # harmonic mean between sens and spec
            sensspec_new = 2*sens*spec/(sens+spec)
            sensspec = (sensspec_new + sensspec_old)/2
            sensspec_old = sensspec_new

            if trial is not None:
                trial.report(f1_val, epoch)
                if epoch > 9:
                    if trial.should_prune():
                        trial.set_user_attr('sens', sens)
                        trial.set_user_attr('spec', spec)
                        trial.set_user_attr('prec', spec)
                        trial.set_user_attr('sensspec', spec)
                        raise optuna.exceptions.TrialPruned()

            if self.writer is not None:
                if trial is None and transfer_subj is None:
                    run = ''
                elif trial is not None:
                    run = str(trial.number)
                elif transfer_subj is not None:
                    run = transfer_subj

                self.writer.add_scalar('val/sens'+run, sens, epoch)
                self.writer.add_scalar('val/spec'+run, spec, epoch)
                self.writer.add_scalar('val/f1'+run, f1_val_new, epoch)
                self.writer.add_scalar('val/sensspec'+run, sensspec_new, epoch)
                self.writer.add_scalar('val/precision'+run, prec, epoch)
                self.writer.add_scalar('val_raw/true_pos'+run, tp, epoch)
                self.writer.add_scalar('val_raw/false_neg'+run, fn, epoch)
                self.writer.add_scalar('val_raw/false_pos'+run, fp, epoch)
                self.writer.add_scalar('val_raw/true_neg'+run, tn, epoch)

            val_loss[epoch] = running_val_loss/num_batch
            if self.writer is not None:
                if trial is None:
                    run = ''
                else:
                    run = str(trial.number)
                self.writer.add_scalar('val/loss' + run, val_loss[epoch], epoch)
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
                    if trial is None:
                        run = 'test'
                    else: 
                        run = 'test_' + str(trial.number)

                    self.writer.add_scalar(run+'/sens', sens, epoch)
                    self.writer.add_scalar(run+'/spec', spec, epoch)
                    self.writer.add_scalar(run+'/f1', f1, epoch)
                    self.writer.add_scalar(run+'/precision', prec, epoch)
                    self.writer.add_scalar(run+'/loss', test_loss, epoch)

            epoch_time = (datetime.now()-time).total_seconds()
            print('Epoch time', epoch_time)
            if self.writer is not None:
                if trial is None and transfer_subj is None:
                    run = ''
                elif trial is not None:
                    run = str(trial.number)
                elif transfer_subj is not None:
                    run = transfer_subj
                self.writer.add_scalar('Loss/epoch_time'+run, epoch_time, epoch)

            if early_stopping and epoch > 10:
                if np.mean(abs(np.diff(train_loss[(epoch-4):(epoch+1)]))) < 5e-5:
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
            return sensspec, sens, spec, f1_val, prec
        else:
            return train_loss, val_loss
    
    def eval(self, data_loader, return_probability = True, return_seiz_type = False):
        y_pred = None

        self.model.eval()
        i = 1
        for batch in data_loader:
            print('Batch', i, 'out of',  data_loader.__len__())
            i+=1
            x = batch[0].float().to(self.device)
            out = self.model(x)
            y_class = torch.argmax(out, axis = -1).cpu().numpy()
            proba = out[:,1].cpu().numpy()

            if y_pred is None:
                y_pred = y_class
                y_true = batch[1]
                probability = proba
                if len(batch) > 2:
                    seiz_type = batch[2]
            else:
                y_pred = np.append(y_pred, y_class, axis = 0)
                y_true = np.append(y_true, batch[1], axis = 0)
                probability = np.append(probability, proba, axis = 0)
                if len(batch) > 2:
                    seiz_type = np.append(seiz_type, batch[2], axis = 0)
                    
        if return_seiz_type:
            if return_probability:
                return y_pred, y_true, seiz_type, probability
            else:
                return y_pred, y_true, seiz_type
        else:
            if return_probability:
                return y_pred, y_true, probability
            else:
                return y_pred, y_true
        


class model_train_ssltf():
    '''
    Class for training pytorch model using 
    semisupervised transfer learning.
    '''
    def __init__(self, target_model, source_model, optimizer, loss_fn, val_loss = None,
                 choose_best = True, writer = None, scheduler = None):
        self.target_model = target_model
        self.source_model = source_model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.target_model.to(self.device)
        self.source_model.to(self.device)
        self.loss_fn.to(self.device)
        self.writer = writer
        self.choose_best = choose_best
        if val_loss is not None:
            self.val_loss = val_loss.to(self.device)

    def train_transfer(self,
                        train_loader,
                        val_loader, 
                        safe_best_model = False,
                        transfer_subj = None,
                        tol = 0,
                        epochs = 10):
        '''
        Train model
        '''

        train_loss = torch.zeros(epochs)
        val_loss = torch.zeros(epochs)

        for epoch in range(epochs):
            running_train_loss = 0
            running_val_loss = 0

            num_batch = 1
            #print('Epoch', epoch + 1, 'out of', epochs)
            self.target_model.train()
            self.source_model.eval()
            for batch in train_loader:
                x = batch[0].float().to(self.device)
                y = batch[1].long().to(self.device)
                self.optimizer.zero_grad()
                # get output of target model to be trained
                out_target, features_target = self.target_model(x, return_features = True)

                # get output of source model
                with torch.no_grad():
                    out_source, features_source = self.source_model(x, return_features = True)

                loss = self.loss_fn(out_target, features_target, out_source, features_source, y)
                loss.backward()
                self.optimizer.step()

                running_train_loss += loss.detach().cpu()
                num_batch += 1
                
            if self.scheduler is not None:
                self.scheduler.step()

            train_loss[epoch] = running_train_loss/num_batch
            if self.writer is not None: 
                run = transfer_subj
                self.writer.add_scalar('train/loss'+run, train_loss[epoch], epoch)
            print('Training loss:', train_loss[epoch])

            # Compute validation loss and metrics
            num_batch = 1
            self.target_model.eval()
            for batch in val_loader:
                x = batch[0].float().to(self.device)
                y = batch[1].long().to(self.device)
                out = self.target_model(x)
                if self.val_loss is None:
                    loss = self.loss_fn(out, y)
                else:
                    loss = self.val_loss(out, y)

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

            # harmonic mean between sens and spec
            sensspec_new = 2*sens*spec/(sens+spec)
            val_loss[epoch] = running_val_loss/num_batch

            if self.writer is not None:
                run = transfer_subj

                self.writer.add_scalar('val/sensspec'+run, sensspec_new, epoch)
                self.writer.add_scalar('val/loss' + run, val_loss[epoch], epoch)

            print('Validation loss:', val_loss[epoch])

            if epoch > 10:
                if np.mean(abs(np.diff(train_loss[(epoch-2):(epoch+1)]))) <= tol:
                    break

        if safe_best_model:
            checkpoint_path = 'models/checkpoints/' + str(datetime.now())  + transfer_subj
            p = Path(checkpoint_path)
            p.mkdir(parents=True, exist_ok=True)
            model_check = checkpoint_path + '/final_model' + '.pt'
            torch.save({'model_state_dict': self.model.state_dict()},
                        model_check)
        if self.writer is not None:
            self.writer.flush()

        return train_loss, val_loss
    
    def train_transfer_ssl(self,
                            train_loader,
                            unlabeled_loader,
                            val_loader, 
                            safe_best_model = False,
                            transfer_subj = None,
                            tol = 0,
                            epochs = 10):
        '''
        Train model
        '''

        train_loss = torch.zeros(epochs)
        val_loss = torch.zeros(epochs)

        for epoch in range(epochs):
            running_train_loss = 0
            running_val_loss = 0

            num_batch = 1
            #print('Epoch', epoch + 1, 'out of', epochs)
            self.target_model.train()
            self.source_model.eval()
            for (batch, unlab_batch) in zip(train_loader, unlabeled_loader):
                x = batch[0].float().to(self.device)
                y = batch[1].long().to(self.device)
                self.optimizer.zero_grad()
                # get output of target model to be trained
                out_target, features_target = self.target_model(x, return_features = True)
                
                # get unlabeled output
                x_unlab = unlab_batch[0].float().to(self.device)
                out_target_unlab, features_target_unlab = self.target_model(x_unlab, return_features = True)

                # get output of source model
                with torch.no_grad():
                    out_source, features_source = self.source_model(x, return_features = True)
                    out_source_unlab, features_source_unlab = self.source_model(x_unlab, return_features = True)

                # compute loss using predictions and features from target and source model
                # for both labeled and unlabeled batches
                loss = self.loss_fn(out_target_labeled = out_target, 
                                    features_target_labeled = features_target, 
                                    out_target_unlabeled = out_target_unlab,
                                    features_target_unlabeled = features_target_unlab,
                                    out_source_labeled = out_source,
                                    features_source_labeled = features_source,
                                    out_source_unlabeled = out_source_unlab,
                                    features_source_unlabeled = features_source_unlab,
                                    y_true = y)
                loss.backward()
                self.optimizer.step()

                running_train_loss += loss.detach().cpu()
                num_batch += 1
                
            if self.scheduler is not None:
                self.scheduler.step()

            train_loss[epoch] = running_train_loss/num_batch
            if self.writer is not None: 
                run = transfer_subj
                self.writer.add_scalar('train/loss'+run, train_loss[epoch], epoch)
            print('Training loss:', train_loss[epoch])

            # Compute validation loss and metrics
            num_batch = 1
            self.target_model.eval()
            for batch in val_loader:
                x = batch[0].float().to(self.device)
                y = batch[1].long().to(self.device)
                out = self.target_model(x)
                if self.val_loss is None:
                    loss = self.loss_fn(out, y)
                else:
                    loss = self.val_loss(out, y)

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

            # harmonic mean between sens and spec
            sensspec_new = 2*sens*spec/(sens+spec)
            val_loss[epoch] = running_val_loss/num_batch

            if self.writer is not None:
                run = transfer_subj

                self.writer.add_scalar('val/sensspec'+run, sensspec_new, epoch)
                self.writer.add_scalar('val/loss' + run, val_loss[epoch], epoch)

            print('Validation loss:', val_loss[epoch])

            if epoch > 10:
                if np.mean(abs(np.diff(train_loss[(epoch-2):(epoch+1)]))) <= tol:
                    break

        if safe_best_model:
            checkpoint_path = 'models/checkpoints/' + str(datetime.now())  + transfer_subj
            p = Path(checkpoint_path)
            p.mkdir(parents=True, exist_ok=True)
            model_check = checkpoint_path + '/final_model' + '.pt'
            torch.save({'model_state_dict': self.model.state_dict()},
                        model_check)
        if self.writer is not None:
            self.writer.flush()

        return train_loss, val_loss
    
    
    def eval(self, data_loader, return_seiz_type = False):
        y_pred = None

        self.target_model.eval()
        i = 1
        for batch in data_loader:
            #print('Batch', i, 'out of',  data_loader.__len__())
            i+=1
            x = batch[0].float().to(self.device)
            out = self.target_model(x)
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
















