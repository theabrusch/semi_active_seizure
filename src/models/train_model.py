import torch
from torch.utils.tensorboard import writer
from src.models import get_optim

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
            running_train_loss = 0
            running_val_loss = 0

            num_batch = 1
            print('Epoch', epoch + 1, 'out of', epochs)
            for batch in train_loader:
                x = batch[0].float().to(self.device)
                y = batch[1].long().to(self.device)
                self.optimizer.zero_grad()
                out = self.model(x)
                loss = self.loss_fn(out, y)
                loss.backward()
                self.optimizer.step()

                running_train_loss += loss.detach().cpu()
                num_batch += 1

            if self.scheduler is not None:
                self.scheduler.step()

            train_loss[epoch] = running_train_loss/num_batch
            if self.writer is not None:
                self.writer.add_scaler('Loss/train', train_loss[epoch], epoch)
            print('Training loss:', train_loss[epoch])

            num_batch = 1
            for batch in val_loader:
                x = batch[0].float().to(self.device)
                y = batch[1].long().to(self.device)
                out = self.model(x)
                loss = self.loss_fn(out, y)

                running_val_loss += loss.detach().cpu()
                num_batch += 1
            
            val_loss[epoch] = running_val_loss/num_batch
            if self.writer is not None:
                self.writer.add_scaler('Loss/val', val_loss[epoch], epoch)
            print('Validation loss:', val_loss[epoch])
        self.writer.flush()
        return train_loss, val_loss
                    

