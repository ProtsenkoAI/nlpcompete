from torch import optim
from torch import nn
from torch.cuda import amp
import transformers
from time import time


class QAWeightsUpdater:
    def __init__(self, lr=2e-5, weight_decay=1e-2, accum_iters=1, warmup=0, lr_end=1e-7,
                 use_amp=False, ):

        self.lr = lr
        self.weight_decay = weight_decay
        self.accum_iters = accum_iters
        self.warmup = warmup
        self.lr_end = lr_end
        self.use_amp = use_amp

        self.criterion = nn.CrossEntropyLoss()
        if self.use_amp:
            self.scaler = amp.GradScaler()

        # should be inited by calling prepare_for_fit()
        self.optimizer = None
        self.lr_scheduler = None

        self.step_idx = 0

    def prepare_for_fit(self, model_manager, nb_train_steps):
        self.optimizer = optim.AdamW(model_manager.get_model().parameters(),
                                     lr=self.lr, weight_decay=self.weight_decay)

        total_steps = nb_train_steps // self.accum_iters
        self.lr_scheduler = transformers.get_polynomial_decay_schedule_with_warmup(optimizer=self.optimizer,
                                                                                   num_warmup_steps=self.warmup,
                                                                                   num_training_steps=total_steps,
                                                                                   lr_end=self.lr_end)

    def fit_with_batch(self, manager, batch):
        inputs, labels = batch
        loss = self._calc_loss(manager, inputs, labels)
        self._backward_loss(loss)

        if (self.step_idx + 1) % self.accum_iters == 0:
            self._optim_step()

        self.step_idx += 1
        return loss.item()

    def _calc_loss(self, manager, inputs, labels):
        with amp.autocast(enabled=self.use_amp):
            try:
                preds, labels = manager.preproc_forward(inputs, labels)
            except Exception as e:
                raise e
            start_labels, end_labels = labels
            start_probs, end_probs = preds
            loss_start = self.criterion(start_probs, start_labels)
            loss_end = self.criterion(end_probs, end_labels)
            loss = (loss_start + loss_end) / 2
        loss = loss / self.accum_iters
        return loss

    def _backward_loss(self, loss):
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def _optim_step(self):
        if self.use_amp:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()
        self.lr_scheduler.step()
        self.optimizer.zero_grad()
