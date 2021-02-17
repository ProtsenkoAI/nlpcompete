import numpy as np
import torch.optim as optim
import torch.nn as nn
import torch
from torch.cuda.amp import autocast, GradScaler
import torch.optim as optim
import transformers

from tqdm import tqdm


class Trainer:
    # TODO: move updating weights to separate component
    # TODO: manage answers out of scope (512 tokenss)
    def __init__(self, model_manager, validator, use_pseudo_labeling=False, epochs=2, 
                 use_early_stopping=True, lr=2e-5, eval_steps=200, stopping_patience=2,
                 weight_decay=1e-2, accum_iters=1, warmup=0, lr_end=1e-7, max_step=3000,
                 use_amp=False):
        self.use_pseudo_labeling = use_pseudo_labeling
        self.label_colname = "labels"
        self.epochs = epochs
        self.eval_steps = eval_steps
        self.validator = validator
        self.use_early_stopping = use_early_stopping
        self.accum_iters = accum_iters
        self.stopping_patience = stopping_patience
        self.warmup = warmup
        self.lr_end = lr_end
        self.max_step = max_step
        self.use_amp = use_amp

        self.eval_vals = []
        self.step_nb = 0

        self.manager = model_manager

        self._init_fitting_instruments(lr, weight_decay)

    def _init_fitting_instruments(self, lr, weight_decay):
        self.optimizer = optim.AdamW(self.manager.get_model().parameters(), lr=lr, weight_decay=weight_decay)
        self.criterion = nn.CrossEntropyLoss()
        if self.use_amp:
            self.scaler = GradScaler() 

    def _init_scheduler(self, train):
        total_steps = len(train) // self.accum_iters * self.epochs
        return transformers.get_polynomial_decay_schedule_with_warmup(optimizer=self.optimizer, 
                                                            num_warmup_steps=self.warmup, 
                                                            num_training_steps=total_steps,
                                                            lr_end=self.lr_end)

    def fit(self, train, test):
        # TODO: reset weights when starting training
        lr_scheduler = self._init_scheduler(train)
        # self.model.reset_weights(self.device)
        for epoch in range(self.epochs):
            if self.early_stopping():
                break
            print(f"Epoch: {epoch}")
            self.train_one_epoch(train, test, lr_scheduler)

        return self.model

    def train_one_epoch(self, train, test, lr_scheduler):
        self.manager.get_model().train()
        losses = []
        for batch in tqdm(train, desc="batch in train"):
            if self.early_stopping():
                break
            loss_val = self.train_one_step(batch, lr_scheduler)
            losses.append(loss_val)
            self.step_nb += 1

            if self.step_nb % self.eval_steps == 0:
                print(f"Step: {self.step_nb}")
                print(f"Mean loss: {np.mean(losses)}")
                losses = []
                eval_value = self.validator.eval(self.manager, test)
                self.eval_vals.append(eval_value)

    def train_one_step(self, batch, lr_scheduler): # or split to x and y?
        inputs, labels = batch
        print("batch before processing", inputs, labels)
        # start_labels, end_labels = self.manager.preproc_labels(labels)
        
        with autocast(enabled=self.use_amp):
            # start_probs, end_probs = self.manager.preproc_forward(inputs)
            preds, labels = self.manager.preproc_forward_labeled(inputs, labels)
            start_labels, end_labels = labels
            start_probs, end_probs = preds
            print("preds, labels", preds, labels)
            loss_start = self.criterion(start_probs, start_labels)
            loss_end = self.criterion(end_probs, end_labels)
            loss = (loss_start + loss_end) / 2
        loss = loss / self.accum_iters
        self._backward_loss(loss)

        if (self.step_nb + 1) % self.accum_iters == 0:
            self._optimizer_step()
            lr_scheduler.step()
            self.optimizer.zero_grad()

        return loss.item()

    def _backward_loss(self, loss):
        if self.use_amp:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

    def _optimizer_step(self):
        if not self.use_amp:
            self.optimizer.step()
        else:
            self.scaler.step(self.optimizer)
            self.scaler.update()

    def pseudo_labeling(self, ):
        raise NotImplementedError

    def early_stopping(self):
        if self.step_nb >= self.max_step:
            return True

        steps_passed = len(self.eval_vals)
        enough_steps_passed = steps_passed > self.stopping_patience

        if self.use_early_stopping and enough_steps_passed:
            best_result_step = np.argmax(self.eval_vals)
            if (steps_passed - best_result_step - 1) >= self.stopping_patience:
                print("EARLY STOPPING", "eval_vals", self.eval_vals, "best_result_step", best_result_step)
                return True
        return False

    def get_eval_vals(self):
        return self.eval_vals
