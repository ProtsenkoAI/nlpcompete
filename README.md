### NLPCompete
NLPC is a project with NLP competitions solutions, and a pipeline
that's being used by all of them.

### Solutions of competitions
You can find .ipynb files in notebooks/

At the moment we have solutions of:
1. [SberQuAD](https://arxiv.org/abs/1912.09723)
2. [RuCoS](https://russiansuperglue.com/tasks/task_info/RuCoS)

### Main components of pipeline:
1. Plenty of useful interfaces based on clean architecture principles, for example:
   1. ModelWithTransformer - to easily load weights from pretrain file
   or HuggingFace servers, reset weights etc.
   2. PseudoLabeler: to easily predict test data and add pseudo labels
   to train set
   3. Container - to download data and pass them to Datasets
   4. Submitter - to make a submission in one line of code
   5. DataProcessor
   6. Others
2. Convenient components to train and use model:
   1. Trainer: supports checkpoints, evaluations, progress bars
   2. WeightsUpdater: update weights with a built-in GradScaler, amp etc.
   3. ModelManager: object composed of torch model and data processor
   makes fitting, submitting and any other operations agnostic to many
   pytorch features.
   4. Others

