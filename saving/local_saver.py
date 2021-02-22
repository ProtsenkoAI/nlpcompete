class LocalSaver:
    def __init__(self, save_dir="./saved_models"):
        self.save_dir = save_dir

    def save(self):
        raise NotImplementedError
