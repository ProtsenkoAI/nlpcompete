from tqdm.notebook import tqdm as notebook_tqdm
from tqdm import tqdm as cli_tqdm


def get_tqdm_obj(mode="notebook"):
    # TODO: abstract away from tqdm: make interface so can replace the library if needed
    if mode not in ["cli", "notebook"]:
        raise ValueError("Available mode values: cli, notebook")
    if mode == "cli":
        chosen_tqdm = cli_tqdm
    if mode == "notebook":
        chosen_tqdm = notebook_tqdm

    def tqdm_with_standard_params(*args, **kwargs):
        return chosen_tqdm(*args, **kwargs, mininterval=1)

    return tqdm_with_standard_params
