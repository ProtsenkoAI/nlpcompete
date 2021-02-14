import sys, os


def add_parent_dir_to_path():
    curr_dir = sys.path[0]
    parent_dir = os.path.dirname(curr_dir)
    sys.path.insert(0, parent_dir)



add_parent_dir_to_path()
