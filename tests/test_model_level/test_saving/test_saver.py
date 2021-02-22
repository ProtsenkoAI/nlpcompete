import unittest

from ...helpers import config, std_objects

config = config.TestsConfig()


class SharedObjects:
    def __init__(self):
        self.processor = std_objects.get_qa_processor()
        self.saver = std_objects.get_local_saver()


shared_objs = SharedObjects()


class TestLocalSaver(unittest.TestCase):
    # TODO: let the assistant class make @from_model() classmethod
    def test_save_then_load(self):
        model = std_objects.get_model()
        name = shared_objs.saver.save(model, shared_objs.processor)
        type_old_model = type(model)
        del model
        model_loaded, proc_loaded = shared_objs.saver.load(name)
        self.assertEqual(type(model_loaded), type_old_model)
        self.assertEqual(type(shared_objs.processor), type(proc_loaded))

    def test_save_multiple_model(self):
        model = std_objects.get_model()
        name1 = shared_objs.saver.save(model, shared_objs.processor)
        name2 = shared_objs.saver.save(model, shared_objs.processor)
        self.assertNotEqual(name1, name2, "two saves have save names, but should have different ones")
