import os

class PluginLoader(object):
    @staticmethod
    def _import(attr, name):
        print("Loading {} from {} plugin...".format(attr, name))
        module = __import__(name, globals(), locals(), [], 1)
        return getattr(module, attr)

    @staticmethod
    def get_trainer(name):
        return PluginLoader._import("Trainer", "Model_{0}".format(name))

    @staticmethod
    def get_available_models():
        models = ()
        for dir in next(os.walk( os.path.dirname(__file__) ))[1]:
            if dir[0:6].lower() == 'model_':
                models += (dir[6:],)
        return models

    @staticmethod
    def get_default_model():
        models = PluginLoader.get_available_models()
        return 'AE' if 'AE' in models else models[0]