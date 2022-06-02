import importlib

def find_option_using_name(model_name):
    """Import the module "model/[model_name]_model.py"."""
    model_file_name = "options." + model_name + "_options"
    modellib = importlib.import_module(model_file_name)
    model = None
    for name, cls in modellib.__dict__.items():
        if name.lower() == model_name.lower() :
            model = cls

    if model is None:
        print("In %s.py, there should be a subclass of BaseModel with class name that matches %s in lowercase." % (model_file_name, model_name))
        exit(0)

    return model


def create_option(opt):
    """Create a model given the option."""
    model = find_option_using_name(opt.model)
    instance = model()
    print("option [%s] was created" % type(instance).__name__)
    return instance