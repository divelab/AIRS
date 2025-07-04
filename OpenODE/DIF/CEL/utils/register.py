r"""A kernel module that contains a global register for unified model, dataset, and OOD algorithms access.
"""


class Register(object):
    r"""
    Global register for unified model, dataset, and OOD algorithms access.
    """

    def __init__(self):
        self.launchers = dict()
        self.models = dict()
        self.datasets = dict()
        self.experiments = dict()

    def launcher_register(self, launcher_class):
        r"""
        Register for pipeline access.

        Args:
            launcher_class (class): pipeline class

        Returns (class):
            pipeline class

        """
        self.launchers[launcher_class.__name__] = launcher_class
        return launcher_class

    def model_register(self, model_class):
        r"""
        Register for model access.

        Args:
            model_class (class): model class

        Returns (class):
            model class

        """
        self.models[model_class.__name__] = model_class
        return model_class

    def dataset_register(self, dataset_class):
        r"""
        Register for dataset access.

        Args:
            dataset_class (class): dataset class

        Returns (class):
            dataset class

        """
        self.datasets[dataset_class.__name__] = dataset_class
        return dataset_class

    def experiment_register(self, exp_class):
        r"""
        Register for OOD algorithms access.

        Args:
            exp_class (class): OOD algorithms class

        Returns (class):
            OOD algorithms class

        """
        self.experiments[exp_class.__name__] = exp_class
        return exp_class


register = Register()  #: The ATTA register object used for accessing models, datasets and OOD algorithms.


