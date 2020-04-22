from abc import ABCMeta, abstractmethod
from inspect import Parameter, Signature, signature

from evalml.exceptions import MethodPropertyNotFoundError
from evalml.pipelines.components.validation_error import ValidationError
from evalml.utils import Logger, classproperty, get_random_state

logger = Logger()


class ComponentValidator(ABCMeta):
    _REQUIRED_INIT_ARGS = ['random_state']
    _INVALID_INIT_ARGS = ['component_obj']

    def __new__(cls, name, bases, dct):
        assert '__init__' in dct
        dct['_default_parameters'] = cls._get_default_parameters(name, bases, dct)
        if '__new__' in dct:
            raise Exception('Component class may not define __new__')
        dct['__new__'] = cls._make_new(name, bases, dct)


    @classmethod
    def _validate_default_parameter(metacls, param_name, param_obj, current_cls_name):
        """Given a class (current_cls), """
        if param_name in ['self']:
            return False
        if param_obj.kind in (Parameter.POSITIONAL_ONLY, Parameter.KEYWORD_ONLY):
            raise ValidationError(("Component '{}' __init__ uses non-keyword argument '{}', which is not " +
                                   "supported").format(current_cls_name, param_name))
        if param_obj.kind in (Parameter.VAR_KEYWORD, Parameter.VAR_POSITIONAL):
            raise ValidationError(("Component '{}' __init__ uses *args or **kwargs, which is not " +
                                   "supported").format(current_cls_name))
        if param_name in metacls._INVALID_INIT_ARGS:
            raise ValidationError(("Component '{}' __init__ should not provide argument '{}'").format(current_cls_name, param_name))
        if param_obj.default == Signature.empty:
            raise ValidationError(("Component '{}' __init__ has no default value for argument '{}'").format(current_cls_name, param_name))
        return True

    @classmethod
    def _get_default_parameters(metacls, name, bases, dct):
        """Introspect on subclass __init__ method to determine default values of each argument.

        Raises exception if subclass __init__ uses any args other than standard keyword args.

        Returns:
            dict: map from parameter name to default value
        """
        if name == 'object':
            return {}
        if len(bases) > 1:
            raise Exception('Components may not use multiple inheritance')
        superclass_defaults = {}
        if len(bases) == 1:
            mro = inspect.getmro(bases[0])
            if len(mro) >= 2:
                new_base = mro[1]
                superclass_defaults = metacls._get_default_parameters(new_base.__qualname__,
                                                                      new_base,
                                                                      new_base.__dict__)

        # get the default parameters from the init method of this class
        sig = signature(dct['__init__'])

        def validate(pair):
            param_name, param_obj = pair
            return metacls._validate_default_parameter(param_name, param_obj, name)
        valid_pairs = filter(lambda pair: validate(pair), sig.parameters.items())

        def get_value(pair):
            param_name, param_obj = pair
            return (param_name, param_obj.default)
        class_defaults = dict(map(lambda pair: get_value(pair), valid_pairs))

        missing_init_args = set(metacls._REQUIRED_INIT_ARGS) - class_defaults.keys()
        if len(missing_init_args):
            name = current_cls.name
            raise ValidationError("Component '{}' __init__ missing values for required parameters: '{}'".format(name, str(missing_init_args)))

        superclass_defaults.update(class_defaults)
        return superclass_defaults

    @classmethod
    def _make_new(metacls, new_class_name, new_class_bases, new_class_dct):
        def __new__(cls, name, bases, dct):
            """Introspect on subclass __init__ method to determine the values saved as state.

            Raises exception if subclass __init__ uses any args other than standard keyword args.
            Also raises exception if parameters defined in subclass __init__ are different from those which
            were provided to ComponentBase.__init__.

            Returns:
                dict: map from parameter name to default value
            """
            sig = signature(self.__init__)
            defaults = self.default_parameters

            def validate(pair):
                param_name, param_obj = pair
                if not self._validate_default_parameter(param_name, param_obj):
                    return False
                if param_name not in self._REQUIRED_INIT_ARGS and not hasattr(self, param_name):
                    name = self.name
                    raise ValidationError(("Component '{}' __init__ has not saved state for parameter '{}'").format(name, param_name))
                return True
            valid_pairs = filter(lambda pair: validate(pair), sig.parameters.items())

            def get_value(pair):
                param_name, param_obj = pair
                return (param_name, getattr(self, param_name))
            values = dict(map(lambda pair: get_value(pair), valid_pairs))

            missing_subclass_init_args = set(self._REQUIRED_INIT_ARGS) - defaults.keys()
            if len(missing_subclass_init_args):
                name = self.name
                raise ValidationError("Component '{}' __init__ missing values for required parameters: '{}'".format(name, str(missing_subclass_init_args)))
            return values


class ComponentBase(metaclass=ComponentValidator):
    """The abstract base class for all evalml components.

    Please see Transformer and Estimator for examples of how to use this class.
    """

    def __init__(self, component_obj=None, random_state=0):
        if not hasattr(self, 'random_state'):
            self.random_state = get_random_state(random_state)
        self._component_obj = component_obj
        self._parameters = self._get_parameters()

    @property
    @classmethod
    @abstractmethod
    def name(cls):
        """Returns string name of this component"""
        return NotImplementedError("This component must have `name` as a class variable.")

    @property
    @classmethod
    @abstractmethod
    def model_family(cls):
        """Returns ModelFamily of this component"""
        return NotImplementedError("This component must have `model_family` as a class variable.")

    @property
    def parameters(self):
        return self._parameters

    @classproperty
    def default_parameters(cls):
        return cls._default_parameters

    _REQUIRED_INIT_ARGS = ['random_state']
    _INVALID_INIT_ARGS = ['component_obj']

    def _get_parameters(self):

    def fit(self, X, y=None):
        """Fits component to data

        Arguments:
            X (pd.DataFrame or np.array): the input training data of shape [n_samples, n_features]
            y (pd.Series, optional): the target training labels of length [n_samples]

        Returns:
            self
        """
        try:
            self._component_obj.fit(X, y)
            return self
        except AttributeError:
            raise MethodPropertyNotFoundError("Component requires a fit method or a component_obj that implements fit")

    def describe(self, print_name=False, return_dict=False):
        """Describe a component and its parameters

        Arguments:
            print_name(bool, optional): whether to print name of component
            return_dict(bool, optional): whether to return description as dictionary in the format {"name": name, "parameters": parameters}

        Returns:
            None or dict: prints and returns dictionary
        """
        if print_name:
            title = self.name
            logger.log_subtitle(title)
        for parameter in self.parameters:
            parameter_str = ("\t * {} : {}").format(parameter, self.parameters[parameter])
            logger.log(parameter_str)
        if return_dict:
            component_dict = {"name": self.name}
            component_dict.update({"parameters": self.parameters})
            return component_dict
