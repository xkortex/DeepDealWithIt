import collections
import os
import time
from configparser import ConfigParser

from keras.callbacks import ModelCheckpoint, CSVLogger, TensorBoard
from keras.models import Sequential, load_model, save_model


def lazyparse(value):
    """
    This function will try very hard to coerce the value.
    If it can't, it'll cowardly give up without an exception and return the original value.
    Life is short, use fast-and-loose parsers.
    :param value:
    :type value: str
    :return: Coerced value, or original string
    >>> lazyparse('')
    ''
    >>> lazyparse('3')
    3
    >>> type(lazyparse('3'))
    <class 'int'>
    """
    value = value.lower()
    if value in ['yes', 'true', 'on', 'y']:
        return True
    if value in ['no', 'false', 'off', 'n']:
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value

class SaveableModel:
    """
    Basic Keras model boilerplate and file handler for saving weights, config files, and models
    """
    sec_model = 'model'
    sec_fit = 'fit'
    sections = [sec_model, sec_fit]
    _cfg_default = {sec: {} for sec in sections}

    def __init__(self, model_name=None, path_model='model.h5', path_weights='weights.h5', path_cfg='model.cfg',
                 path_log='model_logs/logs', verbose=False):
        if model_name is not None:
            model_name = os.path.abspath(model_name)
            basename = os.path.basename(model_name)
            dirname = os.path.dirname(model_name) + '/'
            path_model = dirname + basename + '.h5'
            path_weights = dirname + basename + '_w.h5'
            path_cfg = dirname + basename + '.cfg'
            path_log = dirname + basename + '_logs/'

            os.makedirs(dirname, exist_ok=True)
            os.makedirs(path_log, exist_ok=True)


        self.path_weights = path_weights
        self.path_model = path_model
        self.path_cfg = path_cfg
        self.path_log = path_log
        self.model = Sequential() # do we want to autoload?
        self._cfg = None
        # self._pm = None
        # self._pf = None
        self.verbose = verbose

    def autoload(self, verbose=False):
        """
        Check to see if all the requisite files exist aready. If they do, load from that model. Else, create a
        blank model from the defaults. Ignore how that is done right now.

        This should be called by any subclasses after their call to super().__init__()
        :return:
        """
        # todo: streamline creation and teardown

        if self.file_exists():
            if verbose: print('Loading from {}'.format(self.path_model))
            self.load()
        else:
            if verbose: print('No existing model data found. Initializing from default at {}'.format(self.path_model))
            self._cfg = {'model':{}, 'fit':{}}
            self.new_model(verbose=verbose)
            self.save(verbose=verbose)

    def new_model(self, *args, **kwargs):
        pass

    def load(self):
        """
        Load model (which contains the architecture, weights, loss, optimizer, and state) and the config file
        :return:
        """

        self.load_cfg()
        self.load_model()

    def save(self, verbose=False):
        self.save_cfg(verbose=verbose)
        self.save_model(verbose=verbose)

    def load_model(self, path_model=None, verbose=False):
        path_model = self.path_model if path_model is None else path_model
        self.model = load_model(path_model)

    def save_model(self, path_model=None, verbose=False):
        path_model = self.path_model if path_model is None else path_model
        save_model(self.model, path_model)
        if verbose: print('Saved model: ', path_model)

    def load_cfg(self, path_cfg=None, verbose=False):
        path_cfg = self.path_cfg if path_cfg is None else path_cfg
        cfgParser = ConfigParser()
        # if not os.path.exists(path_cfg):
        cfgParser.read(path_cfg)
        cfg = cfgParser #[self.sec_model]
        # todo: this section needs a lot of polish. but shipped > perfect
        self._cfg = {
            sec: {key: lazyparse(item) for key, item in cfg[sec].items()}
                for sec in cfg.sections()
        }
        _pm = collections.namedtuple('model', cfg['model'])
        _pf = collections.namedtuple('fit', cfg['fit'])
        self.pm = _pm(**self._cfg['model'])
        self.pf = _pf(**self._cfg['fit'])
        if verbose: print('Loaded config: ', path_cfg)

    def save_cfg(self, path_cfg=None, verbose=False):
        path_cfg = self.path_cfg if path_cfg is None else path_cfg
        cfgParser = ConfigParser()
        cfgParser.read_dict(self.cfg)
        with open(path_cfg, 'w') as configfile:
            cfgParser.write(configfile)
        if verbose: print('Saved config: {} \n{}'.format(path_cfg, self.cfg ))

    def make_default_cfg(self, path_cfg=None, verbose=False):
        path_cfg = self.path_cfg if path_cfg is None else path_cfg
        self._cfg = self._cfg_default
        self.save_cfg(path_cfg)

    def load_weights(self, path_weights=None, verbose=False):
        path_weights = self.path_weights if path_weights is None else path_weights
        self.model.load_weights(path_weights)

    def save_weights(self, path_weights=None, verbose=False):
        path_weights = self.path_weights if path_weights is None else path_weights
        self.model.save_weights(path_weights)

    def file_exists(self):
        return os.path.exists(self.path_cfg) and os.path.exists(self.path_model) #and os.path.exists(self.path_weights)

    def update_cfg(self, dictlike):
        self._cfg.update(dictlike)

    def setup_callbacks(self, checkpoint_verbose=1, save_best_only=False, path_weights=None):
        path_weights = self.path_weights if path_weights is None else path_weights
        now = time.strftime("%Y%m%d-%H%M%S")
        checkpointer = ModelCheckpoint(monitor='val_acc', filepath=path_weights, verbose=checkpoint_verbose,
                                       save_best_only=save_best_only)
        csvlogger = CSVLogger(self.path_log + now + '.csv', append=True)  # todo point this to proper location
        tensorboard = TensorBoard()
        self.callbacks = [checkpointer, csvlogger, tensorboard]

    @property
    def cfg(self): return self._cfg

    # @cfg.setter
    # def cfg(self): self

