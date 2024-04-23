from script.utils.util import logger
from script.models.HMPTGN import HMPTGN



def load_model(args):
    if args.model in ['HMPTGN']:
        model = HMPTGN(args)
    else:
        raise Exception('pls define the model')
    logger.info('using model {} '.format(args.model))
    return model
