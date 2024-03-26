import numpy as np
import re
from checklist.perturb import process_ret

def perturb_date_ref(doc, meta=False, seed=None, n=10):
    """Changes the time reference

    Parameters
    ----------
    doc : spacy.token.Doc
        input
    meta : bool
        if True, will return list of (orig_t_ref, new_t_ref) as meta
    seed : int
        random seed
    n : int
        number of temporal locations to replace original locations with

    Returns
    -------
    list(str)
        if meta=True, returns (list(str), list(tuple))
        Strings with numbers replaced.

    """
    if seed is not None:
        np.random.seed(seed)
    dow = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
    months = ['january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december']
    c_dow = [d.capitalize() for d in dow]
    c_months = [m.capitalize() for m in months]
    t_location = [x.text for x in doc if x.text.lower() in dow + months]
    ret = []
    ret_m = []
    for x in t_location:
        #x = x.lower()
        sub_re = re.compile(r'\b%s\b' % x)
        print(sub_re)
        to_sub = c_months+c_dow
        ret.extend([sub_re.sub(n, doc.text) for n in to_sub])
        ret_m.extend([(x, n) for n in to_sub])
    print(ret)
    return process_ret(ret, ret_m=ret_m, n=n, meta=meta)