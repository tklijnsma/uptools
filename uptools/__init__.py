import logging
def setup_logger():
    fmt = logging.Formatter(
        fmt="\033[33m[uptools|%(levelname)8s|%(asctime)s|%(module)s]:\033[0m %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.StreamHandler()
    handler.setFormatter(fmt)
    logger = logging.getLogger("cernsso")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger
logger = setup_logger()


try:
    import awkward as ak
except ImportError:
    try:
        import awkward0 as ak
    except ImportError:
        try:
            import awkward1 as ak
        except ImportError:
            logger.error('Need some version of awkward-arrays installed!')
            raise

try:
    import uproot3 as uproot
except ImportError:
    try:
        import uproot
    except ImportError:
        logger.error('Need some version of uproot installed!')
        raise


UPROOT_VERSION = int(uproot.__version__.split('.',1)[0])
AK_VERSION = int(ak.__version__.split('.',1)[0])


def debug(flag=True):
    logger.setLevel(logging.DEBUG if flag else logging.INFO)


def _iter_trees(tdir, prefix=""):
    """
    Takes a ROOTDirectory-like object, and yields trees
    """
    for key, value in tdir.items():
        try:
            key = key.decode()
        except AttributeError:
            pass
        key = key.split(";")[0]
        if hasattr(value, "numentries") or hasattr(value, "num_entries"):
            yield prefix + key, value
        elif hasattr(value, "items"):
            yield from _iter_trees(value, prefix=prefix + key + "/")


def find_tree(tdir):
    """
    Takes a ROOTDirectory-like object, and finds a tree or raises an Exception
    """
    if isinstance(tdir, str):
        tdir = uproot.open(tdir)
    for path, tree in _iter_trees(tdir):
        logger.info("Using tree %s", path)
        return path, tree
    else:
        raise Exception("Could not find any tree-like object in {}".format(tdir))


def get_event(arrays, i=0):
    return {k: v[i] for k, v in arrays.items()}


def numentries(arrays):
    for k, v in arrays.items():
        return len(v)


def numentries_rootfile(rootfile, treepath=None):
    if treepath is None:
        treepath, tree = find_tree(rootfile)
    else:
        tree = uproot.open(rootfile)[treepath]
    return tree.numentries


def format_rootfiles(rootfiles):
    try:
        if rootfiles.endswith(".root"):
            if ':' in rootfiles and "*" in rootfiles:
                try:
                    import seutils
                except ImportError:
                    logger.error('Need seutils installed for remote wildcards support: pip install seutils')
                    raise
                rootfiles = seutils.ls_wildcard(rootfiles)
            elif '*' in rootfiles:
                import glob
                rootfiles = glob.glob(rootfiles)
            else:
                rootfiles = [rootfiles]
    except AttributeError:
        pass
    return rootfiles


def _decode_keys(arrays):
    try:
        arrays = { k.decode() : v for k, v in arrays.items() }
    except AttributeError:
        pass
    return arrays


def _convert_high_level_array(arrays):
    try:
        if isinstance(arrays, ak.highlevel.Array):
            arrays = { k : arrays[k] for k in arrays.fields }
    except AttributeError:
        pass
    return arrays


def iter_arrays(rootfiles, nmax=None, treepath=None, **kwargs):
    """
    Yields arrays from (list of) rootfiles.
    Up to a maximum of `nmax` entries are yielded in total.
    If `treepath` is None, a tree will be automatically searched for
    """
    do_decode = kwargs.pop('decode', False)
    ntodo = nmax
    for rootfile in format_rootfiles(rootfiles):
        f = uproot.open(rootfile)
        if treepath is None:
            path, t = find_tree(rootfile)
            treepath = path.encode()
        else:
            t = f[treepath]
        if nmax is None:
            for arrays in t.iterate(**kwargs):
                if do_decode: arrays = _decode_keys(arrays)
                yield _convert_high_level_array(arrays)
        else:
            kwargs['entrystop' if UPROOT_VERSION < 4 else 'entry_stop'] = ntodo
            for arrays in t.iterate(**kwargs):
                ntodo -= numentries(arrays)
                if do_decode: arrays = _decode_keys(arrays)
                yield _convert_high_level_array(arrays)
                if ntodo <= 0:
                    return


def iter_arrays_weighted(N, crosssections, rootfiles, **kwargs):
    """
    N: Total number of events to be iterated over
    crosssections: :ist of cross sections
    rootfiles: List of (list of) rootfiles, first dim equal to crosssections
    """
    if len(crosssections) != len(rootfiles):
        raise Exception(
            "Length of cross sections should be equal to number of passed rootfiles"
        )
    norm = sum(crosssections)
    ns_float = [xs / norm * N for xs in crosssections]
    ns = [round(n) for n in ns_float]
    logger.info("Requested %s, doing %s: %s", N, sum(ns), ns)
    for n, this_rootfiles in zip(ns, rootfiles):
        if n == 0:
            continue
        yield from iter_arrays(this_rootfiles, nmax=n, **kwargs)


def iter_events(rootfiles, **kwargs):
    for arrays in iter_arrays(rootfiles, **kwargs):
        for i in range(numentries(arrays)):
            yield get_event(arrays, i)


def get_event_rootfile(rootfile, i_event, **kwargs):
    """
    Calls the iter_events iterator until the desired event is reached
    """
    for i, event in enumerate(iter_events(rootfile, **kwargs)):
        if i == i_event:
            return event
