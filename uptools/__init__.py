import logging

import awkward as ak
import seutils
import uproot3 as uproot
import uproot_methods


def setup_logger():
    fmt = logging.Formatter(
        fmt="\033[33m[cernsso|%(levelname)8s|%(asctime)s|%(module)s]:\033[0m %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler = logging.StreamHandler()
    handler.setFormatter(fmt)
    logger = logging.getLogger("cernsso")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger


logger = setup_logger()


def debug(flag=True):
    logger.setLevel(logging.DEBUG if flag else logging.INFO)


def _iter_trees(tdir, prefix=""):
    """
    Takes a ROOTDirectory-like object, and yields trees
    """
    for key, value in tdir.items():
        key = key.decode().split(";")[0]
        if hasattr(value, "numbranches"):
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
            if seutils.path.has_protocol(rootfiles) and "*" in rootfiles:
                rootfiles = seutils.ls_wildcard(rootfiles)
            else:
                rootfiles = [rootfiles]
    except AttributeError:
        pass
    return rootfiles


def iter_arrays(rootfiles, nmax=None, treepath=None, **kwargs):
    """
    Yields arrays from (list of) rootfiles.
    Up to a maximum of `nmax` entries are yielded in total.
    If `treepath` is None, a tree will be automatically searched for
    """
    ntodo = nmax
    for rootfile in format_rootfiles(rootfiles):
        f = uproot.open(rootfile)
        if treepath is None:
            path, t = find_tree(rootfile)
            treepath = path.encode()
        else:
            t = f[treepath]
        if not (nmax):
            yield from t.iterate(entrystop=nmax, **kwargs)
        else:
            for arrays in t.iterate(entrystop=ntodo, **kwargs):
                ntodo -= numentries(arrays)
                yield arrays
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


class Bunch:
    """
    Wrapper around a collection of branches.
    Primary use case is `Bunch[selection]`, where the selection is applied
    on all contained branches of the arrays object.
    """

    @classmethod
    def empty(cls, branches):
        """Initializes a bunch with the branches defined, but no arrays yet"""
        inst = cls()
        inst.arrays = {b: ak.JaggedArray([], [], []) for b in branches}
        return inst

    def concatenate(self, arrays):
        """Adds arrays to the existing arrays in the bunch"""
        for b in self.arrays.keys():
            self.arrays[b] = ak.concatenate((self.arrays[b], arrays[b]))

    @classmethod
    def from_branches(cls, arrays, branches):
        """
        Takes an arrays (dict-like) and a list (or dict) of brances,
        If branches is dict-like, the keys are used as accessors rather than the branch names.
        """
        inst = cls()
        if isinstance(branches, dict):
            inst.arrays = {k: arrays[v] for k, v in branches.items()}
        else:
            inst.arrays = {b: arrays[b] for b in branches}
        return inst

    def set_branch(self, key, array):
        self.arrays[key] = array

    def set_branches(self, **kwargs):
        self.arrays.update(**kwargs)

    def __getitem__(self, where):
        """Selection mechanism"""
        new = self.__class__()
        new.arrays = {k: v[where] for k, v in self.arrays.items()}
        return new

    def __getattr__(self, key):
        """Allow .attribute access to the branches"""
        if key in self.arrays:
            return self.arrays[key]
        else:
            try:
                ekey = key.encode()
                if ekey in self.arrays:
                    return self.arrays[ekey]
            except Exception:
                pass
        return super(Bunch, self).__getattr__(key)

    def __len__(self):
        for k, v in self.arrays.items():
            return len(v)

    def flatten(self):
        new = self.__class__()
        new.arrays = {k: v.flatten() for k, v in self.arrays.items()}
        return new

    def unflatten(self, counts):
        new = self.__class__()
        new.arrays = {}
        new.arrays = {
            k: ak.JaggedArray.fromcounts(counts, v) for k, v in self.arrays.items()
        }
        return new


class Vectors(Bunch):
    """
    Wrapper around a collection of pts, etas, phis, and energies
    """

    # The default branch postfixes to be searched for
    postfixes = {
        "pt": b"_pt",
        "eta": b"_eta",
        "phi": b"_phi",
        "energy": b"_energy",
        "mass": b"_mass",
    }

    @classmethod
    def from_prefix(cls, prefix, arrays, branches=None):
        # Default branches for vectors
        branch_map = {
            "pt": prefix + cls.postfixes["pt"],
            "eta": prefix + cls.postfixes["eta"],
            "phi": prefix + cls.postfixes["phi"],
            "energy": prefix + cls.postfixes["energy"],
            "mass": prefix + cls.postfixes["mass"],
        }
        # Add manual branches if there are any
        if branches:
            # First make it a list of it's a single bytes object
            if isinstance(branches, bytes):
                branches = [branches]
            # Then, for each branch, try both a prefixed and normal version
            for b in branches:
                b_str = b.decode()
                prefixed_b = prefix + b"_" + b
                prefixed_b_str = prefixed_b.decode()
                if b_str in branch_map or prefixed_b_str in branch_map:
                    # Already in map
                    continue
                elif b in arrays:
                    branch_map[b_str] = b
                elif prefixed_b in arrays:
                    branch_map[b_str] = prefixed_b
                else:
                    raise Exception("Could not find a branch for {}".format(b))
        inst = cls.from_branches(arrays, branch_map)
        inst.prefix = prefix
        return inst

    def __repr__(self):
        return super(Vectors, self).__repr__().replace("object", self.prefix.decode())

    def iter_vectors(self):
        flat = self.flatten()
        for i in range(len(flat)):
            yield uproot_methods.TLorentzVector.from_ptetaphie(
                flat.pt[i], flat.eta[i], flat.phi[i], flat.energy[i]
            )

    def as_vectors(self):
        return list(self.iter_vectors())

    def __iter__(self):
        return self.iter_vectors()


class FourVectorArray:
    def __init__(self, pt, eta, phi, energy):
        self.pt = pt
        self.eta = eta
        self.phi = phi
        self.energy = energy

    @property
    def px(self):
        import numpy as np

        return self.pt * np.cos(self.phi)

    @property
    def py(self):
        import numpy as np

        return self.pt * np.sin(self.phi)

    @property
    def pz(self):
        import numpy as np

        return self.pt * np.sinh(self.eta)

    @property
    def rapidity(self):
        import numpy as np

        return 0.5 * np.log((self.energy + self.pz) / (self.energy - self.pz))

    @property
    def mass2(self):
        return self.energy ** 2 - (self.px ** 2 + self.py ** 2 + self.pz ** 2)

    @property
    def mass(self):
        import numpy as np

        return np.sqrt(self.mass2)
