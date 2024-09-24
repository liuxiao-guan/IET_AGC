import math
import sys

inf = math.inf
nan = math.nan
string_classes = (str, bytes)
PY37 = sys.version_info[0] == 3 and sys.version_info[1] >= 7

def with_metaclass(meta: type, *bases) -> type:
    """Create a base class with a metaclass."""
    # This requires a bit of explanation: the basic idea is to make a dummy
    # metaclass for one level of class instantiation that replaces itself with
    # the actual metaclass.
    class metaclass(meta):  # type: ignore[misc, valid-type]

        def __new__(cls, name, this_bases, d):
            return meta(name, bases, d)

        @classmethod
        def __prepare__(cls, name, this_bases):
            return meta.__prepare__(name, bases)

    return type.__new__(metaclass, 'temporary_class', (), {})
