import math
import sh
import io
import numpy
import scipy
import scipy.optimize

def convert(x, y):
    instr = "{} {}\n".format(x, y)
    result = sh.proj(
        '+proj=tmerc',
        '+lat_0=33',
        '+k_0=0.9996',
        '+x_0=-99517',
        '+y_0=-4998115',
        _in=instr,
    ).stdout
    print(result)
    try:
        parsed = [float(e) for e in result.decode('UTF-8').strip().split('\t')]
    except ValueError:
        return numpy.array([math.inf, math.inf])
    print(x, y, parsed)
    return numpy.array(parsed)

def f(x):
    return math.hypot(*convert(*x))

x0 = [0, 0]
res = scipy.optimize.minimize(f, x0, method="Nelder-Mead")
print(res.x, f(res.x))
