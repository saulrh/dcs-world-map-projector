# Several coordinates from the map, including the origin.
# 
# 24d 26m 47.09s N
# 56d 37m 37.74s E
# X -00191230
# Z +00037965
# 
# 25d 43m 15.80s N
# 58d 33m 49.07s E
# X -00049211
# Z +00232621
# 
# 27d 23m 01.51s N
# 57d 12m 01.30s E
# X +00134024
# Z +00095567
# 
# 30d 15m 55.10s N
# 51d 30m 02.08s E
# X +00466081
# Z -00453636
# 
# 26d 10m 18.55s N
# 56d 14m 30.96s E
# X +00000000
# Z +00000000
# 
# 25d 00m 00.00s N
# 51d 00m 00.00s E
# X -00116539
# Z -00530419

# X and Z appear to represent meters. If I drag the ruler tool across
# the map in the Mission Editor and convert that to meters, it's
# similar to if I take the difference between the coordinates at
# opposite sides of the map.
# 
# X appears to represent latitude primarily (could be some rotation
# involved!) and increases in the North direction. Yeah, a bit odd.
#
# Z appears to represent longitude and increases in the East
# direction. Again, a bit odd.
# 
# The game is DCS: World. It's a free game. We're looking at the
# Persian Gulf DLC map, but you could test your solution with the free
# Caucausus map theoretically (although they use different
# reprojections. To generate new coordinate pairs you open the Mission
# Editor, drag your mouse around, and check the Lat-Long listed in the
# bottom bar. You can use Left ALT+Y to cycle the coordinates in the
# bottom bar between different formats, including a few forms of
# Lat-Long, a MGRS (Military Grid Reference System), and DCS internal
# coordinates (X-Z).
# 
# The overall goal is to create a script that pulls Lat-Long from
# various public, easy-to-reference sources and outputs it as DCS
# coordinates. There appears to be an API that will do this for us in
# a LUA file (a simple config language for missions), but it's not
# flexible enough so we're trying to find a way to write our own
# function in our preferred language.
# 
# The tool of choice appears to be proj, which can be installed using
# sudo apt-get install proj-bin. It's designed to map between
# arbitrary coordinate systems.
# 
# We believe the final command will be something like
#
# proj +proj=tmerc +lat_0=??? +lon_0=??? +k_0=0.9996 +x_0=??? +y_0=???
#
# But we aren't 100% certain. When you run this in terminal, it
# accepts two whitespace-separated decimals as input, which appear to
# be Lat-Long values.
#
# The goal is to determine the unknown values (marked by ???) that map
# from lat-long to the DCS coordinate system. Enough samples are given
# above to both create the model and then test it on points not used
# to create said model.
#
# The equation I sent you over Slack had values for those ???
# numbers. Those were for a different map (the Caucasus Region), and
# we need the numbers for the Persian Gulf region. Each map has a X=0,
# Z=0 point on it, so they definitely don't use the same coordinate
# reprojection.
#
# On the Persian Gulf map, the 0, 0 point is roughly on top of Khasab
# airport.

import csv
import numpy
import dataclasses
import re
from typing import List, NewType
from nptyping import NDArray
import numpy.linalg
import math
import sh
import scipy
import scipy.optimize


LatLon = NewType('LatLon', NDArray[(2,), float])
DCSCoord = NewType('DCSCoord', NDArray[(2,), int])

# k_0, lon_0, x_0, y_0
MercatorParams = NewType('MercatorParams', NDArray[(4,), float])


@dataclasses.dataclass
class Location(object):
    geodetic: LatLon
    ingame: DCSCoord


def parse_ll(s):
    result = re.search(r'(?P<degrees>\d+)d (?P<minutes>\d+)m (?P<seconds>\d+\.\d+)s (?P<direction>[NSEW])', s)
    if not result:
        raise ValueError(f"Not a lat or longitude in 'Nd Nm N.Ns C' format: '{s}'")
    degrees = float(result.group('degrees'))
    minutes = float(result.group('minutes'))
    seconds = float(result.group('seconds'))
    direction = result.group('direction')
    if direction == 'N' or direction == 'E':
        sign = 1
    else:
        sign = -1
    return sign * degrees + (minutes / 60) + (seconds / 3600)

def parse_xz(s):
    result = re.search(r'[XZ] (?P<value>[-+]\d+)', s)
    if not result:
        raise ValueError(f"Not an X or Y coordinate: '{s}'")
    return float(result.group('value'))

def load_data():
    data = []
    with open('data.csv') as f:
        reader = csv.reader(f)
        for row in reader:
            lat = parse_ll(row[0])
            lon = parse_ll(row[1])
            x = parse_xz(row[2])
            z = parse_xz(row[3])
            data.append(Location(
                geodetic=LatLon(numpy.array([lat, lon])),
                ingame=DCSCoord(numpy.array([x, z])),
            ))
    return data


def project(params: MercatorParams, point: LatLon) -> DCSCoord:
    result = sh.proj(
        '+proj=tmerc',
        f'+k_0={params[0]}',
        f'+lon_0={params[1]}'
        f'+x_0={params[2]}',
        f'+y_0={params[3]}',
        _in=f"{point[0]}\t{point[1]}",
    ).stdout
    parsed = [float(e) for e in result.decode('UTF-8').strip().split('\t')]
    return DCSCoord(numpy.array(parsed))


def error(x: MercatorParams, data: List[Location]) -> float:
    sq_err = 0
    for location in data:
        try:
            projected = project(x, location.geodetic)
        except ValueError:
            return math.inf
        location_error = location.ingame - projected
        sq_err += location_error.dot(location_error)
    return sq_err


def main():
    data = load_data()
    example_params = MercatorParams(numpy.array([0.9996, 33, -99517, -4998115]))

    # We provide very large values, rather than None, for
    # unconstrained variables becuase some algorithms (e.g. simulated
    # annealing) can't handle unconstrained variables.
    earth_circ = 40000000
    bounds = numpy.array([
        # k_0 is a scaling factor that has to be larger than 0 but is
        # otherwise unconstrained.
        [0.1, 10],
        # lon_0 is a longitude
        [-90, 90],
        # x_0 and y_0 are technically unconstrained, but as the maps
        # look to be *approximately* in meters we leave it as a
        # meaningful fraction of the circumference of the earth.
        [-earth_circ / 10, earth_circ],
        [-earth_circ / 10, earth_circ],
    ], dtype=float)
    res = scipy.optimize.dual_annealing(
        func=error,
        args=(data,),
        bounds=bounds,
    )
    print(res)
    for location in data:
        projected = project(res.x, location.geodetic)
        print(location.ingame - projected)


if __name__ == "__main__":
    main()
