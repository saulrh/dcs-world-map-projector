import determine_projection
import math

def test_parse_ll():
    assert math.isclose(determine_projection.parse_ll('24d 26m 47.09s N'), 24.4464139)
