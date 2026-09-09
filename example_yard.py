from test_paths import plot_route
from plc import Trajectory2D
from plc import Component
from test_paths import Point, Rect, Yard
import matplotlib.pyplot as plt

# Based on screenshot in slides.
yard_screenshot = Yard(
    [
        Point(99, 277),
        Point(99, 182),
        Point(197, 182),
        Point(197, 143),
        Point(202, 143),
        Point(202, 72),
        Point(211, 72),
        Point(211, 76),
        Point(390, 76),
        Point(390, 44),
        Point(402, 44),
        Point(402, 76),
        Point(778, 76),
        Point(778, 47),
        Point(790, 47),
        Point(790, 76),
        Point(1597, 76),
        Point(1597, 391),
        Point(1445, 391),
        Point(1445, 342),
        Point(990, 342),
        Point(990, 388),
        Point(931, 388),
        Point(931, 352),
        Point(791, 352),
        Point(792, 419),
        Point(778, 419),
        Point(778, 352),
        Point(735, 352),
        Point(735, 285),
        Point(562, 285),
        Point(562, 277),
        Point(735, 277),
        Point(735, 265),
        Point(778, 265),
        Point(778, 165),
        Point(790, 165),
        Point(790, 183),
        Point(822, 183),
        Point(822, 91),
        Point(725, 91),
        Point(725, 83),
        Point(457, 83),
        Point(457, 162),
        Point(480, 162),
        Point(480, 192),
        Point(402, 192),
        Point(402, 418),
        Point(390, 418),
        Point(390, 192),
        Point(108, 192),
        Point(108, 277),
    ]
)

forbidden_zones_screenshot = [
    Rect(Point(400, 83), Point(490, 193)),
    Rect(Point(990, 300), Point(1100, 343)),
]


# Actual yard data given to us (data is in mm)
yard_AM = Yard(
    [
        Point(1510300, 758500) / 1000,
        Point(1510300, 782800) / 1000,
        Point(1817729, 782800) / 1000,
        Point(1817729, 780729) / 1000,
        Point(1879900, 780729) / 1000,
        Point(1879900, 758500) / 1000,
        Point(1790571, 758500) / 1000,
        Point(1790571, 778971) / 1000,
        Point(1776529, 778971) / 1000,
        Point(1776529, 778271) / 1000,
        Point(1765171, 778271) / 1000,
        Point(1765171, 778971) / 1000,
        Point(1753671, 778971) / 1000,
        Point(1753671, 782471) / 1000,
        Point(1665329, 782471) / 1000,
        Point(1665329, 782421) / 1000,
        Point(1658171, 782421) / 1000,
        Point(1658171, 782471) / 1000,
        Point(1625529, 782471) / 1000,
        Point(1625529, 779371) / 1000,
        Point(1604829, 779371) / 1000,
        Point(1604829, 761829) / 1000,
        Point(1607829, 761829) / 1000,
        Point(1607829, 758500) / 1000,
    ]
)

# Already part of the yard geometry
forbidden_zones_AM = []

start_AM = Point(1624250, 780434) / 1000
end_AM = Point(1863580, 779000) / 1000

BRIDGE_AM = Component(a_max=190 / 1000, v_max=1960 / 1000, t_sway=5234 / 1000)
TROLLEY_AM = Component(a_max=132 / 1000, v_max=790 / 1000, t_sway=5834 / 1000)

# Path chosen by the algorithm at AM
chosen_path = [
    start_AM,
    Point(1625404, 782596) / 1000,
    Point(1753796, 782596) / 1000,
    Point(1817229, 779000) / 1000,
    end_AM,
]


if __name__ == "__main__":
    # ax = yard_screenshot.plot(forbidden_zones=forbidden_zones_screenshot)
    # plt.show()
    move = Trajectory2D.through_merged(
        [(p.x, p.y) for p in chosen_path], BRIDGE_AM, TROLLEY_AM
    )
    plot_route(yard_AM, forbidden_zones_AM, move)
