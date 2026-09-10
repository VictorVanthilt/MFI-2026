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
chosen_path_AM = [
    start_AM,
    Point(1625404, 782596) / 1000,
    Point(1753796, 782596) / 1000,
    Point(1817229, 779000) / 1000,
    end_AM,
]

simple_twobasin_yard = Yard(
    [
        Point(0, 0),
        Point(0, 10),
        Point(30, 10),
        Point(30, 0),
        Point(20, 0),
        Point(20, 9.5),
        Point(10, 9.5),
        Point(10, 0),
        # Point(20, 10),
        # Point(10, 10),
    ]
)

yard_LM = Yard(
    [
        Point(1488100, 793400) / 1000,
        Point(1488100, 817700) / 1000,
        Point(1580829, 817700) / 1000,
        Point(1580829, 813229) / 1000,
        Point(1628171, 813229) / 1000,
        Point(1628171, 817700) / 1000,
        Point(1672282, 817700) / 1000,
        Point(1672282, 811556) / 1000,
        Point(1694804, 811556) / 1000,
        Point(1694804, 817700) / 1000,
        Point(1906500, 817700) / 1000,
        Point(1906500, 812501) / 1000,
        Point(1776371, 812501) / 1000,
        Point(1776371, 813171) / 1000,
        Point(1727829, 813171) / 1000,
        Point(1727829, 812501) / 1000,
        Point(1711324, 812501) / 1000,
        Point(1711324, 799371) / 1000,
        Point(1662229, 799371) / 1000,
        Point(1662229, 795028) / 1000,
        Point(1660571, 795028) / 1000,
        Point(1660571, 799371) / 1000,
        Point(1615329, 799371) / 1000,
        Point(1615329, 795028) / 1000,
        Point(1615234, 795028) / 1000,
        Point(1615234, 795093) / 1000,
        Point(1596333, 795093) / 1000,
        Point(1596333, 793400) / 1000,
    ]
)

# Already part of the yard geometry
forbidden_zones_LM = []

start_LM = Point(1810399, 813012) / 1000
end_LM = Point(1553800, 817400) / 1000

BRIDGE_LM = Component(a_max=192 / 1000, v_max=1920 / 1000, t_sway=5625 / 1000)
TROLLEY_LM = Component(a_max=162 / 1000, v_max=811 / 1000, t_sway=6049 / 1000)

# Path chosen by the algorithm at AM for the LM yard
chosen_path_LM = [
    start_LM,
    Point(1776871, 813671) / 1000,
    Point(1710824, 813671) / 1000,
    Point(1695304, 811056) / 1000,
    Point(1580329, 811056) / 1000,
    end_LM,
]


yard_AM2 = Yard(
    [
        Point(25000, 100) / 1000,
        Point(25000, 34507) / 1000,
        Point(26497, 34507) / 1000,
        Point(26497, 34500) / 1000,
        Point(26500, 34500) / 1000,
        Point(26500, 31900) / 1000,
        Point(52100, 31900) / 1000,
        Point(52100, 36300) / 1000,
        Point(111580, 36300) / 1000,
        Point(111580, 31900) / 1000,
        Point(111900, 31900) / 1000,
        Point(111900, 29100) / 1000,
        Point(163400, 29100) / 1000,
        Point(163400, 26400) / 1000,
        Point(175600, 26400) / 1000,
        Point(175600, 31100) / 1000,
        Point(177400, 31100) / 1000,
        Point(177400, 26400) / 1000,
        Point(194000, 26400) / 1000,
        Point(194000, 32480) / 1000,
        Point(193720, 32480) / 1000,
        Point(193720, 34540) / 1000,
        Point(196560, 34540) / 1000,
        Point(196560, 32480) / 1000,
        Point(196200, 32480) / 1000,
        Point(196200, 26400) / 1000,
        Point(232100, 26400) / 1000,
        Point(232100, 36300) / 1000,
        Point(278300, 36300) / 1000,
        Point(278300, 32900) / 1000,
        Point(281900, 32900) / 1000,
        Point(281900, 20100) / 1000,
        Point(291900, 20100) / 1000,
        Point(291900, 16400) / 1000,
        Point(309400, 16400) / 1000,
        Point(309400, 5300) / 1000,
        Point(301900, 5300) / 1000,
        Point(301900, 100) / 1000,
    ]
)

forbidden_zones_AM2 = [
    Rect(
        Point(152500, 25800) / 1000,
        Point(163500, 29200) / 1000,
    ),
    Rect(
        Point(141500, 25800) / 1000,
        Point(158200, 29200) / 1000,
    ),
    Rect(
        Point(130500, 25800) / 1000,
        Point(147000, 29200) / 1000,
    ),
    Rect(
        Point(119300, 25800) / 1000,
        Point(136200, 29200) / 1000,
    ),
    Rect(
        Point(112000, 25800) / 1000,
        Point(119500, 29200) / 1000,
    ),
]

# This one is only when the crane is loaded
forbidden_zones_AM2_loaded = [
    Rect(
        Point(212000, 23750) / 1000,
        Point(232000, 27500) / 1000,
    )
]

start_AM2 = Point(300_000, 11_000) / 1000
end_AM2 = Point(100_000, 33_000) / 1000

BRIDGE_AM2_UNLOADED = Component(150 / 1000, 2000 / 1000, 3890 / 1000)
TROLLEY_AM2_UNLOADED = Component(100 / 1000, 910 / 1000, 3890 / 1000)
BRIDGE_AM2_LOADED = Component(150 / 1000, 2000 / 1000, 4820 / 1000)
TROLLEY_AM2_LOADED = Component(100 / 1000, 910 / 1000, 4820 / 1000)

chosen_path_AM2_unloaded = [
    start_AM2,
    Point(281300, 11000) / 1000,
    Point(111399, 25199) / 1000,
    end_AM2,
]

chosen_path_AM2_loaded = [
    start_AM2,
    Point(111399, 15800) / 1000,
    end_AM2,
]

if __name__ == "__main__":
    # move = Trajectory2D.through_merged(
    #     [(p.x, p.y) for p in chosen_path_AM], BRIDGE_AM, TROLLEY_AM
    # )
    # plot_route(yard_AM, forbidden_zones_AM, move)
    move = Trajectory2D.through_merged(
        [(p.x, p.y) for p in chosen_path_AM2_unloaded],
        BRIDGE_AM2_UNLOADED,
        TROLLEY_AM2_UNLOADED,
    )
    plot_route(yard_AM2, forbidden_zones_AM2, move)
    move = Trajectory2D.through_merged(
        [(p.x, p.y) for p in chosen_path_AM2_loaded],
        BRIDGE_AM2_LOADED,
        TROLLEY_AM2_LOADED,
    )
    plot_route(yard_AM2, forbidden_zones_AM2 + forbidden_zones_AM2_loaded, move)
    # move = Trajectory2D.through_merged(
    #     [(p.x, p.y) for p in chosen_path_LM], BRIDGE_LM, TROLLEY_LM
    # )
    # plot_route(yard_LM, forbidden_zones_LM, move)
