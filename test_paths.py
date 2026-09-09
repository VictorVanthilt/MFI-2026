from typing import Self

import numpy as np

from plc import BRIDGE, TROLLEY, Trajectory2D

EPSILON = 1e-9


class Point:
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def __add__(self, other: Self) -> Self:
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Self) -> Self:
        return Point(self.x - other.x, self.y - other.y)

    def __mul__(self, other: float) -> Self:
        return Point(self.x * other, self.y * other)

    def __truediv__(self, other: float) -> Self:
        return Point(self.x / other, self.y / other)

    def __repr__(self) -> str:
        return f"({self.x}, {self.y})"


class Rect:
    def __init__(self, bottom_left: Point, top_right: Point):
        self.bottom_left = bottom_left
        self.top_right = top_right

    @classmethod
    def bounding_box(cls, a: Point, b: Point) -> Self:
        bottom_left = Point(min(a.x, b.x), min(a.y, b.y))
        top_right = Point(max(a.x, b.x), max(a.y, b.y))
        return cls(bottom_left, top_right)

    def contains_point(self, point: Point) -> bool:
        return (
            self.bottom_left.x <= point.x <= self.top_right.x
            and self.bottom_left.y <= point.y <= self.top_right.y
        )

    def center(self) -> Point:
        return (self.top_right + self.bottom_left) / 2

    def width(self) -> float:
        return self.top_right.x - self.bottom_left.x

    def height(self) -> float:
        return self.top_right.y - self.bottom_left.y

    # Check if there is overlap with the interior of the other rectangle
    def strict_intersects(self, other: Self) -> bool:
        our_center = self.center()
        other_center = other.center()
        diff = our_center - other_center
        if self.width() + other.width() - 2 * abs(diff.x) < EPSILON:
            return False
        if self.height() + other.height() - 2 * abs(diff.y) < EPSILON:
            return False
        return True

    def intersects_segment(self, a: Point, b: Point) -> bool:
        return (
            seg_seg_intersect(
                self.bottom_left, Point(self.bottom_left.x, self.top_right.y), a, b
            )
            or seg_seg_intersect(
                self.bottom_left, Point(self.top_right.x, self.bottom_left.y), a, b
            )
            or seg_seg_intersect(
                self.top_right, Point(self.bottom_left.x, self.top_right.y), a, b
            )
            or seg_seg_intersect(
                self.top_right, Point(self.top_right.x, self.bottom_left.y), a, b
            )
        )


# Colinear segments are not treated as intersecting
def seg_seg_intersect(a1: Point, a2: Point, b1: Point, b2: Point) -> bool:
    b = a2 - a1
    d = b2 - b1
    bdotdperp = b.x * d.y - b.y * d.x
    if abs(bdotdperp) < EPSILON:
        # Aligned, so not considered as intersecting.
        return False
    c = b1 - a1
    t = (c.x * d.y - c.y * d.x) / bdotdperp
    if t < EPSILON or t > 1 - EPSILON:
        return False
    u = (c.x * b.y - c.y * b.x) / bdotdperp
    if u < EPSILON or u > 1 - EPSILON:
        return False
    return True


# Ray cast from a point to decide whether it is inside the yard. The slope is
# deliberately irrational: a 45-degree ray leaves any lattice point exactly
# through a lattice corner of the yard, and seg_seg_intersect drops both edges
# meeting there, so the crossings come out even and an interior point reads as
# outside.
RAY = Point(1e9, 1e9 * 0.7071067811865476 * 1.2345678901234567)


class Yard:
    def __init__(self, points: list[Point]):
        self.points = points

    # Check if the rectangle is contained inside the yard
    def contains(self, rect: Rect) -> bool:
        num_collisions = 0
        for i in range(len(self.points)):
            # Abuse python negative indexing
            if rect.intersects_segment(self.points[i - 1], self.points[i]):
                # print(
                #     f"Rect({rect.bottom_left} {rect.top_right}) {self.points[i - 1]} {self.points[i]}"
                # )
                return False
            if seg_seg_intersect(
                rect.center(),
                rect.center() + RAY,
                self.points[i - 1],
                self.points[i],
            ):
                num_collisions += 1
        # At this point none of the rectangle edges intersect any of the yard boundaries
        # To check if we are inside it is enough to know if the center of the rectangle is inside
        # This is true if the number of intersection with an arbitrary ray starting at the center
        # with the yard boundaries is odd.
        return num_collisions % 2 == 1


def path_valid(yard: Yard, forbidden_zones: list[Rect], path: list[Point]) -> bool:
    for i in range(len(path) - 1):
        bounding_box = Rect.bounding_box(path[i], path[i + 1])
        for forbidden_zone in forbidden_zones:
            if bounding_box.strict_intersects(forbidden_zone):
                # print(
                #     f"{bounding_box.bottom_left}, {bounding_box.top_right} -- {forbidden_zone.bottom_left}, {forbidden_zone.top_right}"
                # )
                # Path bounding box overlaps with a forbidden zone
                return False
        if not yard.contains(bounding_box):
            # Path bounding box is not contained in the yard
            return False
    return True


def trajectory_points(move: Trajectory2D, n_points: int = 2000) -> list[Point]:
    """Sample a Trajectory2D into the polyline that `path_valid` expects.

    The handover times -- where one leg gives way to the next -- are always
    sampled, so the corners of the path land exactly on the polyline instead
    of being cut by the grid. The rest of the samples are uniform in time.
    """
    grid = np.linspace(move.t0, move.t_end, n_points)
    grid = np.union1d(grid, move.handovers)
    if grid.size < 2:
        # A move that takes no time collapses to a single instant, and a
        # one-point path would pass every check by having no steps at all.
        grid = np.repeat(grid, 2)
    x, y = move.pos(grid)
    return [Point(float(px), float(py)) for px, py in zip(x, y)]


def trajectory_valid(
    yard: Yard,
    forbidden_zones: list[Rect],
    move: Trajectory2D,
    n_points: int = 2000,
) -> bool:
    """Check the path a Trajectory2D actually traces against the yard.

    `path_valid` judges a leg by the bounding box of its endpoints, which is
    the right call for a waypoint list: the two axes run independently, so
    anywhere in that box is reachable. Here the timing is already fixed, so
    the box of each *sampled step* is a much tighter -- and still
    conservative -- envelope around the real curve.

    Raise `n_points` if the path skims a boundary: between two samples the
    envelope is only as good as the grid is fine.
    """
    return path_valid(yard, forbidden_zones, trajectory_points(move, n_points))


def naive_total_time(path: list[Point]) -> float:
    total_time = 0
    for i in range(len(path) - 1):
        a = path[i]
        b = path[i + 1]
        trolley_dist = abs(a.y - b.y)
        bridge_dist = abs(a.x - b.x)
        total_time += max(
            TROLLEY.trajectory(trolley_dist).duration,
            BRIDGE.trajectory(bridge_dist).duration,
        )
    return total_time


if __name__ == "__main__":
    # A bunch of tests.
    assert seg_seg_intersect(Point(0, 1), Point(0, -1), Point(1, 0), Point(-1, 0)), (
        "Segments should intersect"
    )
    assert seg_seg_intersect(Point(0, 1), Point(0, 0), Point(1, 1), Point(-1, 0)), (
        "Segments should intersect"
    )
    assert not seg_seg_intersect(
        Point(0, 1), Point(0, 0), Point(0, 0.5), Point(0, 2)
    ), "Segments should not intersect"

    # This is the test problem.
    yard = Yard(
        [
            Point(0, 16),
            Point(0, 0),
            Point(60, 0),
            Point(60, 16),
        ]
    )
    forbidden1 = Rect(Point(26, 2), Point(32, 10))
    forbidden2 = Rect(Point(42, 8), Point(50, 12))
    a = Point(0, 12)
    b = Point(58, 6)
    assert not path_valid(yard, [forbidden1, forbidden2], [a, b]), (
        "Path crosses forbidden zones"
    )
    assert path_valid(
        yard, [forbidden1, forbidden2], [a, Point(20, 12), Point(20, 5)]
    ), "Path is inside yard and doesn't cross forbidden zones"

    # Example of valid path from A to B in the test problem.
    path1 = [a, Point(40, 14), Point(55, 14), b]
    assert path_valid(
        yard,
        [forbidden1, forbidden2],
        path1,
    ), "Path is inside yard and doesn't cross forbidden zones"
    print("Naive total time:", naive_total_time(path1))
    # Another valid path
    path2 = [a, Point(58, 12), b]
    assert path_valid(yard, [forbidden1, forbidden2], path2), (
        "Path is inside yard and doesn't cross forbidden zones"
    )
    print("Naive total time:", naive_total_time(path2))
    # Not a valid path, but just to see how well we can do
    print("Naive total time:", naive_total_time([a, b]))

    # The same paths, judged as the trajectories they actually turn into. A
    # leg's real curve stays inside the bounding box `path_valid` checks, so
    # a valid waypoint list stays valid once it is driven -- and a straight
    # run through a forbidden zone stays invalid.
    for path, expected in ((path1, True), (path2, True), ([a, b], False)):
        move = Trajectory2D.through([(p.x, p.y) for p in path])
        assert trajectory_valid(yard, [forbidden1, forbidden2], move) is expected, (
            f"trajectory through {path} should be "
            f"{'valid' if expected else 'invalid'}"
        )
        print(f"Trajectory through {path}: {move.duration:.1f} s")
