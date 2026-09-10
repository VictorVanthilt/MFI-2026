import math
import sys
import tempfile
from pathlib import Path
from typing import Self

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba
from matplotlib.patches import Polygon, Rectangle
from matplotlib.ticker import MaxNLocator, MultipleLocator

from plc import (
    INK, INK_SOFT, PAPER, RULE, RULE_FINE, BRIDGE, TROLLEY, Trajectory2D, plot_moves,
)

EPSILON = 1e-9

KEEP_OUT = "#A4262C"    # the one warm colour on the sheet: stay out

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

    # Inside the rectangle and off its border, so a degenerate rectangle -- a
    # move along one axis only -- strictly contains nothing.
    def strictly_contains_point(self, point: Point) -> bool:
        return (
            self.bottom_left.x + EPSILON < point.x < self.top_right.x - EPSILON
            and self.bottom_left.y + EPSILON < point.y < self.top_right.y - EPSILON
        )

    def corners(self) -> list[Point]:
        return [
            self.bottom_left,
            Point(self.top_right.x, self.bottom_left.y),
            self.top_right,
            Point(self.bottom_left.x, self.top_right.y),
        ]

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

    def plot(self, ax=None, **style):
        """Draw the rectangle as a patch on `ax`.

        The default look is a zone to keep out of, marked the way a drawing
        marks one: hatched at 45 degrees over a barely-tinted fill, so a path
        crossing it still reads through the ruling.
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4), facecolor=PAPER)
        ax.add_patch(
            Rectangle(
                (self.bottom_left.x, self.bottom_left.y),
                self.width(),
                self.height(),
                **{
                    "facecolor": to_rgba(KEEP_OUT, 0.06),
                    "edgecolor": KEEP_OUT,
                    "hatch": "////",
                    "linewidth": 1.0,
                    **style,
                },
            )
        )
        return ax


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


# Is `p` on the closed segment a-b?
def point_on_segment(p: Point, a: Point, b: Point) -> bool:
    ab = b - a
    ap = p - a
    length = math.hypot(ab.x, ab.y)
    if length < EPSILON:
        return math.hypot(ap.x, ap.y) <= EPSILON
    # Distance from the infinite line through a and b, then how far along it.
    if abs(ab.x * ap.y - ab.y * ap.x) / length > EPSILON:
        return False
    along = (ap.x * ab.x + ap.y * ab.y) / length
    return -EPSILON <= along <= length + EPSILON


class Yard:
    def __init__(self, points: list[Point]):
        self.points = points

    # Check if the point is inside the yard, or on its boundary
    def contains_point(self, point: Point) -> bool:
        for i in range(len(self.points)):
            # Abuse python negative indexing
            if point_on_segment(point, self.points[i - 1], self.points[i]):
                return True

        # Crossing number along the ray running left from the point. An edge
        # counts only if it spans the point's height with the lower end
        # included and the upper end excluded, so an edge that merely touches
        # that height is not counted and one passing through it is counted
        # exactly once -- vertices and horizontal edges included. Unlike a
        # ray at a fixed angle there is no direction left to be unlucky with.
        inside = False
        for i in range(len(self.points)):
            a, b = self.points[i - 1], self.points[i]
            if (a.y > point.y) != (b.y > point.y):
                # The half-open test above guarantees a.y != b.y here.
                x_hit = a.x + (point.y - a.y) * (b.x - a.x) / (b.y - a.y)
                if point.x < x_hit:
                    inside = not inside
        return inside

    # Does the closed segment a-b stay inside the yard?
    def contains_segment(self, a: Point, b: Point) -> bool:
        # Whether the segment is in or out can only change where it meets an
        # edge, so cut it at every such point and test the middle of each
        # piece. Both ends being inside is not enough on its own: a segment
        # can leave through a notch and come back, touching the walls only at
        # the two points where it does.
        span = b - a
        length = span.x**2 + span.y**2
        if length < EPSILON:
            return self.contains_point(a)

        cuts = {0.0, 1.0}
        for i in range(len(self.points)):
            p, q = self.points[i - 1], self.points[i]
            edge = q - p
            offset = p - a
            denominator = span.x * edge.y - span.y * edge.x
            if abs(denominator) < EPSILON:
                # Parallel. Only an edge lying along the segment can cut it,
                # and then it is the ends of the overlap that do.
                if abs(offset.x * span.y - offset.y * span.x) < EPSILON:
                    for corner in (p, q):
                        reach = corner - a
                        cuts.add((reach.x * span.x + reach.y * span.y) / length)
                continue
            t = (offset.x * edge.y - offset.y * edge.x) / denominator
            u = (offset.x * span.y - offset.y * span.x) / denominator
            if 0.0 <= u <= 1.0:
                cuts.add(t)

        ordered = sorted(x for x in cuts if 0.0 <= x <= 1.0)
        return all(
            self.contains_point(a + span * (0.5 * (s + t)))
            for s, t in zip(ordered, ordered[1:])
        )

    # Check if the rectangle is contained inside the yard
    def contains(self, rect: Rect) -> bool:
        # A rectangle with no width or height is really a segment, and the
        # tests below cannot see one leave the yard through a notch and come
        # back: its corners are its two ends, no edge crosses it, and it
        # strictly contains nothing. Walk it instead.
        if rect.width() <= EPSILON or rect.height() <= EPSILON:
            return self.contains_segment(rect.bottom_left, rect.top_right)

        # The rectangle is convex and the yard is a simple polygon, so the
        # rectangle is contained exactly when its whole border is in the
        # yard, no yard edge cuts across it, no yard corner pokes into it,
        # and its middle is inside. The corner and crossing tests catch a
        # notch reaching into the rectangle; the border walk catches the
        # case those two are blind to -- a yard edge running along one side
        # of the rectangle and out through it, which crosses nothing and
        # leaves every corner on a wall, while the inside of the rectangle
        # is out of the yard; and the centre catches a rectangle exactly
        # filling a notch, whose border is all wall and whose inside is all
        # outside. Touching counts as contained: a path may run along a wall.
        if not all(self.contains_point(corner) for corner in rect.corners()):
            return False
        corners = rect.corners()
        for a, b in zip(corners, corners[1:] + corners[:1]):
            if not self.contains_segment(a, b):
                return False
        for i in range(len(self.points)):
            if rect.strictly_contains_point(self.points[i - 1]):
                return False
            if rect.intersects_segment(self.points[i - 1], self.points[i]):
                return False
        return self.contains_point(rect.center())

    def plot(self, forbidden_zones=(), ax=None, aspect: float = 2.0):
        """Draw the yard, and the zones inside it that have to stay clear.

        The sheet the route is drawn on: graph paper, the yard walls inked
        over it, and the zones hatched out. Both are context rather than
        data, so nothing here competes with the path that goes on top.

        Pass `ax` to draw onto existing axes; a figure shaped like the yard is
        made otherwise. The axes come back either way, ready for a
        `Trajectory2D.plot(ax=ax)` on top.
        """
        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]

        if ax is None:
            # A yard is usually much wider than it is deep, and the axes are
            # drawn `aspect` times stretched upwards, so a square figure would
            # be mostly empty.
            span = max(max(xs) - min(xs), 1e-9)
            height = 10.0 * aspect * (max(ys) - min(ys)) / span
            _, ax = plt.subplots(
                figsize=(10.0, min(max(height + 2.4, 3.4), 9.0)), facecolor=PAPER
            )
        ax.figure.set_facecolor(PAPER)
        ax.set_facecolor(PAPER)

        ax.add_patch(
            Polygon(
                [(p.x, p.y) for p in self.points],
                closed=True,
                facecolor="none",
                edgecolor=INK,
                linewidth=1.8,
                joinstyle="miter",
                label="yard",
                zorder=2,
            )
        )
        for i, zone in enumerate(forbidden_zones):
            # One legend entry between them, not one apiece.
            zone.plot(ax, label="forbidden" if i == 0 else None, zorder=3)

        # Graph paper: a metre rule under a ten-metre one.
        ax.xaxis.set_minor_locator(MultipleLocator(2))
        ax.yaxis.set_minor_locator(MultipleLocator(2))
        # Whole metres up the side; the default lands on halves.
        ax.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
        ax.grid(True, which="minor", color=RULE_FINE, linewidth=0.5)
        ax.grid(True, which="major", color=RULE, linewidth=0.7)
        ax.set_axisbelow(True)     # the paper reads through the yard, not over it

        ax.set_xlabel("bridge (m)", fontsize=9, color=INK_SOFT)
        ax.set_ylabel("trolley (m)", fontsize=9, color=INK_SOFT)
        ax.tick_params(labelsize=8, colors=INK_SOFT, length=2)
        ax.tick_params(which="minor", length=0)
        # Two rules, not four: the yard walls are the frame here.
        for side, spine in ax.spines.items():
            spine.set_visible(side in ("left", "bottom"))
            spine.set_color(RULE)
            spine.set_linewidth(0.8)
        ax.set_aspect(aspect)
        ax.autoscale_view()
        # A band above the yard for the legend, and a little air below it.
        ax.margins(0.03, 0.22)
        return ax


def plot_route(yard, forbidden_zones, moves, ax=None, save=None, aspect=2.0,
               labels=None):
    """Draw one crane move, or several, inside the yard they have to keep to.

    The yard and the zones it must stay out of go down first, then the path
    the crane traces through them. `moves` is one move or a list of them; a
    list goes down on the one plan, each route tagged with its entry in
    `labels`, or A, B, C... (see `plot_moves`). Pass `ax` to draw onto
    existing axes; a new figure is made and shown otherwise.

    `save` writes the drawing out as well as putting it on screen. A plan of a
    yard ends up in a report, so it is a PDF -- vector, and it scales -- unless
    the name given says otherwise; matplotlib takes the format from the
    suffix. A folder that is not there yet is made rather than complained
    about, and handing in `ax` saves the whole figure it belongs to rather
    than the one panel.
    """
    if isinstance(moves, Trajectory2D):
        moves = [moves]
    if labels is None:
        labels = [chr(ord("A") + i) for i in range(len(moves))]

    own_figure = ax is None
    ax = yard.plot(forbidden_zones, ax=ax, aspect=aspect)
    plot_moves(moves, labels=labels, ax=ax, aspect=aspect)

    # A thin margin all round; the legend lives above the axes, so the yard
    # needs no headroom carved out of its own drawing.
    low, high = ax.dataLim.intervaly
    span = max(high - low, 1e-9)
    ax.set_ylim(low - 0.07 * span, high + 0.07 * span)

    # Say whether the route came out clear, and if it did not, of what. The
    # test is the PLC's own, and it is a blunt one: per leg, the box spanned
    # by the two waypoints, and does that box hit anything. It never looks at
    # the curve the crane actually drives, so a move and its merged twin are
    # judged the same -- see `trajectory_valid` for the stricter question.
    checked = [(label, move) for label, move in zip(labels, moves)
               if move.waypoints is not None]
    faults = {}
    for label, move in checked:
        route = [Point(x, y) for x, y in move.waypoints]
        if path_valid(yard, forbidden_zones, route):
            continue
        if path_valid(yard, (), route):
            faults[label] = "enters a forbidden zone"
            # The yard itself is clear, so any zone that fails on its own is
            # one a leg runs into. Light those up.
            for zone in forbidden_zones:
                if not path_valid(yard, [zone], route):
                    zone.plot(ax, facecolor=to_rgba(KEEP_OUT, 0.3), linewidth=2.0,
                              zorder=6)
        else:
            faults[label] = "leaves the yard"

    # One route needs no naming; with several, each verdict says whose it is.
    named = len(moves) > 1
    if checked:
        if named:
            note = "  ".join(
                f"{label} \u2717 {faults[label]}" if label in faults
                else f"{label} \u2713"
                for label, _ in checked
            )
        else:
            note = f"\u2717 {faults[checked[0][0]]}" if faults else "\u2713 valid path"
        ax.set_title(note, loc="right", fontsize=9.5, family="monospace", pad=30,
                     color=KEEP_OUT if faults else INK)
        # The legend floats just above the axes, so both titles move up a
        # band to keep out of its way.
        ax.set_title(ax.get_title(loc="left"), loc="left", fontsize=11,
                     color=INK, family="monospace", pad=30)

    if faults:
        alarm = "\n".join(f"{label} {fault}".upper() if named else fault.upper()
                          for label, fault in faults.items())
        # A drawing that cannot be built gets stamped across the middle, and
        # ruled off on all four sides. There is no reading the plan without
        # seeing it.
        ax.text(0.5, 0.42, alarm, transform=ax.transAxes, ha="center",
                va="center", fontsize=26, fontweight="bold", family="monospace",
                color=KEEP_OUT, zorder=10,
                bbox={"facecolor": PAPER, "alpha": 0.8, "edgecolor": KEEP_OUT,
                      "linewidth": 1.8, "boxstyle": "square,pad=0.55"})
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(KEEP_OUT)
            spine.set_linewidth(1.8)

    # One legend row, floated above the axes so it can never sit on the
    # yard, however wide and thin the yard turns out to be. Several routes
    # each put in a start and an end; the key wants them once.
    handles, names = ax.get_legend_handles_labels()
    entries = dict(zip(names, handles))
    ax.legend(
        list(entries.values()),
        list(entries),
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=len(entries),
        frameon=False,
        fontsize=8.5,
        labelcolor=INK_SOFT,
        handletextpad=0.4,
        columnspacing=1.6,
    )
    if own_figure:
        # A little more than the default all round: the note is right-aligned
        # to the edge of the plan, and wants air outside it.
        plt.tight_layout(pad=1.6)

    if save is not None:
        # Written out before the figure goes up: a backend that hands it to a
        # window can leave nothing behind to save afterwards.
        out = Path(save)
        out = out if out.suffix else out.with_suffix(".pdf")
        out.parent.mkdir(parents=True, exist_ok=True)
        ax.figure.savefig(out, bbox_inches="tight")

    if own_figure:
        plt.show()
    return ax
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


def leg_boundaries(move: Trajectory2D) -> list[float]:
    """Every time at which either axis starts or finishes one of its moves.

    Within one move an axis never reverses: its velocity runs 0 -> peak -> 0
    without changing sign, so the position is monotonic. Between two of these
    times both axes are therefore monotonic, which means the curve they trace
    stays inside the bounding box of its two endpoints.
    """
    times = {move.t0, move.t_end}
    for side in (move.bridge, move.trolley):
        # A TrajectoryChain carries its moves; a bare Trajectory is one move.
        for leg in getattr(side, "trajectories", (side,)):
            times.update((leg.t0, leg.t_end))
    return sorted(x for x in times if move.t0 <= x <= move.t_end)


def trajectory_points(move: Trajectory2D, n_points: int = 2000) -> list[Point]:
    """Sample a Trajectory2D into the polyline that `path_valid` expects.

    Every leg boundary is sampled, so no step ever straddles one and each step
    spans a stretch where both axes are monotonic. The bounding box of a step
    is then a true envelope of the curve inside it, whatever `n_points` is.
    The uniform samples in between only shrink those boxes.

    This is the path as a list of points -- for drawing it, or for feeding to
    anything that wants waypoints. `trajectory_valid` does not go through it:
    it splits the awkward stretches instead of sampling everything evenly.
    """
    grid = np.union1d(
        np.linspace(move.t0, move.t_end, max(n_points, 2)),
        leg_boundaries(move),
    )
    if grid.size < 2:
        # A move that takes no time collapses to a single instant, and a
        # one-point path would pass every check by having no steps at all.
        grid = np.repeat(grid, 2)
    x, y = move.pos(grid)
    return [Point(float(px), float(py)) for px, py in zip(x, y)]


def _stretch_valid(
    yard: Yard,
    forbidden_zones: list[Rect],
    move: Trajectory2D,
    t0: float,
    t1: float,
    depth: int,
) -> bool:
    """Is the curve between times t0 and t1 clear? Split it if unsure."""
    x, y = move.pos([t0, t1])
    ends = [Point(float(x[0]), float(y[0])), Point(float(x[1]), float(y[1]))]

    # `path_valid` on the two ends checks their bounding box, which -- as long
    # as t0 and t1 sit within one leg -- is a box the curve cannot leave. So a
    # pass here settles the whole stretch.
    if path_valid(yard, forbidden_zones, ends):
        return True

    # A failure does not settle anything: the box is bigger than the curve, so
    # what it hit may be somewhere the curve never goes. Halve it and ask
    # again about the two tighter boxes.
    mid = 0.5 * (t0 + t1)
    if depth <= 0 or not t0 < mid < t1:
        # Out of depth, or the stretch is too short to halve any further.
        return False
    return _stretch_valid(
        yard, forbidden_zones, move, t0, mid, depth - 1
    ) and _stretch_valid(yard, forbidden_zones, move, mid, t1, depth - 1)


def trajectory_valid(
    yard: Yard,
    forbidden_zones: list[Rect],
    move: Trajectory2D,
    max_depth: int = 14,
) -> bool:
    """Check the path a Trajectory2D actually traces against the yard.

    `path_valid` judges a leg by the bounding box of its endpoints, which is
    the right call for a waypoint list: the two axes run independently, so
    anywhere in that box is reachable. Here the timing is already fixed, so
    the curve is one particular line through that box and most of the box is
    somewhere it never goes.

    So the check starts from one box per leg and splits any box that fails in
    half, down to `max_depth` levels. Splitting only ever tightens the
    envelope around the curve, and a stretch that still looks blocked at full
    depth is reported blocked -- erring, as the box does, on the safe side.
    Only the boxes that straddle a boundary get split, so a path well clear of
    everything costs one check per leg.
    """
    bounds = leg_boundaries(move)
    if len(bounds) < 2:
        # A move that takes no time is a single instant; check that spot.
        bounds = bounds * 2
    return all(
        _stretch_valid(yard, forbidden_zones, move, t0, t1, max_depth)
        for t0, t1 in zip(bounds, bounds[1:])
    )


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

    # A yard with a slot cut into it, to exercise the non-convex case.
    notched = Yard(
        [
            Point(0, 0),
            Point(60, 0),
            Point(60, 16),
            Point(35, 16),
            Point(35, 6),
            Point(20, 6),
            Point(20, 16),
            Point(0, 16),
        ]
    )
    assert notched.contains_point(Point(27, 3)), "below the slot is inside"
    assert not notched.contains_point(Point(27, 12)), "the slot is outside"
    assert notched.contains_point(Point(20, 12)), "the slot wall is inside"
    assert notched.contains(Rect(Point(10, 1), Point(50, 5))), "clears the slot"
    assert not notched.contains(Rect(Point(10, 1), Point(50, 8))), "reaches in"
    for path, expected in (
        ([Point(5, 3), Point(55, 3)], True),
        ([Point(5, 3), Point(27, 12)], False),
        ([Point(10, 14), Point(50, 14)], False),
    ):
        move = Trajectory2D.through([(p.x, p.y) for p in path])
        assert trajectory_valid(notched, [], move) is expected, (
            f"trajectory through {path} should be "
            f"{'valid' if expected else 'invalid'} in the notched yard"
        )

    # The drawing runs. Handing it axes keeps it from opening a window, so
    # this stays a check rather than a distraction.
    _, scratch = plt.subplots()
    for route in (path2, [a, b]):     # one that clears the zones, one that does not
        plot_route(
            yard,
            [forbidden1, forbidden2],
            Trajectory2D.through_merged([(p.x, p.y) for p in route]),
            ax=scratch,
        )
    notched.plot(ax=scratch)

    # Saving writes a real file, a PDF when the name does not say otherwise,
    # and makes the folder it was pointed at.
    with tempfile.TemporaryDirectory() as folder:
        plot_route(
            yard,
            [forbidden1, forbidden2],
            Trajectory2D.through_merged([(p.x, p.y) for p in path2]),
            ax=scratch,
            save=Path(folder) / "plans" / "route",
        )
        written = Path(folder, "plans", "route.pdf")
        assert written.exists(), "plot_route(save=...) wrote nothing"
        assert written.read_bytes()[:5] == b"%PDF-", "what it wrote is not a pdf"
    plt.close()

    # `python test_paths.py --plot` draws the test problem instead of only
    # asserting things about it: the yard, the two zones to keep out of, and
    # the route through them driven both ways.
    if "--plot" in sys.argv:
        route = [(p.x, p.y) for p in path1]
        _, axes = plt.subplots(2, 1, figsize=(10, 7))
        for ax, (name, move) in zip(
            axes,
            (
                ("stopping on every point", Trajectory2D.through(route)),
                ("merged", Trajectory2D.through_merged(route)),
            ),
        ):
            plot_route(yard, [forbidden1, forbidden2], move, ax=ax)
            ax.set_title(name, fontsize=10, color=INK_SOFT)
        plt.tight_layout()
        plt.show()
