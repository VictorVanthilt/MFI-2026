"""Checks on the merged trajectories `Trajectory2D.through_merged` builds.

Run me: `python test_merging.py`.

Merging is only allowed to take the standstills out of a route. Everything
else has to survive it: the crane still passes through every waypoint, it
still keeps inside a_max and v_max, every pulse is still a whole number of
sway periods long, a route that cleared the yard still clears it, and what
standstills are left are waypoints the load really does stop on.
"""

import numpy as np

from plc import BRIDGE, TROLLEY, Trajectory, Trajectory2D, _Merger, _time_covered
from test_paths import Point, Rect, Yard, path_valid, trajectory_valid

TOL = 1e-9


def merged_with_times(points):
    """The merged move, plus the times at which it passes each waypoint."""
    waypoints = [(float(x), float(y)) for x, y in points]
    merger = _Merger((BRIDGE, TROLLEY), waypoints[0])
    move = Trajectory2D(*merger.run(waypoints))
    # The route taken apart here has to be the one through_merged hands out.
    assert move.duration == Trajectory2D.through_merged(points).duration
    return move, merger.times


def check_inverse():
    """`_time_covered` really is the inverse of the position curve."""
    # examplecheck.m puts jerkdistinv(j=2, tp=3, lambda=5, x=162) through
    # fsolve; this is the closed form of the same answer.
    assert abs(_time_covered(2, 3, 5, 162) - 12.0196306598) < 1e-9

    # Against the position curve sympy derived, over shapes of every kind:
    # narrow pulses and wide, no cruise and plenty.
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(2000):
        j, tp = rng.uniform(1e-3, 5), rng.uniform(0.1, 20)
        lam = float(rng.choice([0.0, rng.uniform(0, 60)]))
        move = Trajectory.from_shape(BRIDGE, j, tp, lam)
        x = rng.uniform(0, move.distance)
        worst = max(worst, abs(move.pos(_time_covered(j, tp, lam, x)) - x))
    assert worst < 1e-9, f"the inverse is off by {worst}"


def check_route(points, yard=None, forbidden=()):
    """Everything merging promises, checked on one route."""
    move, times = merged_with_times(points)
    naive = Trajectory2D.through(points)

    assert len(times) == len(points)
    assert abs(times[-1] - move.t_end) < TOL, "the clock and the moves disagree"

    # 1. The crane passes exactly through every waypoint. Merging takes the
    #    standstills out of a route; it does not round the corners off.
    for (x, y), when in zip(points, times):
        at = move.pos(when)
        assert abs(at[0] - x) < 1e-6 and abs(at[1] - y) < 1e-6, (
            f"at t={when} the crane is at {at}, not on {(x, y)}"
        )

    # 2. Every axis limit still holds, cruise stretched or not.
    grid = np.linspace(move.t0, move.t_end, 20_000)
    for side, axis in ((move.bridge, BRIDGE), (move.trolley, TROLLEY)):
        assert max(abs(side.vel(grid))) <= axis.v_max + 1e-6
        assert max(abs(side.accel(grid))) <= axis.a_max + 1e-6
        for leg in side:
            # Anti-sway only works if every pulse is a whole number of sway
            # periods long. A merged move grows its cruise, never its pulses.
            periods = leg.tp / axis.t_sway
            assert abs(periods - round(periods)) < TOL, f"t_p = {leg.tp}"

    # 3. The load only ever comes to a full stop on a waypoint, it really is
    #    at rest where `standstills` says it is, and merging only ever takes
    #    stops away -- it never invents a new place to stop.
    naive_stops = [naive.pos(when) for when in naive.standstills]
    assert len(move.standstills) <= len(naive_stops)
    for when in move.standstills:
        assert abs(move.bridge.vel(when)) < TOL, f"still moving at t={when}"
        assert abs(move.trolley.vel(when)) < TOL, f"still moving at t={when}"
        at = move.pos(when)
        assert any(abs(at[0] - x) < 1e-6 and abs(at[1] - y) < 1e-6 for x, y in points), (
            f"the load stops at {at}, which is not a waypoint"
        )
        assert any(
            abs(at[0] - x) < 1e-6 and abs(at[1] - y) < 1e-6 for x, y in naive_stops
        ), f"merging added a stop at {at}"

    # 4. A route that cleared the yard still clears it: within a leg neither
    #    axis turns around, so the curve stays inside the bounding box that
    #    `path_valid` judged that leg by.
    if yard is not None:
        corners = [Point(x, y) for x, y in points]
        if path_valid(yard, forbidden, corners):
            assert trajectory_valid(yard, forbidden, move), "merging left the yard"

    return move.duration, naive.duration


if __name__ == "__main__":
    check_inverse()

    # A two point route has nothing to merge, so it has to come out as the
    # move `through` already made.
    for points in ([(0, 12), (58, 6)], [(0, 12), (0, 6)], [(3, 3), (3, 3)]):
        merged = Trajectory2D.through_merged(points)
        plain = Trajectory2D.through(points)
        grid = np.linspace(plain.t0, plain.t_end + 1.0, 500)
        assert merged.duration == plain.duration
        assert np.allclose(merged.pos(grid), plain.pos(grid))

    # The test problem: the yard and the two forbidden zones from test_paths.py.
    yard = Yard([Point(0, 16), Point(0, 0), Point(60, 0), Point(60, 16)])
    forbidden = [Rect(Point(26, 2), Point(32, 10)), Rect(Point(42, 8), Point(50, 12))]

    routes = [
        [(0, 12), (40, 14), (55, 14), (58, 6)],    # test_paths.py, path1
        [(0, 12), (58, 12), (58, 6)],              # test_paths.py, path2
        [(0, 12), (50, 12), (58, 6)],              # testing.py
        [(0, 12), (58, 14), (58, 6)],              # plc.py; the trolley turns
        [(0, 12), (26, 0), (58, 6)],               # testing.py; all diagonal
        [(0, 3), (10, 3), (25, 3), (40, 3), (58, 8)],           # one long merge
        [(0, 12), (20, 12), (20, 5), (45, 5), (45, 14), (58, 14)],  # a staircase
    ]
    print(f"{'route':58s} {'stopping':>9s} {'merged':>9s}")
    for points in routes:
        merged, plain = check_route(points, yard, forbidden)
        print(f"{str(points):58s} {plain:8.1f}s {merged:8.1f}s")
        assert merged <= plain + TOL, "merging cost time on a hand-picked route"

    # Random routes, to reach what hand-picked ones miss: runs of legs along
    # one axis (which is what merging is for), legs that double back, and
    # diagonal legs that cannot merge at all.
    rng = np.random.default_rng(1)
    saved, lost = [], []
    for _ in range(300):
        x, y = rng.uniform(0, 60), rng.uniform(0, 16)
        points = [(x, y)]
        for _ in range(int(rng.integers(2, 7))):
            roll = rng.random()
            if roll < 0.2:                       # a diagonal leg
                x, y = rng.uniform(0, 60), rng.uniform(0, 16)
            elif roll < 0.6:
                x = rng.uniform(0, 60)
            else:
                y = rng.uniform(0, 16)
            points.append((x, y))
        merged, plain = check_route(points, yard, forbidden)
        (saved if merged <= plain + TOL else lost).append(plain - merged)

    # Merging is a local decision -- three points at a time -- so it is not
    # always a win. Joining two legs can push the move over the n it needs for
    # a_max, widening its pulses, and in case 1b the joined move is held back
    # so that it still passes the middle point; between them that can cost
    # more than the standstill it saves. Rare and small, but it does happen.
    print(
        f"\n{len(saved)} of {len(saved) + len(lost)} random routes came out no "
        f"slower, saving {np.mean(saved):.1f} s on average, up to "
        f"{max(saved):.1f} s"
    )
    if lost:
        print(f"{len(lost)} came out slower, by up to {-min(lost):.2f} s")
    assert len(lost) <= 0.05 * (len(saved) + len(lost)), "merging lost time too often"
    assert not lost or -min(lost) < 5.0, "merging lost more time than expected"

    print("\nall checks passed")
