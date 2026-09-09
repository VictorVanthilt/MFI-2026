"""Anti-sway crane motion profile, derived symbolically.

The load hangs on a cable, so it is a pendulum of period `t_sway`. Shaping the
acceleration as four equal jerk pulses, each exactly `tp = n * t_sway` long.

"""

import math
from typing import NamedTuple, Optional

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

# ---------------------------------------------------------------------------
# Symbols
# ---------------------------------------------------------------------------

t = sp.Symbol("t", nonnegative=True)
j_s, tp_s = sp.symbols("j t_p", positive=True)
lam_s = sp.Symbol("lambda", nonnegative=True)
d_s, a_max_s = sp.symbols("d a_max", positive=True)

# (jerk, duration) for each phase, in order.
_PHASES = (
    (j_s, tp_s),             # 1  jerk up:    accel 0 -> +a_p
    (-j_s, tp_s),            # 2  jerk down:  accel +a_p -> 0, velocity hits v_p
    (sp.Integer(0), lam_s),  # 3  cruise at v_p
    (-j_s, tp_s),            # 4  jerk down:  accel 0 -> -a_p
    (j_s, tp_s),             # 5  jerk up:    accel -a_p -> 0, velocity hits 0
)


class Symbolic(NamedTuple):
    """The derivation, in terms of the symbols j, t_p, lambda."""

    jerk: sp.Expr      # jerk(t), piecewise
    accel: sp.Expr     # a(t),    piecewise
    vel: sp.Expr       # v(t),    piecewise
    pos: sp.Expr       # x(t),    piecewise
    distance: sp.Expr  # x at the end of phase 5
    lam: sp.Expr       # cruise time solving distance == d
    tp: sp.Expr        # t_p of a cruise-free move at peak accel a_max


def _derive() -> Symbolic:
    """Integrate the jerk profile phase by phase, carrying continuity."""
    u, s = sp.Dummy("u"), sp.Dummy("s")
    jerk_pieces, accel_pieces, vel_pieces, pos_pieces = [], [], [], []

    t_start = sp.Integer(0)          # when the current phase begins
    a0 = v0 = x0 = sp.Integer(0)     # state carried in from the previous phase

    for jerk, duration in _PHASES:
        # Integrate within the phase, using local time s = t - t_start.
        accel = a0 + jerk * s
        vel = v0 + sp.integrate(accel.subs(s, u), (u, 0, s))
        pos = x0 + sp.integrate(vel.subs(s, u), (u, 0, s))

        t_end = t_start + duration
        local = {s: t - t_start}
        jerk_pieces.append((jerk, t <= t_end))
        accel_pieces.append((sp.expand(accel.subs(local)), t <= t_end))
        vel_pieces.append((sp.expand(vel.subs(local)), t <= t_end))
        pos_pieces.append((sp.expand(pos.subs(local)), t <= t_end))

        # Hand the end-of-phase state to the next phase.
        a0, v0, x0 = (sp.expand(e.subs(s, duration)) for e in (accel, vel, pos))
        t_start = t_end

    total = sp.simplify(x0)

    # Cruise time that makes the profile cover exactly d.
    lam = sp.solve(sp.Eq(total, d_s), lam_s)[0]

    # Pulse width of a move with no cruise (lambda = 0) run at peak accel
    # a_max. Peak accel is a_p = j*t_p, so j = a_max/t_p.
    tp = sp.solve(sp.Eq(total.subs({lam_s: 0, j_s: a_max_s / tp_s}), d_s), tp_s)[0]

    return Symbolic(
        jerk=sp.Piecewise(*jerk_pieces, (sp.Integer(0), True)),
        accel=sp.Piecewise(*accel_pieces, (sp.Integer(0), True)),
        vel=sp.Piecewise(*vel_pieces, (sp.Integer(0), True)),
        pos=sp.Piecewise(*pos_pieces, (total, True)),
        distance=total,
        lam=sp.simplify(lam),
        tp=sp.simplify(tp),
    )


SYM = _derive()


# Lambdify once: a profile's shape depends only on (j, t_p, lambda).
_EVAL = {
    name: sp.lambdify((t, j_s, tp_s, lam_s), expr, "numpy")
    for name, expr in (
        ("jerk", SYM.jerk),
        ("accel", SYM.accel),
        ("vel", SYM.vel),
        ("pos", SYM.pos),
    )
}


def _plot_profile(t, jerk, accel, vel, pos, boundaries=()):
    """Draw jerk/accel/vel/pos on four stacked axes.

    `boundaries` marks the times where one move hands over to the next.
    """
    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(8, 6))
    for a, values, label, unit in zip(
        ax,
        (jerk, accel, vel, pos),
        ("jerk", "accel", "vel", "pos"),
        ("m/s³", "m/s²", "m/s", "m"),
    ):
        a.plot(t, values, label=label)
        a.set_ylabel(f"{label} ({unit})")
        for boundary in boundaries:
            a.axvline(boundary, color="grey", linestyle="--", linewidth=0.8)
        a.grid()
        a.legend()
    ax[3].set_xlabel("time (s)")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# One axis
# ---------------------------------------------------------------------------


class Component:
    """Anti-sway motion profile generator for a single crane axis."""

    def __init__(self, a_max: float, v_max: float, t_sway: float):
        self.a_max = float(a_max)
        self.v_max = float(v_max)
        self.t_sway = float(t_sway)

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(a_max={self.a_max}, "
            f"v_max={self.v_max}, t_sway={self.t_sway})"
        )

    def n_max(self) -> int:
        """Sway periods needed for the accel phase to reach v_max.

        Beyond this the velocity saturates and a cruise phase is required.
        """
        return math.ceil(self.v_max / (self.t_sway * self.a_max))

    @property
    def min_time(self) -> float:
        """Shortest possible move: one sway period per jerk pulse."""
        return 4.0 * self.t_sway

    def n(self, distance: float) -> int:
        """Sway periods per jerk pulse needed to cover `distance`."""
        return math.ceil(np.sqrt(abs(distance) / (2 * self.a_max * self.t_sway**2)))

    def trajectory(
        self, distance: float, x0: float = 0.0, t0: float = 0.0
    ) -> "Trajectory":
        """The move covering `distance`, starting at `x0` at time `t0`."""
        return Trajectory(self, distance, x0, t0)

    def plot_dist_time(self, ax):
        dist_range = np.linspace(0, 100, 200)
        time_values = [self.trajectory(d).duration for d in dist_range]
        ax.plot(dist_range, time_values, label="Bridge")


# ---------------------------------------------------------------------------
# One move
# ---------------------------------------------------------------------------


def _time_covered(j: float, tp: float, lam: float, x: float) -> float:
    """How long a profile of shape (j, tp, lam) takes to cover `x`.

    The inverse of `SYM.pos` -- jerkdist and jerkdistinv in the MATLAB -- and
    no solver is needed for it. The first pulse is a plain cube root, the
    cruise is a straight line, and the braking half is the accelerating half
    run backwards, so only the second pulse takes any work: there
    6x/j = t**3 - 2 (t - tp)**3, which the substitution t = 2 tp + w turns into
    the depressed cubic w**3 - 6 tp**2 w + q. All three of its roots are real,
    and the one inside the pulse -- w in [-tp, 0] -- is the middle branch of
    the trigonometric form.

    `j` and `x` are magnitudes: a move has the same shape whichever way it
    runs. `x` outside the move clamps to its two ends.
    """
    if j <= 0.0 or tp <= 0.0:
        return 0.0

    total = j * tp**2 * (2.0 * tp + lam)
    x = min(max(x, 0.0), total)

    first_pulse = j * tp**3 / 6.0                      # covered as accel peaks
    cruise_start = j * tp**3                           # ... as v_p is reached
    cruise_end = cruise_start + j * tp**2 * lam        # ... as braking starts

    if x <= first_pulse:
        return float(np.cbrt(6.0 * x / j))
    if x <= cruise_start:
        q = 6.0 * x / j - 6.0 * tp**3
        theta = math.acos(-q / (4.0 * math.sqrt(2.0) * tp**3))
        root = math.sqrt(2.0) * math.cos(theta / 3.0 - 2.0 * math.pi / 3.0)
        return 2.0 * tp * (1.0 + root)
    if x <= cruise_end:
        return 2.0 * tp + (x - cruise_start) / (j * tp**2)
    return 4.0 * tp + lam - _time_covered(j, tp, lam, total - x)


class Trajectory:
    """A single anti-sway move along one axis.

    All five phases -- jerk up, jerk down, cruise, jerk down, jerk up -- are
    always present; a move too short to reach v_max simply gets a cruise phase
    of zero length. `distance` is signed, and the trajectory knows where (`x0`)
    and when (`t0`) it starts, so it can be evaluated on an absolute time axis
    and chained with the moves before and after it.
    """

    def __init__(
        self,
        component: Component,
        distance: float,
        x0: float = 0.0,
        t0: float = 0.0,
    ):
        self.component = component
        self.distance = float(distance)
        self.x0 = float(x0)
        self.t0 = float(t0)

        d = abs(self.distance)
        sign = math.copysign(1.0, self.distance)

        # Widen the pulses until the acceleration limit is met, but never past
        # n_max: beyond that the velocity has saturated, so longer pulses buy
        # no extra speed and only stretch the move.
        self.n = min(component.n(d), component.n_max())

        if self.n == 0:
            # Standing still: five empty phases.
            self.tp = self.j = self.lam = 0.0
            self.saturated = False
        else:
            self.tp = component.t_sway * self.n
            # Cruise at v_max for whatever the four pulses leave over.
            j_max = component.v_max / self.tp**2
            lam = float(SYM.lam.subs({d_s: d, j_s: j_max, tp_s: self.tp}))
            self.saturated = lam > 0.0     # does the move reach v_max?

            if self.saturated:
                self.j, self.lam = sign * j_max, lam
            else:
                # The four pulses alone already overshoot d, so there is no
                # cruise and the jerk backs off to cover exactly d instead.
                self.j, self.lam = sign * d / (2 * self.tp**3), 0.0

        self.ap = self.j * self.tp                  # peak acceleration
        self.vp = self.j * self.tp**2               # peak (cruise) velocity
        self.duration = 4.0 * self.tp + self.lam

    @classmethod
    def from_shape(cls, component, j, tp, lam, x0: float = 0.0, t0: float = 0.0):
        """A move given by its profile rather than by the distance it covers.

        Merging stretches the cruise phase of a move that is already running,
        which leaves a profile no distance would have been given on its own,
        so the shape has to be handed over as it stands.
        """
        move = cls.__new__(cls)
        move.component = component
        move.j, move.tp, move.lam = float(j), float(tp), float(lam)
        move.n = round(move.tp / component.t_sway)
        move.saturated = move.lam > 0.0
        move.ap = move.j * move.tp
        move.vp = move.j * move.tp**2
        move.duration = 4.0 * move.tp + move.lam
        move.distance = move.vp * (2.0 * move.tp + move.lam)
        move.x0, move.t0 = float(x0), float(t0)
        return move

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(distance={self.distance}, x0={self.x0}, "
            f"t0={self.t0}, n={self.n}, j={self.j}, tp={self.tp}, "
            f"lam={self.lam}, duration={self.duration})"
        )

    @property
    def t_end(self) -> float:
        """When the move finishes."""
        return self.t0 + self.duration

    @property
    def x_end(self) -> float:
        """Where the move finishes."""
        return self.x0 + self.distance

    @property
    def phases(self) -> tuple:
        """The five (jerk, duration) pairs, in order."""
        return (
            (self.j, self.tp),
            (-self.j, self.tp),
            (0.0, self.lam),
            (-self.j, self.tp),
            (self.j, self.tp),
        )

    def _eval(self, name: str, time, offset: float = 0.0):
        # Before t0 and after t_end the axis sits still, so clamping the query
        # into the profile extends it with the right constant.
        local = np.clip(np.asarray(time, dtype=float) - self.t0, 0.0, self.duration)
        values = np.asarray(_EVAL[name](local, self.j, self.tp, self.lam), dtype=float)
        # A degenerate profile can collapse to a constant, which lambdify
        # returns as a scalar rather than an array.
        out = np.broadcast_to(values, local.shape) + offset
        return float(out) if out.ndim == 0 else out

    def jerk(self, time):
        """Jerk at absolute time(s) `time`."""
        return self._eval("jerk", time)

    def accel(self, time):
        """Acceleration at absolute time(s) `time`."""
        return self._eval("accel", time)

    def vel(self, time):
        """Velocity at absolute time(s) `time`."""
        return self._eval("vel", time)

    def pos(self, time):
        """Position at absolute time(s) `time`, measured from x0."""
        return self._eval("pos", time, self.x0)

    def time_at(self, position) -> float:
        """When the move passes `position` -- jerkdistinv in the MATLAB.

        An axis never turns around within a move, so it passes each position
        on the way exactly once. Positions short of the start or beyond the
        end clamp to the move's own start and end times.
        """
        covered = (position - self.x0) * (1.0 if self.distance >= 0.0 else -1.0)
        return self.t0 + _time_covered(abs(self.j), self.tp, self.lam, covered)

    def sample(self, n_points: int = 600):
        """Evaluate the trajectory on a time grid, for plotting.

        Returns (t, jerk, accel, vel, pos) as numpy arrays.
        """
        end = self.t_end if self.duration > 0.0 else self.t0 + 1.0
        grid = np.linspace(self.t0, end, n_points)
        return grid, self.jerk(grid), self.accel(grid), self.vel(grid), self.pos(grid)

    def plot(self, n_points: int = 600):
        _plot_profile(*self.sample(n_points))


# ---------------------------------------------------------------------------
# A sequence of moves
# ---------------------------------------------------------------------------


class TrajectoryChain:
    """A sequence of moves along one axis, run back to back.

    Each move has to pick up exactly where the previous one left off: same
    axis, same position, and no earlier in time. A move that starts later than
    the previous one ended is the axis standing still in between, waiting for
    the other axis to catch up. Once chained, the sequence answers the same
    questions as a single `Trajectory` does.
    """

    TOL = 1e-9  # slack allowed on the tail-head match

    def __init__(self, trajectories):
        self.trajectories = tuple(trajectories)
        if not self.trajectories:
            raise ValueError("a chain needs at least one trajectory")

        for i, (prev, nxt) in enumerate(zip(self.trajectories, self.trajectories[1:])):
            if nxt.component is not prev.component:
                raise ValueError(
                    f"trajectory {i + 1} runs on a different axis than trajectory {i}"
                )
            if abs(nxt.x0 - prev.x_end) > self.TOL:
                raise ValueError(
                    f"trajectory {i} ends at x={prev.x_end} but trajectory "
                    f"{i + 1} starts at x={nxt.x0}"
                )
            if nxt.t0 < prev.t_end - self.TOL:
                raise ValueError(
                    f"trajectory {i + 1} starts at t={nxt.t0}, before "
                    f"trajectory {i} ends at t={prev.t_end}"
                )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({len(self)} moves, "
            f"x0={self.x0} -> {self.x_end}, t0={self.t0} -> {self.t_end})"
        )

    def __len__(self) -> int:
        return len(self.trajectories)

    def __iter__(self):
        return iter(self.trajectories)

    def __getitem__(self, index):
        return self.trajectories[index]

    @property
    def component(self) -> Component:
        """The axis the whole chain runs on."""
        return self.trajectories[0].component

    @property
    def t0(self) -> float:
        """When the first move starts."""
        return self.trajectories[0].t0

    @property
    def x0(self) -> float:
        """Where the first move starts."""
        return self.trajectories[0].x0

    @property
    def t_end(self) -> float:
        """When the last move finishes."""
        return self.trajectories[-1].t_end

    @property
    def x_end(self) -> float:
        """Where the last move finishes."""
        return self.trajectories[-1].x_end

    @property
    def duration(self) -> float:
        """Time from the start of the first move to the end of the last."""
        return self.t_end - self.t0

    @property
    def distance(self) -> float:
        """Net displacement over the whole chain."""
        return self.x_end - self.x0

    def _eval(self, name: str, time):
        # Hand each query time to the move that owns it. Outside the chain the
        # first and last move clamp to their own resting state, which is what
        # standing still before and after the sequence looks like.
        grid = np.atleast_1d(np.asarray(time, dtype=float))
        ends = np.array([tr.t_end for tr in self.trajectories])
        idx = np.clip(np.searchsorted(ends, grid, side="left"), 0, len(self) - 1)

        out = np.empty(grid.shape)
        for i in np.unique(idx):
            mask = idx == i
            out[mask] = getattr(self.trajectories[i], name)(grid[mask])
        return float(out[0]) if np.ndim(time) == 0 else out

    def jerk(self, time):
        """Jerk at absolute time(s) `time`."""
        return self._eval("jerk", time)

    def accel(self, time):
        """Acceleration at absolute time(s) `time`."""
        return self._eval("accel", time)

    def vel(self, time):
        """Velocity at absolute time(s) `time`."""
        return self._eval("vel", time)

    def pos(self, time):
        """Position at absolute time(s) `time`."""
        return self._eval("pos", time)

    def sample(self, n_points: int = 600):
        """Evaluate the whole chain on a time grid, for plotting.

        Returns (t, jerk, accel, vel, pos) as numpy arrays.
        """
        end = self.t_end if self.duration > 0.0 else self.t0 + 1.0
        grid = np.linspace(self.t0, end, n_points)
        return grid, self.jerk(grid), self.accel(grid), self.vel(grid), self.pos(grid)

    def plot(self, n_points: int = 600):
        _plot_profile(
            *self.sample(n_points),
            boundaries=[tr.t_end for tr in self.trajectories[:-1]],
        )


# ---------------------------------------------------------------------------
# Both axes at once
# ---------------------------------------------------------------------------


def _moves(chain):
    """The individual moves of a chain, or the single move of a trajectory."""
    return getattr(chain, "trajectories", (chain,))


class Trajectory2D:
    """A bridge motion and a trolley motion, run side by side.

    The bridge supplies the x coordinate and the trolley the y coordinate. The
    two need not cover the same time span: both are evaluated on one common
    time axis, and an axis that has already finished -- or has not started yet
    -- simply stands still while the other one moves.

    Either side may be a `Trajectory` or a `TrajectoryChain`.
    """

    def __init__(self, bridge, trolley):
        if bridge.component is trolley.component:
            raise ValueError("bridge and trolley must run on different axes")
        self.bridge = bridge
        self.trolley = trolley

    @classmethod
    def through(cls, points, bridge=None, trolley=None, t0: float = 0.0):
        """Build the move that starts and stops at each of `points`.

        Both axes set off together at the start of a leg, and the one that
        gets there first waits at the waypoint until the other arrives, so
        the load comes to a full stop on every point. Each leg therefore
        costs the slower of the two axes.

        `points` is a list of (bridge, trolley) tuples. The axes default to
        BRIDGE and TROLLEY.
        """
        bridge = BRIDGE if bridge is None else bridge
        trolley = TROLLEY if trolley is None else trolley

        waypoints = [(float(x), float(y)) for x, y in points]
        if len(waypoints) < 2:
            raise ValueError("need at least two points to move between")

        bridge_moves, trolley_moves = [], []
        (x, y), start = waypoints[0], float(t0)
        for next_x, next_y in waypoints[1:]:
            bridge_moves.append(bridge.trajectory(next_x - x, x, start))
            trolley_moves.append(trolley.trajectory(next_y - y, y, start))
            start += max(bridge_moves[-1].duration, trolley_moves[-1].duration)
            x, y = next_x, next_y

        return cls(TrajectoryChain(bridge_moves), TrajectoryChain(trolley_moves))

    @classmethod
    def through_merged(cls, points, bridge=None, trolley=None, t0: float = 0.0):
        """Build the move through `points` the way the PLC really drives it.

        `through` stops dead on every waypoint. The PLC does better: it looks
        at three points at a time and, where the geometry allows, lets one
        axis run straight through the middle one while the other makes its
        move in the shadow of that. The crane still passes exactly through
        every point, and each axis still crosses a leg without turning around,
        so a waypoint list that cleared the yard still clears it -- the load
        just does not come to a standstill in between.

        Three points at a time is a narrow view, so merging is not always a
        win: joining two legs can push the move past the n it needs to stay
        inside a_max, which widens all four of its pulses, and a joined move
        is held back far enough to still pass the middle point. Together those
        can cost a second or two more than the standstill they save. Rare --
        one route in a few hundred, and only ever by a little -- but worth
        knowing when this is what scores a route.

        `points` is a list of (bridge, trolley) tuples. The axes default to
        BRIDGE and TROLLEY.
        """
        bridge = BRIDGE if bridge is None else bridge
        trolley = TROLLEY if trolley is None else trolley

        waypoints = [(float(x), float(y)) for x, y in points]
        if len(waypoints) < 2:
            raise ValueError("need at least two points to move between")

        return cls(*_Merger((bridge, trolley), waypoints[0], t0).run(waypoints))

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}({self.start} -> {self.end}, "
            f"t0={self.t0} -> {self.t_end})"
        )

    @property
    def t0(self) -> float:
        """When the first of the two axes starts moving."""
        return min(self.bridge.t0, self.trolley.t0)

    @property
    def t_end(self) -> float:
        """When the last of the two axes finishes."""
        return max(self.bridge.t_end, self.trolley.t_end)

    @property
    def duration(self) -> float:
        """Total time the combined move takes."""
        return self.t_end - self.t0

    @property
    def start(self) -> tuple:
        """Where the move begins, as (bridge, trolley)."""
        return (self.bridge.x0, self.trolley.x0)

    @property
    def end(self) -> tuple:
        """Where the move ends, as (bridge, trolley)."""
        return (self.bridge.x_end, self.trolley.x_end)

    @property
    def handovers(self) -> list:
        """Times where either axis hands over from one move to the next.

        The last move of each axis ends the axis rather than handing over, so
        it is left out -- the same boundaries `TrajectoryChain.plot` draws.
        """
        moves = (*_moves(self.bridge)[:-1], *_moves(self.trolley)[:-1])
        return sorted({move.t_end for move in moves})

    @property
    def standstills(self) -> list:
        """When the load comes to a full stop, once for every time it does.

        A move begins and ends at zero velocity and an axis stands still
        between its moves, so the load is at rest exactly when neither axis
        has a move running. Two moves that touch leave no gap on the clock,
        but the load does stop between them, so a touch counts as much as a
        gap does -- that instant is every waypoint of a `through` move, and
        the ones `through_merged` could not merge away.

        The start and the end of the whole move are left out; the load is at
        rest there whatever it does in between.
        """
        spans = sorted(
            (move.t0, move.t_end)
            for move in (*_moves(self.bridge), *_moves(self.trolley))
            if move.duration > 0.0
        )

        stops, busy_until = [], None
        for start, end in spans:
            if busy_until is None or start < busy_until:
                # The very first move, or one that overlaps what is already
                # running: either way nothing has come to a stop.
                busy_until = end if busy_until is None else max(busy_until, end)
                continue
            stops.append(busy_until)   # everything stopped, and this starts it off again
            busy_until = end
        return stops

    def pos(self, time) -> tuple:
        """Position at absolute time(s) `time`, as (bridge, trolley)."""
        return (self.bridge.pos(time), self.trolley.pos(time))

    def sample(self, n_points: int = 600):
        """Evaluate both axes on a common time grid, for plotting.

        Returns (t, x, y) as numpy arrays.
        """
        end = self.t_end if self.duration > 0.0 else self.t0 + 1.0
        grid = np.linspace(self.t0, end, n_points)
        return (
            grid,
            np.asarray(self.bridge.pos(grid), dtype=float),
            np.asarray(self.trolley.pos(grid), dtype=float),
        )

    def plot(self, n_points: int = 600, ax=None):
        """Plot the path traced through the yard, coloured by time.

        Pass `ax` to draw onto existing axes (a yard outline, say); the axes
        are returned either way.
        """
        grid, x, y = self.sample(n_points)

        own_figure = ax is None
        if own_figure:
            _, ax = plt.subplots(figsize=(8, 5))

        # Colour the path by time -- in real space the clock is otherwise
        # invisible, and a corner where one axis waits for the other looks
        # like any other.
        points = np.stack([x, y], axis=1).reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        path = LineCollection(segments, cmap="viridis", array=grid[:-1], linewidth=2)
        ax.add_collection(path)
        ax.figure.colorbar(path, ax=ax, label="time (s)")

        stops = self.standstills
        if stops:
            ax.plot(*self.pos(stops), "o", color="grey", markersize=5,
                    label="standstill")
        ax.plot(x[0], y[0], "o", color="tab:green", label="start")
        ax.plot(x[-1], y[-1], "s", color="tab:red", label="end")

        ax.set_xlabel("bridge (m)")
        ax.set_ylabel("trolley (m)")
        ax.set_aspect("equal")
        ax.autoscale_view()
        ax.grid()
        ax.legend()
        if own_figure:
            plt.tight_layout()
            plt.show()
        return ax


# ---------------------------------------------------------------------------
# Merging
# ---------------------------------------------------------------------------
#
# Translated from mergingcode/func_merging.m, with three transcription slips
# in it put right on the way: func_n is handed a_max where t_sway belongs;
# jerkdistinv is handed the auxiliary axis's index where the main axis's jerk
# belongs, and that axis's distance too, when the value wanted is the moveP2.ti
# computed two lines below; and the branch that cannot merge out of a
# standstill leaves Flag_standstill clear although the crane does stop there.


def _lone_axis(a, b) -> Optional[int]:
    """The one axis that gets the crane from `a` to `b`, if there is one.

    0 is the bridge and 1 the trolley, in the order they appear in a waypoint.
    """
    if a[0] == b[0]:
        return 1
    if a[1] == b[1]:
        return 0
    return None


def _runs_through(axis: int, p1, p2, p3) -> bool:
    """Can one axis take p1 -> p2 -> p3 in a single move?

    Only if it does not turn around on the way. Merging sizes the joined move
    on the distance from p1 to p3, and that is the distance actually travelled
    only when both legs push the axis the same way. (The MATLAB skips the
    check -- it is the sign checking left as a TODO in examplecheck.m -- and a
    crane that doubles back would be given far too short a move.)
    """
    step, ahead = p2[axis] - p1[axis], p3[axis] - p2[axis]
    return step * ahead >= 0.0 and step + ahead != 0.0


class _Ongoing:
    """The main-axis move the crane is in the middle of at a waypoint.

    `moveP1` in the MATLAB: which axis carries the move, the move itself, and
    `ti`, how far into it the crane is as it passes the waypoint.
    """

    def __init__(self, axis: int, move: Trajectory):
        self.axis = axis
        self.move = move
        self.ti = 0.0

    def reach(self, position: float):
        """Note that the crane passes the waypoint at `position`."""
        self.ti = self.move.time_at(position) - self.move.t0

    def extend(self, distance: float):
        """Stretch the cruise phase to carry the move `distance` further.

        The four pulses are left alone, so the peak acceleration and velocity
        -- and with them a_max, v_max and the anti-sway width t_p = n*t_sway --
        are still the ones the move was sized with. Only the cruise grows.
        """
        move = self.move
        self.move = Trajectory.from_shape(
            move.component,
            move.j,
            move.tp,
            move.lam + abs(distance) / abs(move.vp),
            move.x0,
            move.t0,
        )

    def delay(self, wait: float):
        """Push the whole move `wait` later; `ti` runs from its own start."""
        self.move.t0 += wait

    @property
    def time_left(self) -> float:
        """What is left to run as the crane passes the waypoint (time_remi)."""
        return self.move.duration - self.ti

    @property
    def braking(self) -> bool:
        """Is the move already on its way down at the waypoint? (Flag_dec)"""
        return self.ti > self.move.lam + 2.0 * self.move.tp


class _Merger:
    """The PLC's merging strategy, played out over a list of waypoints.

    The PLC only ever sees three points at a time. Standing at p1 with p2 and
    p3 ahead it asks whether the axis carrying the crane into p2 can run
    straight through and on towards p3: two moves become one, and the
    standstill in between disappears. Whatever it decides, the state at p2 is
    handed on and the same question asked of (p2, p3, p4).

    The axis carrying the joined move is the main axis (`i` in the MATLAB);
    the other one (`j`) makes a plain standstill-to-standstill move in the
    shadow of it. Merging is only allowed when there is room for that move and
    for the main axis's own braking on top, which is what keeps the crane
    passing exactly through every waypoint instead of cutting the corner.
    """

    def __init__(self, components, start, t0: float = 0.0):
        self.components = tuple(components)
        self.moves = ([], [])       # what each axis has been given to do
        self.x = list(start)        # where each axis stands at the current point
        self.t0 = float(t0)
        self.t = float(t0)          # when the crane passes the current point
        self.times = [float(t0)]    # ... for every point walked so far
        self.ongoing = None         # the main-axis move still running, if any

    def run(self, points) -> tuple:
        """Walk the whole list, and hand back the two axes' chains."""
        for p1, p2, p3 in zip(points, points[1:], points[2:]):
            self._step(p1, p2, p3)
            self.times.append(self.t)
        # There is nothing to look ahead to on the last leg, so the crane
        # stops -- which is where the main-axis move was headed anyway.
        self._stop_at(points[-2], points[-1])
        self.times.append(self.t)
        return tuple(
            # An axis that never had to move still owes the chain one move.
            TrajectoryChain(moves or [component.trajectory(0.0, x, self.t0)])
            for moves, component, x in zip(self.moves, self.components, self.x)
        )

    # -- handing work to an axis --------------------------------------------

    def _move(self, axis: int, distance: float, t0: float) -> float:
        """Give `axis` a standstill-to-standstill move; how long does it take?"""
        move = self.components[axis].trajectory(distance, self.x[axis], t0)
        if move.duration > 0.0:
            self.moves[axis].append(move)
        self.x[axis] = move.x_end
        return move.duration

    def _start(self, axis: int, target: float, t0: float) -> _Ongoing:
        """Set the main axis off on a move running all the way to `target`."""
        move = self.components[axis].trajectory(
            target - self.x[axis], self.x[axis], t0
        )
        self.ongoing = _Ongoing(axis, move)
        return self.ongoing

    def _land(self) -> float:
        """Let the main-axis move run out; how much of it was left?"""
        run, self.ongoing = self.ongoing, None
        self.moves[run.axis].append(run.move)
        self.x[run.axis] = run.move.x_end
        return run.time_left

    # -- one look at three points -------------------------------------------

    def _step(self, p1, p2, p3):
        """Move from p1 to p2, with p3 in view."""
        if self.ongoing is None:
            self._from_standstill(p1, p2, p3)
        else:
            self._carry_on(p1, p2, p3)

    def _from_standstill(self, p1, p2, p3):
        """The crane is at rest on p1. Can it leave p2 without stopping?"""
        main = _lone_axis(p1, p2)
        if main is not None and _runs_through(main, p1, p2, p3):
            # The leg into p2 needs one axis only, so that axis carries on
            # through p2 towards p3 in a single move, with no acceleration or
            # deceleration in between, and the other one waits at p1 until the
            # crane reaches p2.
            run = self._start(main, p3[main], self.t)
            run.reach(p2[main])
            self.t += run.ti
            return

        main = _lone_axis(p2, p3)
        if main is not None and _runs_through(main, p1, p2, p3):
            # The leg out of p2 needs one axis only, so that axis again takes
            # both legs in one move -- held back long enough that it is still
            # passing p2 as the other axis arrives there.
            other = 1 - main
            run = self._start(main, p3[main], self.t)
            run.reach(p2[main])
            leg = self._move(other, p2[other] - p1[other], self.t)
            run.delay(max(0.0, leg - run.ti))
            self.t += max(run.ti, leg)
            return

        # Neither leg is straight enough to join onto the next one.
        self._stop_at(p1, p2)

    def _carry_on(self, p1, p2, p3):
        """The crane passes p1 mid-move. Can that move swallow p2 as well?"""
        run = self.ongoing
        main, other = run.axis, 1 - run.axis
        step = p2[main] - p1[main]      # what the main axis does on this leg
        ahead = p3[main] - p2[main]     # ... and what it does on the next one
        leg = self.components[other].trajectory(p2[other] - p1[other]).duration

        if (
            not run.braking                          # not yet slowing down, and
            and step * ahead > 0.0                   # not about to turn around,
            and run.time_left - leg >= 2.0 * run.move.tp   # and time to spare
        ):
            # All three hold, so the move is stretched to reach p3 as well:
            # the cruise takes over the distance the braking would have
            # covered, and the other axis makes its leg while that happens.
            self._move(other, p2[other] - p1[other], self.t)
            was = run.ti
            run.extend(ahead)
            run.reach(p2[main])
            self.t += run.ti - was
            return

        # Merging stops here: the main axis brakes to a standstill on p2,
        # which is where its move was always going to end.
        self._stop_at(p1, p2)

    def _stop_at(self, p1, p2):
        """Come to a standstill on p2: whatever is running finishes there."""
        if self.ongoing is None:
            legs = [self._move(axis, p2[axis] - p1[axis], self.t) for axis in (0, 1)]
        else:
            other = 1 - self.ongoing.axis
            legs = [self._land(), self._move(other, p2[other] - p1[other], self.t)]
        self.t += max(legs)


# Axis limits from toy_example.m
BRIDGE = Component(a_max=0.30, v_max=2, t_sway=4)
TROLLEY = Component(a_max=0.25, v_max=1, t_sway=5)


if __name__ == "__main__":
    # Routes given as (bridge, trolley) waypoints, driven both ways. `through`
    # comes to a full stop on every point: both axes set off together at the
    # start of a leg and the faster of the two waits at the waypoint for the
    # slower to arrive. `through_merged` drives them the way the PLC does,
    # running an axis straight through a waypoint where it can.
    #
    # The first route gains nothing: the trolley goes up to 14 and back down
    # to 6, and an axis that turns around has to stop to do it. On the second
    # the bridge never turns around, so it takes all three legs in one move.
    for points in ([(0, 12), (58, 14), (58, 6)],
                   [(0, 12), (40, 14), (55, 14), (58, 6)]):
        move = Trajectory2D.through(points)
        merged = Trajectory2D.through_merged(points)
        print(points)
        print(f"  stopping on every point: {move.duration:6.1f} s")
        print(f"  merged:                  {merged.duration:6.1f} s")
    merged.plot()

