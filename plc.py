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

        handovers = self.handovers
        if handovers:
            ax.plot(*self.pos(handovers), "o", color="grey", markersize=4,
                    label="handover")
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


# Axis limits from toy_example.m
BRIDGE = Component(a_max=0.30, v_max=2, t_sway=4)
TROLLEY = Component(a_max=0.25, v_max=1, t_sway=5)


if __name__ == "__main__":
    # A route given as (bridge, trolley) waypoints. The load comes to a full
    # stop on each one: both axes set off together at the start of a leg and
    # the faster of the two waits at the waypoint for the slower to arrive.
    points = [(0, 12), (58, 14), (58, 6)]
    move = Trajectory2D.through(points)
    print(move)
    print(f"total time: {move.duration} s")
    move.plot()

