"""Anti-sway crane motion profile, derived symbolically.

The load hangs on a cable, so it is a pendulum of period `t_sway`. Shaping the
acceleration as four equal jerk pulses, each exactly `tp = n * t_sway` long.

"""

import math
from typing import NamedTuple, Optional

import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


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


# ---------------------------------------------------------------------------
# One axis
# ---------------------------------------------------------------------------


class Profile(NamedTuple):
    """The concrete profile chosen for one move."""

    time: float      # total duration, 4*tp + lambda
    n: int           # sway periods per jerk pulse
    tp: float        # jerk pulse width, n * t_sway
    j: float         # jerk
    ap: float        # peak acceleration, j * tp
    vp: float        # peak (cruise) velocity, j * tp**2
    lam: float       # cruise duration
    saturated: bool  # True if the move is long enough to reach v_max


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

    def n_max(self, distance) -> int:
        """Sway periods needed for the accel phase to reach v_max.

        Beyond this the velocity saturates and a cruise phase is required.
        """
        return math.ceil(self.v_max / (self.t_sway * self.a_max))

    @property
    def min_time(self) -> float:
        """Shortest possible move: one sway period per jerk pulse."""
        return 4.0 * self.t_sway

    def n(self, distance: float) -> float:
        """Time to move `distance` along this axis."""
        return math.ceil(np.sqrt(abs(distance) / (2 * self.a_max * self.t_sway**2)))

    def profile(self, distance: float) -> Profile:
        """Choose the profile covering `distance`."""
        distance = abs(float(distance))
        n, n_max = self.n(distance), self.n_max(distance)

        if n == 0:  # standing still
            return Profile(0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, False)

        if n < n_max:
            # Short move: velocity never saturates, so there is no cruise and
            # the jerk is whatever covers the distance in four pulses.
            tp = self.t_sway * n
            j = distance/(2 * tp**3)
            ap = j * tp
            vp = j * tp**2
            lam = 0.0
            saturated = False
            time = 4.0 * n * self.t_sway
        else:
            # Long move: clamp to n_max, run at v_max, and cruise for the rest.
            n = n_max
            tp = self.t_sway * n
            ap = self.v_max / tp
            vp = ap * tp
            j = ap / tp    
            lam = float(SYM.lam.subs({d_s: distance, j_s: j, tp_s: tp}))
            lam, saturated = max(lam, 0.0), True

        return Profile(
            time=4.0 * tp + lam,
            n=n,
            tp=tp,
            j=j,
            ap = ap, 
            vp = vp,
            lam=lam,
            saturated=saturated,
        )

    def time(self, distance: float) -> float:
        """Time to move `distance` along this axis."""
        return self.profile(distance).time

    def sample(self, distance: float, n_points: int = 600):
        """Evaluate the profile on a time grid, for plotting.

        Returns (t, jerk, accel, vel, pos) as numpy arrays.
        """
        prof = self.profile(distance)
        subs = {j_s: prof.j, tp_s: prof.tp, lam_s: prof.lam}
        grid = np.linspace(0.0, prof.time or 1.0, n_points)

        out = [grid]
        for expr in (SYM.jerk, SYM.accel, SYM.vel, SYM.pos):
            f = sp.lambdify(t, expr.subs(subs), "numpy")
            # A degenerate profile can collapse to a constant, which lambdify
            # returns as a scalar rather than an array.
            values = np.asarray(f(grid), dtype=float)
            out.append(np.broadcast_to(values, grid.shape).copy())
        return tuple(out)
    
    def plot_dist_time(self, ax):
        dist_range = np.linspace(0, 100, 200)
        time_values = [self.time(d) for d in dist_range]
        ax.plot(dist_range, time_values, label="Bridge")
       
    
    def plot(self, distance: float, n_points: int = 600):
        fig, ax = plt.subplots(4, 1, sharex=True, figsize=(8, 6))
        t, jerk, accel, vel, pos = self.sample(distance, n_points)
        ax[0].plot(t, jerk, label="jerk")
        ax[1].plot(t, accel, label="accel")
        ax[2].plot(t, vel, label="vel")
        ax[3].plot(t, pos, label="pos")
        ax[0].set_ylabel("jerk (m/s³)")
        ax[1].set_ylabel("accel (m/s²)")
        ax[2].set_ylabel("vel (m/s)")
        ax[3].set_ylabel("pos (m)")
        ax[3].set_xlabel("time (s)")
        for a in ax:
            a.grid()
            a.legend()
        plt.tight_layout()
        plt.show()


# Axis limits from toy_example.m
BRIDGE = Component(a_max=0.30, v_max=2, t_sway=4)
TROLLEY = Component(a_max=0.25, v_max=1, t_sway=5)
