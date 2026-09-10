"""
Choose one waypoint per overlap region to minimise the travel time from
start to goal, then plot the result.

Needs max_x_path.py and path_overlaps.py in the same folder.

If plc / test_paths are importable, the real Trajectory2D is used.
If not, a crude surrogate timing model kicks in so the rest of the
pipeline (GA, plotting) can still be exercised — the printout says which.
"""

import numpy as np
import matplotlib.pyplot as plt

from path_overlaps import build_population, plot_layout

rng = np.random.default_rng()

# ----------------------------------------------------------------------
# timing back end
# ----------------------------------------------------------------------
try:
    from plc import Trajectory2D
    from test_paths import plot_route
    REAL_TIMING = True
except ImportError:
    Trajectory2D, plot_route = None, None
    REAL_TIMING = False


def total_time(move):
    for name in ("total_time", "T", "time", "duration", "t_total", "tf"):
        if hasattr(move, name):
            val = getattr(move, name)
            return float(val() if callable(val) else val)
    public = [a for a in dir(move) if not a.startswith("_")]
    raise AttributeError(
        "Could not find the total time on the Trajectory2D object.\n"
        f"Available attributes: {public}")


def _surrogate_time(path, v_x=1.0, v_y=0.5, t_settle=4.0):
    """Stand-in only. Axis-decoupled travel plus a settling time per segment."""
    P = np.asarray(path, float)
    d = np.abs(np.diff(P, axis=0))
    return float(np.sum(np.maximum(d[:, 0] / v_x, d[:, 1] / v_y) + t_settle))

from plc import Trajectory2D, Component

BRIDGE_LM  = Component(a_max=192/1000, v_max=1920/1000, t_sway=5625/1000)
TROLLEY_LM = Component(a_max=162/1000, v_max=811/1000,  t_sway=6049/1000)

def travel_time(path, bridge=BRIDGE_LM, trolley=TROLLEY_LM):
    if REAL_TIMING:
        return total_time(Trajectory2D.through_merged(path,
                                                      bridge=bridge,
                                                      trolley=trolley))
    return _surrogate_time(path)

#def travel_time(path):
  #  """Total time for a list of (x, y) tuples."""
   # if REAL_TIMING:
   #     return total_time(Trajectory2D.through_merged(path))
  #  return _surrogate_time(path)


# ----------------------------------------------------------------------
# path helpers
# ----------------------------------------------------------------------
def make_path(p_start, p_goal, mid):
    mid = np.asarray(mid, float).reshape(-1, 2)
    return ([tuple(map(float, p_start))]
            + [(float(x), float(y)) for x, y in mid]
            + [tuple(map(float, p_goal))])


def path_length(path):
    P = np.asarray(path, float)
    return float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1)))


class Objective:
    """
    Travel time, with two additions over the raw value.

    eps * path_length : a tie-break. If through_merged quantises the time
        (your trials returned exactly 84.0 and 96.0 from different
        waypoints), the landscape has flat plateaus and selection cannot
        rank two individuals sitting on one. The length term gives a slope
        to descend, scaled small enough never to override a real difference.

    cache : identical or near-identical chromosomes are common once the
        population converges, and each evaluation is expensive.
    """

    def __init__(self, p_start, p_goal, eps=1e-3, decimals=6):
        self.p_start, self.p_goal = p_start, p_goal
        self.eps, self.decimals = eps, decimals
        self.cache = {}
        self.n_calls = 0

    def one(self, row):
        key = tuple(np.round(row, self.decimals))
        if key in self.cache:
            return self.cache[key]
        path = make_path(self.p_start, self.p_goal, np.asarray(row).reshape(-1, 2))
        try:
            T = travel_time(path)
            score = T + self.eps * path_length(path)
        except Exception:
            T, score = np.inf, np.inf
        self.n_calls += 1
        self.cache[key] = (T, score)
        return T, score

    def __call__(self, P):
        return np.array([self.one(row)[1] for row in np.atleast_2d(P)])

    def raw_time(self, row):
        return self.one(row)[0]


# ----------------------------------------------------------------------
# GA
# ----------------------------------------------------------------------
def tournament(pop, fit, k=3):
    idx = rng.integers(0, len(pop), size=(len(pop), k))
    return pop[idx[np.arange(len(pop)), np.argmin(fit[idx], axis=1)]]


def crossover(parents, alpha=0.5, pc=0.9):
    kids = parents.copy()
    for i in range(0, len(parents) - 1, 2):
        if rng.random() < pc:
            lo = np.minimum(parents[i], parents[i + 1])
            hi = np.maximum(parents[i], parents[i + 1])
            d = hi - lo
            kids[i] = rng.uniform(lo - alpha * d, hi + alpha * d)
            kids[i + 1] = rng.uniform(lo - alpha * d, hi + alpha * d)
    return kids


def mutate(pop, gen, n_gen, lo, hi, sigma0=0.35, sigma_min=2e-4,
           pm=0.4, p_jump=0.03):
    """
    Per-gene bounds, so each waypoint stays inside its own overlap rectangle.
    Clipping (not reflection) is deliberate: the optimum here sits ON the
    boundary y = 2, and clipping is what puts genes exactly there.
    """
    span = hi - lo
    t = gen / max(n_gen - 1, 1)
    scale = sigma0 * (sigma_min / sigma0) ** t

    mask = rng.random(pop.shape) < pm
    pop = pop + mask * rng.normal(0.0, 1.0, pop.shape) * (scale * span)

    jump = rng.random(pop.shape) < p_jump
    pop = np.where(jump, lo + span * rng.random(pop.shape), pop)

    return np.clip(pop, lo, hi)


def corner_scan(obj, ovl):
    """Brute force over the corners of every overlap rectangle (4^m paths)."""
    from itertools import product
    ovl = np.asarray(ovl, float)
    corners = [[(r[0], r[2]), (r[0], r[3]), (r[1], r[2]), (r[1], r[3])]
               for r in ovl]
    best_row, best_score = None, np.inf
    for combo in product(*corners):
        row = np.array(combo, float).ravel()
        score = obj(row)[0]
        if score < best_score:
            best_row, best_score = row, score
    return best_row, obj.raw_time(best_row)


def ga_path(obj, lo, hi, n_pop=40, n_gen=60, seed_rows=None, verbose=True):
    n_dim = len(lo)
    pop = lo + (hi - lo) * rng.random((n_pop, n_dim))
    if seed_rows is not None:
        seed = np.atleast_2d(seed_rows)
        pop[:len(seed)] = np.clip(seed, lo, hi)

    fit = obj(pop)
    history = []
    for g in range(n_gen):
        elite, elite_f = pop[np.argmin(fit)].copy(), fit.min()
        child = mutate(crossover(tournament(pop, fit)), g, n_gen, lo, hi)
        pop, fit = child, obj(child)
        j = np.argmax(fit)
        pop[j], fit[j] = elite, elite_f
        history.append(obj.raw_time(pop[np.argmin(fit)]))
        if verbose and (g % 10 == 0 or g == n_gen - 1):
            print(f"  gen {g:3d}   best time {history[-1]:.4f}"
                  f"   evaluations so far {obj.n_calls}")
    best_row = pop[np.argmin(fit)]
    return best_row, obj.raw_time(best_row), history


def ga_restarts(obj, lo, hi, n_restarts=3, **kw):
    best_row, best_T, best_hist = None, np.inf, None
    for r in range(n_restarts):
        print(f"restart {r}:")
        row, T, hist = ga_path(obj, lo, hi, **kw)
        if T < best_T:
            best_row, best_T, best_hist = row, T, hist
    return best_row, best_T, best_hist


# ----------------------------------------------------------------------
if __name__ == "__main__":
    np.set_printoptions(suppress=True)

    from polygon_to_rects import polygon_to_box_obs

    yard_LM = np.array([
        (1488100, 793400), (1488100, 817700), (1580829, 817700), (1580829, 813229),
        (1628171, 813229), (1628171, 817700), (1672282, 817700), (1672282, 811556),
        (1694804, 811556), (1694804, 817700), (1906500, 817700), (1906500, 812501),
        (1776371, 812501), (1776371, 813171), (1727829, 813171), (1727829, 812501),
        (1711324, 812501), (1711324, 799371), (1662229, 799371), (1662229, 795028),
        (1660571, 795028), (1660571, 799371), (1615329, 799371), (1615329, 795028),
        (1615234, 795028), (1615234, 795093), (1596333, 795093), (1596333, 793400),
    ], dtype=float) / 1000.0

    yard, obstacles = polygon_to_box_obs(yard_LM)
    p_start = [1810399 / 1000, 813012 / 1000]
    p_goal  = [1553800 / 1000, 817400 / 1000]
    Lmin    = 5.0

    #print("timing back end:", ...)   # rest unchanged from here
#----------------------------




    # ------------------------------------------------------------------

    print("timing back end:",
          "Trajectory2D" if REAL_TIMING else "SURROGATE (plc not importable)")

    out = build_population(yard, obstacles, p_start, p_goal, Lmin, n_paths=1)
    ovl, lo, hi = out["ovl"], out["lo"], out["hi"]
    print("\noverlap regions [x1 x2 y1 y2]:\n", ovl)
    print(f"-> {len(ovl)} waypoints, {len(lo)} genes")
    print("lo:", lo)
    print("hi:", hi)

    # ---- figure 1: geometry only, no path ----------------------------
    plot_layout(yard, obstacles, p_start, p_goal,
                out["path_rects"], ovl, Lmin,
                waypoints=None, save="layout.png")
    plt.show(block=False)      # draw without blocking the script
    plt.pause(0.1)             # flush the draw, or the window stays blank

    input("\nGeometry shown. Press Enter to run the optimisation...")

    obj = Objective(p_start, p_goal)

    c_row, c_T = corner_scan(obj, ovl)
    print(f"\ncorner scan  : {np.round(c_row.reshape(-1, 2), 3).tolist()}"
          f"   time {c_T:.4f}   ({obj.n_calls} evaluations)")

    print("\nGA (seeded with the corner-scan winner):")
    best_row, best_T, hist = ga_restarts(obj, lo, hi, n_restarts=3,
                                         n_pop=40, n_gen=60,
                                         seed_rows=c_row)

    path = make_path(p_start, p_goal, best_row.reshape(-1, 2))

    print("\n" + "=" * 58)
    print("decided path")
    for i, (x, y) in enumerate(path):
        tag = ("start" if i == 0 else
               "goal" if i == len(path) - 1 else f"waypoint {i - 1}")
        print(f"  {tag:<12s} ({x:9.4f}, {y:9.4f})")
    print(f"minimum travel time : {best_T:.4f}")
    print(f"total evaluations   : {obj.n_calls}")
    print("=" * 58)

    # ---- figure 2: same geometry plus the chosen path ----------------
    ax = plot_layout(yard, obstacles, p_start, p_goal,
                     out["path_rects"], ovl, Lmin,
                     waypoints=path)          # new figure, ax=None

    mid = np.asarray(path[1:-1], float)
    ax.plot(mid[:, 0], mid[:, 1], "o", color="yellow", markersize=11,
            markeredgecolor="k", markeredgewidth=1.5, zorder=9)
    for i, (x, y) in enumerate(mid):
        ax.annotate(f"w{i}\n({x:.2f}, {y:.2f})", (x, y),
                    textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=8, fontweight="bold", zorder=9)
    ax.set_title(f"{len(out['path_rects'])} rectangles, "
                 f"$L_{{min}}$ = {Lmin:g},  best time = {best_T:.3f}")
    ax.figure.savefig("best_path.png", dpi=150, bbox_inches="tight")

    plt.show()