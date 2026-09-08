"""Python translation of the MATLAB anti-sway motion timing model.

Translated from:
  - func_n.m      -> func_n()
  - toy_example.m -> toy_example()

The MATLAB function signature is [time, j, tp, lambda, vp] = func_n(...).
In the first branch (n < n_max) MATLAB leaves j, tp, lambda and vp unassigned,
so they are returned as None here to mirror that behaviour.
"""

import math
from typing import NamedTuple, Optional


class FuncNResult(NamedTuple):
    """Outputs of func_n, in the same order as the MATLAB return list."""

    time: float
    j: Optional[float] = None
    tp: Optional[float] = None
    lambda_: Optional[float] = None
    vp: Optional[float] = None


def func_n(a_max: float, v_max: float, t_sway: float, distance: float) -> FuncNResult:
    """Travel time for an anti-sway move over `distance`.

    a_max    -- maximum acceleration
    v_max    -- maximum velocity
    t_sway   -- sway (pendulum) period
    distance -- distance to travel
    """
    n_max = math.ceil(v_max / (t_sway * a_max))
    n = math.ceil(math.sqrt(distance / (2 * a_max * t_sway**2)))

    if n < n_max:
        # j = distance / (2 * tp**3)
        # ap = j * tp
        # vp = tp * ap
        time = 4 * n * t_sway
        return FuncNResult(time)

    n = n_max
    tp = t_sway * n
    ap = v_max / tp
    vp = ap * tp
    j = ap / tp
    j_bound = min(a_max / tp, v_max / tp**2)
    if j > j_bound:
        raise ValueError("here1")

    lambda_ = (distance - 2 * j * tp**3) / (j * tp**2)
    if lambda_ < 0:
        lambda_ = 0
    time = lambda_ + 4 * tp

    return FuncNResult(time, j, tp, lambda_, vp)


def toy_example() -> float:
    """Toy example -- Bridge: 0->58."""
    # bridge:
    a_max, v_max, t_sway, distance = 0.30, 2, 4, 58
    time1 = func_n(a_max, v_max, t_sway, distance).time
    print(f"time1 = {time1}")

    # trolley:
    a_max, v_max, t_sway, distance = 0.25, 1, 5, 6
    time2 = func_n(a_max, v_max, t_sway, distance).time
    print(f"time2 = {time2}")

    total = time1 + time2
    print(f"time1 + time2 = {total}")
    return total


if __name__ == "__main__":
    toy_example()
