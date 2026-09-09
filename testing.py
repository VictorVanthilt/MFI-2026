from plc import Trajectory2D
from test_paths import Yard, Point, Rect, plot_route

# Toy example
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
forbidden3 = Rect(Point(20, 0), Point(40, 2))
obstacles = [forbidden1, forbidden2]

path = [(0, 12), (24, 2), (40, 2), (58, 6)]
move = Trajectory2D.through_merged(path)
plot_route(yard, obstacles, move)

path = [(0, 12), (24, 0), (40, 0), (58, 6)]
move = Trajectory2D.through_merged(path)
plot_route(yard, obstacles, move)

path = [(0, 12), (58, 6)]
move = Trajectory2D.through_merged(path)
plot_route(yard, obstacles, move)


# move.plot()
# move.bridge.plot()
# move.trolley.plot()
print(move.t_end)
# move.plot()
