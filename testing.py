from plc import Trajectory2D

# move = Trajectory2D.through([(0, 12), (32, 12), (40, 6), (58, 6)])
# move.plot()
# print(move.t_end)

move = Trajectory2D.through([(0, 12), (26, 0),  (58, 6)], t0=0.0)
move.plot()
move.trolley.plot()

