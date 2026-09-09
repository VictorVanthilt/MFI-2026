from plc import Trajectory2D

path1 = [(0, 12), (24, 2), (40, 2), (58, 6)]
move = Trajectory2D.through(path1)
print(move.t_end)
move.plot()

move = Trajectory2D.through_merged(path1)
print(move.t_end)
move.plot()

# path = [(0, 12), (26, 2), (32, 2), (58, 6)]
# move = Trajectory2D.through_merged(path)
# move.plot()
# move.bridge.plot()
# move.trolley.plot()
# print(move.t_end)
# move.plot()
