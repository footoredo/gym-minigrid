from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from gym import spaces

class MemoryEnv(MiniGridEnv):
    """
    This environment is a memory test. The agent starts in a small room
    where it sees an object. It then has to go through a narrow hallway
    which ends in a split. At each end of the split there is an object,
    one of which is the same as the object in the starting room. The
    agent has to remember the initial object, and go to the matching
    object at split.
    """

    def __init__(
        self,
        seed,
        size=8,
        random_length=False,
        easy_mode=0,
        use_shape=True,
        max_steps=None
    ):
        self.random_length = random_length
        self.easy_mode = easy_mode
        """
            1: face towards hint
            2: colored hint
            4: doubled hint
        """
        self.use_shape = use_shape
        task_obs_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.int32)
        super().__init__(
            seed=seed,
            grid_size=size,
            max_steps=5*size**2 if max_steps is None else max_steps,
            # Set this to True for maximum speed
            see_through_walls=False,
            extra_obs_spaces={
                "target": task_obs_space
            }
        )

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.horz_wall(0, 0)
        self.grid.horz_wall(0, height-1)
        self.grid.vert_wall(0, 0)
        self.grid.vert_wall(width - 1, 0)

        assert height % 2 == 1
        upper_room_wall = height // 2 - 2
        lower_room_wall = height // 2 + 2
        if self.random_length:
            hallway_end = self._rand_int(4, width - 2)
        else:
            hallway_end = width - 3

        # Start room
        for i in range(1, 5):
            self.grid.set(i, upper_room_wall, Wall())
            self.grid.set(i, lower_room_wall, Wall())
        self.grid.set(4, upper_room_wall + 1, Wall())
        self.grid.set(4, lower_room_wall - 1, Wall())

        # Horizontal hallway
        for i in range(5, hallway_end):
            self.grid.set(i, upper_room_wall + 1, Wall())
            self.grid.set(i, lower_room_wall - 1, Wall())

        # Vertical hallway
        for j in range(0, height):
            if j != height // 2:
                self.grid.set(hallway_end, j, Wall())
            self.grid.set(hallway_end + 2, j, Wall())

        # Fix the player's start position and orientation
        self.agent_pos = (self._rand_int(1, hallway_end + 1), height // 2)
        self.agent_dir = 2 if self.easy_mode & 1 else 0

        if self.use_shape:
            # objects = [lambda: Key('green'), lambda: Ball('green')] 
            objects = [Ball, Key]
            colors = ['green', 'green']
        else:
            # objects = [lambda: Ball('yellow'), lambda: Ball('green')] 
            objects = [Ball, Ball]
            colors = ['green', 'red']

        # Place objects
        start_room_obj_id = self._rand_int(0, 2)
        start_room_obj = objects[start_room_obj_id]
        if self.use_shape:
            if self.easy_mode & 2:
                start_room_obj_color = 'blue'
            else:
                start_room_obj_color = 'green'
        else:
            start_room_obj_color = colors[start_room_obj_id]
        # start_room_obj_color = 'red' if self.easy_mode else 'green'
        # start_room_obj_color = 'green'
        self.grid.set(1, height // 2 - 1, start_room_obj(start_room_obj_color))
        if self.easy_mode & 4:
            self.grid.set(1, height // 2 + 1, start_room_obj(start_room_obj_color))

        other_objs_id = self._rand_elem([[0, 1], [1, 0]])
        # other_objs = self._rand_elem([[Ball, Key], [Key, Ball]])
        pos0 = (hallway_end + 1, height // 2 - 2)
        pos1 = (hallway_end + 1, height // 2 + 2)
        self.grid.set(*pos0, objects[other_objs_id[0]](colors[other_objs_id[0]]))
        self.grid.set(*pos1, objects[other_objs_id[1]](colors[other_objs_id[1]]))

        # Choose the target objects
        if start_room_obj_id == other_objs_id[0]:
            self.success_pos = (pos0[0], pos0[1] + 1)
            self.failure_pos = (pos1[0], pos1[1] - 1)
        else:
            self.success_pos = (pos1[0], pos1[1] - 1)
            self.failure_pos = (pos0[0], pos0[1] + 1)

        self.mission = 'go to the matching object at the end of the hallway'
        self.target = start_room_obj_id

    def reset(self):
        obs = super().reset()
        obs['target'] = np.array([self.target], dtype=np.int32)
        return obs

    def step(self, action):
        if action == MiniGridEnv.Actions.pickup:
            action = MiniGridEnv.Actions.toggle
        obs, reward, done, info = MiniGridEnv.step(self, action)

        obs["target"] = np.array([self.target], dtype=np.int32)

        if tuple(self.agent_pos) == self.success_pos:
            reward = self._reward()
            done = True
        if tuple(self.agent_pos) == self.failure_pos:
            reward = 0
            done = True

        return obs, reward, done, info

class MemoryS17Random(MemoryEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(seed=seed, size=17, random_length=True, **kwargs)

register(
    id='MiniGrid-MemoryS17Random-v0',
    entry_point='gym_minigrid.envs:MemoryS17Random',
)

class MemoryS13Random(MemoryEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(seed=seed, size=13, random_length=True, **kwargs)

register(
    id='MiniGrid-MemoryS13Random-v0',
    entry_point='gym_minigrid.envs:MemoryS13Random',
)

class MemoryS13(MemoryEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(seed=seed, size=13, **kwargs)

register(
    id='MiniGrid-MemoryS13-v0',
    entry_point='gym_minigrid.envs:MemoryS13',
)

class MemoryS11(MemoryEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(seed=seed, size=11, **kwargs)

register(
    id='MiniGrid-MemoryS11-v0',
    entry_point='gym_minigrid.envs:MemoryS11',
)

class MemoryS9(MemoryEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(seed=seed, size=9, **kwargs)

register(
    id='MiniGrid-MemoryS9-v0',
    entry_point='gym_minigrid.envs:MemoryS9',
)

class MemoryS7(MemoryEnv):
    def __init__(self, seed=None, **kwargs):
        super().__init__(seed=seed, size=7, **kwargs)

register(
    id='MiniGrid-MemoryS7-v0',
    entry_point='gym_minigrid.envs:MemoryS7',
)
