from gym_minigrid.minigrid import Ball
from gym_minigrid.roomgrid import RoomGrid
from gym_minigrid.register import register
import numpy as np

class UnlockEnhanced(RoomGrid):
    """
    Unlock a door
    """

    def __init__(self, seed=None, room_size=16, num_keys=10, max_steps=None, hide_key=False, start_with_key=False, reward_decay=True):
        self.num_keys = num_keys
        self.start_with_key = start_with_key
        self.reward_decay = reward_decay
        super().__init__(
            num_rows=1,
            num_cols=2,
            room_size=room_size,
            max_steps=8*room_size**2 if max_steps is None else max_steps,
            hide_carrying=hide_key,
            seed=seed
        )

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        # Make sure the two rooms are directly connected by a locked door
        door, _ = self.add_door(0, 0, 0, locked=True)
        # Add a key to unlock the door
        self.keys = []
        for _ in range(self.num_keys):
            key, pos = self.add_object(0, 0, 'key', door.color)
            self.keys.append((key, pos))

        self.place_agent(0, 0)

        if self.start_with_key:
            key, pos = self.keys[0]
            self.carrying = key
            self.carrying.cur_pos = np.array([-1, -1])
            self.grid.set(*pos, None)
        # print(self.start_with_key, self.carrying, flush=True)

        self.door = door
        self.mission = "open the door"
        
    def reset(self):
        self.picked_key = self.start_with_key
        return super().reset()

    def step(self, action):
        obs, reward, done, info = super().step(action)
        
        if not self.picked_key:
            if self.carrying is not None:
                self.picked_key = True
                reward += self._reward() if self.reward_decay else 1.

        if action == self.actions.toggle:
            if self.door.is_open:
                reward = self._reward() if self.reward_decay else 1.
                done = True

        return obs, reward, done, info

register(
    id='MiniGrid-UnlockEnhanced-v0',
    entry_point='gym_minigrid.envs:UnlockEnhanced'
)
