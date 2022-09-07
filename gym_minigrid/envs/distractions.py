from gym_minigrid.roomgrid import RoomGrid
# from gym_minigrid.register import register
from gym.utils import seeding

class Distractions(RoomGrid):
    """
    None -> key-green 1
    key-green -> key-magenta 2
    key-green -> key-blue 2
    None -> box-olive 1
    key-magenta -> key-red 3
    key-red -> ball-olive 4
    key-blue -> ball-grey 3
    key-red -> box-maroon 4
    """

    def __init__(
        self,
        num_rows=5,
        num_nodes=8,
        # obj_type="ball",
        room_size=6,
        seed=None,
        **kwargs,
    ):
        self._num_nodes = num_nodes
        super().__init__(
            room_size=room_size,
            num_rows=num_rows,
            max_steps=30*room_size**2,
            seed=seed,
            **kwargs
        )

    def _gen_rooms(self, num_nodes):
        rng, _ = seeding.np_random(1337)
        colors = self.color_names
        has_child = [False]
        parents = [None]
        heights = [0]
        for i in range(num_nodes):
            p = rng.randint(i + 1)
            has_child[p] = True
            parents.append(p)
            has_child.append(False)
            heights.append(heights[p] + 1)
        used = set()
        nodes = [None]
        rooms = []
        for i in range(num_nodes):
            p = parents[i + 1]
            if has_child[i + 1]:
                obj = 'key'
            else:
                obj = rng.choice(['ball', 'box'])
            while True:
                color = rng.choice(colors)
                item = f'{obj}-{color}'
                if item not in used:
                    break
            rooms.append((i, nodes[p][4:] if nodes[p] is not None else None, item))
            used.add(item)
            nodes.append(item)
            # print(nodes[p], '->', item, heights[i + 1])
        # print(rooms)
        return rooms


    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        rooms = self._gen_rooms(self._num_nodes)
        assert len(rooms) <= self.num_rows * 2
        room_indices = self._rand_subset(range(self.num_rows * 2), len(rooms))

        # Connect the middle column rooms into a hallway
        for j in range(1, self.num_rows):
            self.remove_wall(1, j, 3)

        for room, room_idx in zip(rooms, room_indices):
            room_num, door_color, item = room
            row_idx = room_idx >> 1
            orient = room_idx & 1
            col_idx = orient * 2
            if door_color is not None:
                door, _ = self.add_door(col_idx, row_idx, col_idx, color=door_color, locked=True)
            else:
                door, _ = self.add_door(col_idx, row_idx, col_idx, locked=False)
            door._name = f"door-room-{room_num}"
            item_kind, item_color = item.split('-')
            obj, _ = self.add_object(col_idx, row_idx, kind=item_kind, color=item_color)
            obj._name = f"{item_kind}-{item_color}"

        # Place the agent in the middle
        self.place_agent(1, self.num_rows // 2)

        # Make sure all rooms are accessible
        self.connect_all()

        self.mission = ""

    def step(self, action):
        obs, reward, done, info = super().step(action)

        # if action == self.actions.pickup:
        #     if self.carrying and self.carrying == self.obj:
        #         # reward = self._reward()
        #         done = True

        return obs, reward, done, info


if __name__ == "__main__":
    env = Distractions()
    env._gen_rooms(15)