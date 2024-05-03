from panda_gym.envs.core import RobotTaskEnv

import pybullet as p
from project.env import Action
from typing import Optional


class KeyboardController:

    def __init__(self):
        # keyboard.Listener(on_press=self.on_press).start()
        pass

    def get_key(self) -> int:
        keys = p.getKeyboardEvents()
        for k, v in keys.items():
            # print(k, v)
            if v == 3:
                return k

        return -1

    def get_action(self) -> Optional[Action]:
        """
        get_action returns the key -> action conversion

        Fingers:
        a - close
        d - open

        Elevation:
        q - down
        e - up

        Movement:
        u i o
        j   l
        m , .

        u - northwest
        i - north
        o - northeast
        j - west
        l - east
        m - southwest
        , - south
        . - southeast
        """
        key = self.get_key()

        if key == 113:
            return Action(Action.DOWN)
        elif key == 101:
            return Action(Action.UP)
        elif key == 117:
            return Action(Action.NORTHWEST)
        elif key == 105:
            return Action(Action.NORTH)
        elif key == 111:
            return Action(Action.NORTHEAST)
        elif key == 106:
            return Action(Action.WEST)
        elif key == 108:
            return Action(Action.EAST)
        elif key == 109:
            return Action(Action.SOUTHWEST)
        elif key == 44:
            return Action(Action.SOUTH)
        elif key == 46:
            return Action(Action.SOUTHEAST)
        elif key == 97:
            return Action(Action.CLOSE_GRIPPER)
        elif key == 100:
            return Action(Action.OPEN_GRIPPER)

        return None
