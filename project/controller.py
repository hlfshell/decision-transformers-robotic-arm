from panda_gym.envs.core import RobotTaskEnv

import pybullet as p
from project.env import Action
from typing import Any, Dict, Optional

import pygame

from threading import Lock, Thread

from time import sleep


class KeyboardController:

    def get_key(self) -> int:
        keys = p.getKeyboardEvents()
        for k, v in keys.items():
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


class GamePad:

    JoyButtonUp = 1540
    JoyButtonDown = 1539
    JoyAxisMotion = 1536

    LeftJoystickX = 0
    LeftJoystickY = 1
    ShoulderRight = 7
    ShoulderLeft = 6
    TriggerRight = 4
    TriggerLeft = 5

    def __init__(self):
        pygame.init()
        pygame.joystick.init()
        self.__joystick = pygame.joystick.Joystick(0)
        self.__joystick.init()

        self.state: Dict[str, Any] = {
            "movement": [0.0, 0.0],
            "raise": False,
            "lower": False,
            "open": False,
            "close": False,
        }

        self.__lock = Lock()

        self.__thread = Thread(target=self.__get_joystick)
        self.__thread.start()

        self.__keep_alive = Thread(target=self.__keep_alive)
        self.__keep_alive.start()

    def get_action(self) -> Optional[Action]:
        """
        get_action queries the latest state and generates one of our discretized
        actions for it. Returns None if nothing has been specified.
        """

        # We can only report one button at a time (as we have discretized our actions
        # as such). This means that we want to prioritize certain triggers over others
        # as a first come first serve kind of deal.
        with self.__lock:
            if self.state["raise"]:
                return Action(Action.UP)
            elif self.state["lower"]:
                return Action(Action.DOWN)
            elif self.state["open"]:
                return Action(Action.OPEN_GRIPPER)
            elif self.state["close"]:
                return Action(Action.CLOSE_GRIPPER)
            elif self.state["movement"][0] != 0 or self.state["movement"][1] != 0:
                if (
                    self.state["movement"][0] == 1.0
                    and self.state["movement"][1] == 1.0
                ):
                    return Action(Action.NORTHEAST)
                elif (
                    self.state["movement"][0] == -1.0
                    and self.state["movement"][1] == 1.0
                ):
                    return Action(Action.NORTHWEST)
                elif (
                    self.state["movement"][0] == -1.0
                    and self.state["movement"][1] == -1.0
                ):
                    return Action(Action.SOUTHWEST)
                elif (
                    self.state["movement"][0] == 1.0
                    and self.state["movement"][1] == -1.0
                ):
                    return Action(Action.SOUTHEAST)
                elif self.state["movement"][0] == 1.0:
                    return Action(Action.EAST)
                elif self.state["movement"][0] == -1.0:
                    return Action(Action.WEST)
                elif self.state["movement"][1] == 1.0:
                    return Action(Action.NORTH)
                elif self.state["movement"][1] == -1.0:
                    return Action(Action.SOUTH)

        return None

    def __keep_alive(self):
        """
        The controller likes to disconnect regularly; by sending
        an effectively null rumble command on the regular we prevent
        the controller from disconnecting or sleeping.
        """
        while True:
            self.__joystick.rumble(0, 0, 500)
            sleep(1.0)

    def __get_joystick(self):
        """
        __get_joystick will run constantly in the background, reading
        incoming events and saving them to memory for whenever our
        main process queries for the latest that the controller
        has for it.

        Our goal is to preprocess these events and figure out general
        intent of the user into singular discrete actions for
        the next query
        """

        left_trigger = False
        right_trigger = False
        left_shoulder = False
        right_shoulder = False
        joystick_axis = [0.0, 0.0]

        x_axis = 0.0
        y_axis = 0.0

        while True:
            for event in pygame.event.get():
                if event.type == self.JoyButtonDown:
                    if event.button == self.ShoulderRight:
                        right_shoulder = True
                    elif event.button == self.ShoulderLeft:
                        left_shoulder = True
                elif event.type == self.JoyButtonUp:
                    if event.button == self.ShoulderRight:
                        right_shoulder = False
                    elif event.button == self.ShoulderLeft:
                        left_shoulder = False
                elif event.type == self.JoyAxisMotion:
                    if event.axis == self.LeftJoystickX:
                        joystick_axis[0] = event.value
                    elif event.axis == self.LeftJoystickY:
                        joystick_axis[1] = event.value
                    elif event.axis == self.TriggerRight:
                        if event.value > 0.8:
                            right_trigger = True
                        else:
                            right_trigger = False
                    elif event.axis == self.TriggerLeft:
                        if event.value > 0.8:
                            left_trigger = True
                        else:
                            left_trigger = False

            forward_trigger = 0.8
            diagonal_trigger = 0.5

            if (
                joystick_axis[0] > diagonal_trigger
                and joystick_axis[1] > diagonal_trigger
            ):
                x_axis = -1.0
                y_axis = -1.0
            elif (
                joystick_axis[0] < -diagonal_trigger
                and joystick_axis[1] > diagonal_trigger
            ):
                x_axis = 1.0
                y_axis = -1.0
            elif (
                joystick_axis[0] < -diagonal_trigger
                and joystick_axis[1] < -diagonal_trigger
            ):
                x_axis = 1.0
                y_axis = 1.0
            elif (
                joystick_axis[0] > diagonal_trigger
                and joystick_axis[1] < -diagonal_trigger
            ):
                x_axis = -1.0
                y_axis = 1.0
            elif joystick_axis[0] > forward_trigger:
                x_axis = -1.0
                y_axis = 0.0
            elif joystick_axis[0] < -forward_trigger:
                x_axis = 1.0
                y_axis = 0.0
            elif joystick_axis[1] > 0.8:
                x_axis = 0.0
                y_axis = -1.0
            elif joystick_axis[1] < -0.8:
                x_axis = 0.0
                y_axis = 1.0
            else:
                x_axis = 0.0
                y_axis = 0.0

            # Save our current state to be read on request
            with self.__lock:
                self.state["movement"] = [x_axis, y_axis]
                self.state["raise"] = left_trigger
                self.state["lower"] = right_trigger
                self.state["open"] = left_shoulder
                self.state["close"] = right_shoulder

            sleep(0.1)
