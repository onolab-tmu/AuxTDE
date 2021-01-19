# -*- coding: utf-8 -*-
import time


class tictoc:
    """
    MATLAB like tic toc functions

    parameters
    ----------
    tag: str, optional
        The name of the process

    usage
    ----------
    task = tictoc("name of task")
    task.tic()
    task.toc()
    """

    def __init__(self, tag=""):
        self.st = None
        self.tag = tag if tag == "" else tag + ": "

    def tic(self):
        self.st = time.time()

    def toc(self):
        if self.st is not None:
            print(
                "{}Elapsed time is {:.6f} seconds.".format(
                    self.tag, time.time() - self.st
                )
            )
        else:
            print("tic() must be called before calling toc().")
