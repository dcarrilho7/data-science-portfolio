# environment.py

class GridWorld2D:
    def __init__(self):
        self.grid = [
            ["S", ".", ".", "X", "."],
            [".", "X", ".", "X", "."],
            [".", ".", ".", ".", "."],
            ["X", ".", "X", ".", "G"],
            [".", ".", ".", "X", "."],
        ]
        self.rows = 5
        self.cols = 5
        self.start = (0, 0)
        self.state = self.start

    def reset(self):
        self.state = self.start
        return self._state_id(self.state)

    def step(self, action):
        r, c = self.state

        if action == "UP":
            r -= 1
        elif action == "DOWN":
            r += 1
        elif action == "LEFT":
            c -= 1
        elif action == "RIGHT":
            c += 1

        # stay in bounds
        r = max(0, min(self.rows - 1, r))
        c = max(0, min(self.cols - 1, c))

        self.state = (r, c)
        cell = self.grid[r][c]

        if cell == "G":
            return self._state_id(self.state), 10, True
        if cell == "X":
            return self._state_id(self.state), -10, True

        return self._state_id(self.state), -0.1, False

    def _state_id(self, state):
        r, c = state
        return r * self.cols + c
