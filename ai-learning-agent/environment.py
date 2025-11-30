class LineWorld:
    def __init__(self, size=6):
        self.size = size
        self.start = 0
        self.goal = size - 1
    
    def reset(self):
        self.position = self.start
        return np.array([self.position], dtype=np.float32)
    
    def step(self, action):

        if action == 0 and self.position > 0:
            self.position -= 1  
        elif action == 1 and self.position < self.goal:
            self.position +=1

        done = self.position == self.goal

        if done:
            reward = 1.0
        else: 
            reward = -0.01

        next_state = np.array([self.position], dtype=np.float32)
        return next_state, reward, done, {}
