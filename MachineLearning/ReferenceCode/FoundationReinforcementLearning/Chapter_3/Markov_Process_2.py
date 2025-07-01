import numpy as np
from dataclasses import dataclass
from typing import Optional
import itertools

@dataclass
class Process2:
    @dataclass
    class State:
        price: int
        is_prev_move_up: Optional[bool]
        
    level_param: int # level to which price mean-reverts
    alpha2: float = 0.75 # strength of reverse-pull (value in [0,1])
        
    def up_prob(self, state: State) -> float:
        return 1. / (1 + np.exp(-self.alpha1 * (self.level_param - state.price)))
    
    # random.binomial(n, p, size=None)    
    # n: number of trials, >= 0
    # p: probability of success on each trial [0,1]
    # size: output shape, if None, returns a single value
    def next_state(self, state: State) -> State:
        up_move: int = np.random.binomial(1, self.up_prob(state), 1)[0]
        # up_move will be 0 or 1
        # if 0, price = state.price + 0 * 2 - 1 = state.price - 1 (will go down)
        # if 1, price = state.price + 1 * 2 - 1 = state.price + 1 (will go up)
        return Process1.State(price=state.price + up_move * 2 - 1)
        
def simulation(process, start_state):
    state = start_state
    while True:
        yield state
        state = process.next_state(state)

def process1_price_traces(
            start_price: int,
            level_param: int,
            alpha1: float,
            time_steps: int,
            num_traces: int) -> np.ndarray:
        process = Process1(level_param=level_param, alpha1=alpha1)
        start_state = Process1.State(price=start_price)
        return np.vstack([
        np.fromiter((s.price for s in itertools.islice(
        simulation(process, start_state),
        time_steps + 1
        )), float) for _ in range(num_traces)])

if __name__ == "__main__":
    # Example usage
    start_price = 100
    level_param = 100
    alpha1 = 0.25
    time_steps = 10
    num_traces = 5
    
    traces = process1_price_traces(start_price, level_param, alpha1, time_steps, num_traces)
    print(traces)