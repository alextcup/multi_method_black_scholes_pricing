from dataclasses import dataclass

@dataclass(frozen=True)
class Option:
    S0: float
    K: float
    T: float
    r: float
    sigma: float
    div_yield: float = 0.0
    option_type: str = "call"
    exercise: str = "european"

    def __post_init__(self):
        if self.S0 <= 0:
            raise ValueError("S0 must be positive")
        
        if self.K <= 0:
            raise ValueError("K must be positive")
        
        if self.T <= 0:
            raise ValueError("T must be positive")
        
        if self.sigma <= 0:
            raise ValueError("sigma must be positive")

        if self.div_yield < 0:
            raise ValueError("div_yield must be non-negative")
        
        if self.option_type not in ("call", "put"):
            raise ValueError("option_type must be 'call' or 'put'")
        
        if self.exercise not in ("european", "american"):
            raise ValueError("exercise must be 'european' or 'american'")
