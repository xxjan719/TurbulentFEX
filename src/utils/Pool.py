from .constant import POOL_LIMIT 

from dataclasses import dataclass

@dataclass
class Candidate:
    score: float
    model: any
    loss: float
    action: list
    expression: str = None  # Store expression explicitly
    
    @property
    def error(self):
        return self.loss

    def get_expression(self):
        if self.expression is None:
            return self.model.expression_visualize_simplified()
        return self.expression
    

class Pool:
    def __init__(self):
        self.POOL_LIMIT = POOL_LIMIT
        self.candidates = []
        self.top_score_threshold = 0.

    def add(self, score, model, loss, op_seq):
        if score <= self.top_score_threshold:
            return
        # Store the expression at the time of adding to preserve the sequence-expression correspondence
        expression = model.expression_visualize_simplified()
        self.candidates.append(Candidate(score, model, loss, op_seq, expression))
        self.sort()
        if len(self.candidates) > self.POOL_LIMIT:
            self.candidates.pop(0)
        if len(self.candidates) == self.POOL_LIMIT:
            self.top_score_threshold = self.candidates[0].score

    def sort(self):
        self.candidates.sort(key=lambda c: c.score)

    def __len__(self):
        return len(self.candidates)

    def __iter__(self):
        return iter(self.candidates)


        