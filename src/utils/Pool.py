from .constant import POOL_LIMIT 

from dataclasses import dataclass

@dataclass
class Candidate:
    score: float
    model: any
    loss: float
    action: list
    @property
    def error(self):
        return self.loss

    @property
    def expression(self):
        return self.model.expression_visualize()
    

class Pool:
    def __init__(self):
        self.POOL_LIMIT = POOL_LIMIT
        self.candidates = []
        self.top_score_threshold = 0.

    def add(self, score, model, loss, op_seq):
        if score <= self.top_score_threshold:
            return
        self.candidates.append(Candidate(score, model, loss,op_seq))
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


        