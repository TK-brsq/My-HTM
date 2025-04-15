from dataclasses import dataclass, field
from typing import List, Dict, Tuple
import numpy as np
import random

@dataclass
class Synapse:
    target: Tuple[int, int, int]
    weight: float

@dataclass
class Segment:
    seg_id: int
    num_synapses: int
    synapses: List[Synapse] = field(default_factory=list)

    def add_synapse(self, target: Tuple[int, int, int], weight: float):
        if len(self.synapses) >= self.num_synapses:
            raise ValueError('maximum synapse limit in segment')
        self.synapses.append(Synapse(target, weight))
    
    def get_active_source(self, w_threshold=0.2):
        active_sy = (self.synapses_weight >= w_threshold)
        return self.synapses_src[active_sy]
    
    def calc_overlap(self, active_state):


@dataclass
class Cell:
    cel_id: int
    num_segments: int
    segments: List[Segment] = field(default_factory=list)
    num_synapses: int = 64

    def __post_init__(self):
        for seg_id in range(self.num_segments):
            self.segments.append(Segment(seg_id, self.num_synapses))

@dataclass
class Column:
    col_id: int
    num_cells: int
    num_synapses: int
    input_size: int
    cells: List[Cell] = field(default_factory=list)
    synapses_src: np.ndarray = field(init=False)
    synapses_weights: np.ndarray = field(init=False)

    def __post_init__(self):
        for cell_id in range(self.num_cells):
            self.cells.append(Cell(cell_id, 32))
        self.synapses_src = np.random.choice(self.input_size, self.num_synapses, replace=False)
        self.synapses_weights = np.random.normal(0.2, 0.025, self.num_synapses)

    def get_active_synapses_idx(self, w_threshold=0.2):
        active_sy = (self.synapses_weights >= w_threshold)
        return self.synapses_src[active_sy]


@dataclass
class SpatialPooler:
    input_size: int
    num_columns: int
    num_synapses: int
    num_cells: int
    columns: List[Column] = field(default_factory=list)

    def __post_init__(self):
        for col_id in range(self.num_columns):
            column = Column(col_id, self.num_cells, self.num_synapses, self.input_size)
            self.columns.append(column)

    def calc_overlap(self, input: np.ndarray, o_threshold: int) -> np.ndarray:
        overlaps = np.zeros(self.num_columns, dtype=int)
        for col_id, column in enumerate(self.columns):
            active_synapses_idx = column.get_active_synapses_idx()
            overlap = np.sum(input[active_synapses_idx] > 0)
            overlaps[col_id] = overlap if overlap >= o_threshold else 0
        return overlaps
    
    def inhibitation(self, overlaps: np.ndarray, num_active_col: int) -> List[int]:
        if num_active_col > len(overlaps):
            return list(range(len(overlaps)))
        top_k_idx = np.argsort(overlaps)[-num_active_col:]
        return top_k_idx.tolist()
    
    def learn(self, active_col_idx, w_threshold: float):
        for idx in active_col_idx:
            column = self.columns[idx]
            weights = column.synapses_weights
            up_idx = weights >= w_threshold
            down_idx = ~up_idx
            weights[up_idx] = np.clip(weights[up_idx] + 0.02, 0.0, 1.0)
            weights[down_idx] = np.clip(weights[down_idx] - 0.02, 0.0, 1.0)

@dataclass
class TemporalPooling:
    columns: List[Column]
    num_cells: int
    active_state: np.ndarray = field(init=False) # column x cell
    predictive_state: np.ndarray = field(init=False) # column x cell

    def __post_init__(self):
        self.active_state = np.zeros((len(self.columns), self.num_cells))
        self.predictive_state = np.zeros((len(self.columns), self.num_cells))
        
    def calc_active_state(self, active_colmn_idx):
        for col_idx in active_colmn_idx:
            predicted = False
            for cel_idx in self.num_cells:
                if self.predictive_state[col_idx, cel_idx] == True:
                    self.active_state[col_idx, cel_idx] == True
                    predicted = True
            if predicted == False:
                self.active_state[col_idx] = True
    
    def calc_predictive_state(self):
        for col in self.columns:
            for cel in col.cells:
                for seg in cel.segments:
