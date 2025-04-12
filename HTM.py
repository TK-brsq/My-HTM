from dataclasses import dataclass
from typing import List, Dict
import numpy as np
import random

@dataclass
class Segment:
    seg_id: int
    synapses: List[Dict]

    def add_synapse(self, source_idx: int, weight: float):
        self.synapses.append({'source_idx': source_idx, 'weight': weight})
    
    def get_active_synapses(self, threshold=0.2):
        return [s for s in self.synapses if s['weight'] >= threshold]

@dataclass
class Cell:
    cel_id: int
    segments: List[Segment]

    def add_segment(self, seg_id=0, synapses=[]):
        self.segments.append(Segment(seg_id, synapses))

@dataclass
class Column:
    col_id: int
    synapses: List[Dict]
    cells: List[Cell]

    def __post_init__(self):
        for cell in self.cells:
            cell.add_segment()
    def get_active_synapses_idx(self, w_threshold=0.2):
        return [idx for idx, weight in self.synapses.items() if weight >= w_threshold]

@dataclass
class SpatialPooler:
    input_size: int
    num_columns: int
    synapse_per_col: int
    cells_per_col: int
    columns: List[Column]

    def __post_init__(self):
        for col_id in range(self.num_columns):
            cells = [Cell(cel_id, []) for cel_id in range(self.cells_per_col)]

            source_idx = np.random.choice(self.input_size, self.synapse_per_col, replace=False)
            synapses = [{'source_idx': idx, 'weight': random.gauss(0.2, 0.03)}\
                        for idx in source_idx]
            column = Column(col_id, synapses, cells)
            self.columns.append(column)

    def calc_overlap(self, input: np.ndarray, o_threshold: int) -> np.ndarray:
        overlaps = np.zeros(self.num_columns, dtype=int)
        for col_id, column in enumerate(self.columns):
            active_synapses_idx = column.get_active_synapses_idx()
            overlap = sum(1 for idx in active_synapses_idx if input[idx] > 0)
            if overlap < o_threshold:
                overlap = 0
            overlaps[col_id] = overlap
        return overlaps
    
    def inhibitation(self, overlaps: np.ndarray, num_active_col: int) -> List[int]:
        if num_active_col > len(overlaps):
            return list(range(len(overlaps)))
        active_idx = np.argpartition(-overlaps, num_active_col)[num_active_col]
        return active_idx.tolist()
    
    def learn(self, active_col_idx, w_threshold: float):
        for idx in active_col_idx:
            column = self.columns[idx]
            for s in column.synapses:
                if s['weight'] >= w_threshold:
                    s['weight'] = min(1, s['weight'] + 0.03)
                else:
                    s['weight'] = max(0, s['weight'] - 0.02)