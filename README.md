# README

## Overview

This folder contains Noah Larbalestier's (my) implementation for PA5, where the goal is to build a generic SAT solver using **GSAT** and **WalkSAT**. The provided `.cnf` files are sudoku encodings, the solver works for any CNF formula with integer literals given in the `.cnf`file format.

The key files are:

- **SAT.py** — my SAT solver implementation  
- **run_tests.py** — my helper script for running solver tests  

As Given: 

- **Sudoku.py**, **sudoku2cnf.py**, **display.py** — provided utilities  
- `.cnf` files — Sudoku constraint encodings  
- `.sud` files — human-readable Sudoku puzzles  
- `.sol` files — solver output created by the solver  

---

## SAT Solver (SAT.py)

`SAT.py` implements two local-search SAT algorithms:

1. **GSAT**
2. **WalkSAT**

The solver is  general; it can run on any CNF file using integer literals, not just Sudoku. It is sinspried by both the wikipedia aritcle on GSAT/WalkSAT and also the research paper `A New Method for Solving Hard Satisfiability Problems` (1992)

### Features

- incremental clause evaluation, meaning flips happen to try to satisfy and it re-evals a CNF instance with some assignment after each flip
- GSAT and WalkSAT  
- Adjustable parameters (noise `p`, max flips, max tries, seed)  
- `.sol` output file  
- CLI usage and a testing file 
- determinsitic behavior via using `--seed`  

### Default Algo Params

Based on testing and assignment recommendations:
- **WalkSAT default noise p = 0.7**
- **GSAT default noise p = 0.5**  
  //Not explicitly specified in the assignment- lower `p` in GSAT encourages fewer random moves, helping algo escape local minima while maintaining more focused search compared to WalkSAT, which is better suited for higher noise values//

- You can override either default by passing a `--p` val arg from the command line

### Running the Solver 
Solve a CNF file: (default WalkSAT)

```bash
python3 SAT.py rows.cnf
```

Choose GSAT:

```bash
python3 SAT.py rows.cnf --algo gsat
```

Set custom params:

```bash
python3 SAT.py rows.cnf --algo walksat --p 0.7 --max-flips 100000 --max-tries 50 --seed 3
```

Specify output:

```bash
python3 SAT.py rows.cnf --out solution.sol
```

The solver writes `.sol` files, each variable as (positive = true, negative = false)

To visualize Sudoku solutions:

```bash
python3 display.py rows.sol
```

---

## Testing Script (run_tests.py)

`run_tests.py` is a test harness for trying the solver on the provided CNF files


by defualt, running

```bash
python3 run_tests.py
```

executes a few quick tests on small CNFs:

- `one_cell.cnf`
- `all_cells.cnf`
- `rows.cnf`

Each writes a `.sol` file if successful and True to stdout.

Addiitonally, 

within `run_tests.py`, there are optional lines for harder CNFs that may fail or take a very long time with these naive SAT algos:

- `rows_and_cols.cnf`
- `puzzle1.cnf`
- `puzzle2.cnf`
- `puzzle_bonus.cnf`

### further customizing Tests--

can easily modify

- Which CNF file to run  
- Which algorithm to use  
- Noise (`p`)  
- Max flips / tries  
- Seeds  

### MaxWalkSAT (optional extension)

As an extension, `SAT.py` also implements **MaxWalkSAT**, a weighted variant of WalkSAT.

- Each clause is given a weight `w_i` (default 1.0 for all).
- The algorithm tries to **maximize the total weight of satisfied clauses**, or equivalently minimize the weighted cost of unsatisfied clauses.
- The code supports a `weights` vector, so in problems like Sudoku you can make certain constraints “harder” than others.i.e, 
  - Giving unit clauses (e.g., given Sudoku clues) a larger weight than other constraints (those to find)
  - This strongly discourages MaxWalkSAT from violating known cell values, while still allowing a little flexibility during random moves.
  - While there are other implementations and Max Walk cold be coupled with helpers/other input methods to capitaize on weighted clauses,
  - this also assists with the problem of "hard" constraints, i.e, the fixed sudoku clues. 

From the command line you can run:

```bash
python3 SAT.py puzzle1.cnf --algo maxwalksat --p 0.7 --max-flips 100000 --max-tries 20

## .sol Files

The solver outputs `.sol` files where each line contains a literal:

- Positive integer → variable assigned //T

## Notes

- purely local search
- no backtracking / DPLL is implemented.
- Performance varies by CNF complexity; WalkSAT generally performs better on harder sudoku encodings
- no external libs needed

