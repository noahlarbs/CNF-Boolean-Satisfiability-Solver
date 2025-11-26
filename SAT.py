#Noah Larbalestier; F006B9P
#Soroush Vosoughi, COSC76 25F
#PA5 - Logic: SAT with GSAT and WalkSAT
# This module contains the CNFFormula and SAT classes, which are used to represent and solve CNF formulas.

from __future__ import annotations
import argparse
import random
from typing import List, Sequence, Optional

import random as rng

#_________#
#rep CNFS as a list of clauses, where each clause is a list of ints. Positive k means
#variable k, negative -k means NOT variable k.
class CNFFormula:
    #constructor takes a list of clauses and sets the clauses and num_variables
    def __init__(self, clauses: List[List[int]]):
        self.clauses: List[List[int]] = clauses
        self.num_variables: int = self.numVars(clauses)

    #static method gets the num of vars (ints) used in a cnf formula
    #uses a helper function to find the max variable in the clauses
    @staticmethod
    def numVars(clauses: list[list[int]]) -> int:
        max_var = 0
        for clause in clauses:
            for i in clause:
                v = abs(i)
                if v > max_var:
                    max_var = v
        return max_var

    #class method loads a cnf file and returns a CNFFormula object
    #reads the file line by line, and splits the line into a list of ints
    @classmethod
    def from_file(cls, path: str) -> "CNFFormula":
        """
        Loads a CNF file where each non-empty line is a clause, consisting
        of space-separated integers (no trailing 0 is assumed).
        Blank lines are ignored.
        """
        clauses: List[List[int]] = []
        with open(path, "r") as f:
            for l in f:
                line = l.strip()
                if not line:
                    continue
                parts = line.split()
                clause = [int(p) for p in parts]#need to filter any more? ie no zeores 
                clauses.append(clause)
        return cls(clauses)

#takes an assignment and a clause (a list of ints), returns True if the clause is satisfied by the assignment
    def eval(self, assignment: list[Optional[bool]], clause: list[int]):
        """
        Returns True if the clause is satisfied by the given assignment.
        assignment is 1-indexed: assignment[var_id] -> bool or None.
        """
        for int in clause:
            var_id = abs(int)
            sign = int > 0
            val = assignment[var_id]
            if val is None:
                #treat not assigned as false //is this wrong?
                continue
            if (val and sign) or ((not val) and (not sign)):
                return True
        return False

    #returns number of clauses satisfied by the assignment
    def count_satisfied(self, assignment: Sequence[Optional[bool]]):
        return sum(1 for clause in self.clauses if self.eval(assignment, clause))

    #returns list of indices of the clauses not satisfied by the assignment
    def unsatisfied(self, assignment: Sequence[Optional[bool]]) -> List[int]:
        return [i for i, clause in enumerate(self.clauses) if not self.eval(assignment, clause)]

#_________#
#SAT solver class - a SAT inst is a formula and an assgt that satisfies it
#constructor takes a cnf path and sets the cnf, assgt, model, pos_occurrences, and neg_occurrences
#pos_occurrences and neg_occurrences are lists of lists of ints, idx is the variable id and the list is the clauses  contain it
class SAT:
   
    def __init__(self, cnf_path: str):
        self.cnf: CNFFormula = CNFFormula.from_file(cnf_path)
        #1-indexed assignment array
        self.assignment: List[Optional[bool]] = [None] * (self.cnf.num_variables + 1)
        #model is the assignment that satisfies the formula
        self.model: Optional[List[Optional[bool]]] = None
        #variable occurrence indices for incremental updates (per var, asks, which clauses contain it?)
        self.pos_occurrences: List[List[int]] = [[] for _ in range(self.cnf.num_variables + 1)]
        self.neg_occurrences: List[List[int]] = [[] for _ in range(self.cnf.num_variables + 1)]

        #populate the pos_occurrences and neg_occurrences lists
        for idx, clause in enumerate(self.cnf.clauses):
            for lit in clause:
                v = abs(lit)
                if lit > 0:
                    self.pos_occurrences[v].append(idx)
                else:
                    self.neg_occurrences[v].append(idx)


    #helpers 
    #for gsat and walksat, randomize the assignment (as spoken about in the article), other approach is to use a backtracking algorithm
    def random_assignment(self, rng: random.Random) -> None:
        for var_id in range(1, self.cnf.num_variables + 1):
            self.assignment[var_id] = bool(rng.getrandbits(1))

    #returns true if the assignment satisfies all the clauses
    def is_satisfied(self) -> bool:
        return self.cnf.count_satisfied(self.assignment) == len(self.cnf.clauses)

    #flip assignment of the variable id
    def flip_in_place(self, var_id: int) -> None:
        assert 1 <= var_id <= self.cnf.num_variables
        current = self.assignment[var_id]
        if current is None:
            #default unassigned to F before flipping? or backtrack?
            current = False
        self.assignment[var_id] = not current

    ### removed scoring helper - returns the delta in satisfied clauses if we flip var_id
    #was unused and expensive to compute, b/c for eahc flip we had to re-evaluate all the clauses####
    
    #_________#
    #GSAT - helpers below
    # this algo will try to find a satisfying assignment for the formula by flipping variables and scoring the changes
    # max_tries is the number of times to try to find a satisfying assignment
    # max_flips is the number of flips to try
    # p_random_move is the probability of flipping a random variable
    # seed is the random seed
    def gsat(self, max_tries=20, max_flips=20000, p_random_move=0.5, seed=None):
        rng_local = random.Random(seed)
        for _ in range(max_tries):
            self.random_assignment(rng_local)
            
            #initialize clause cache - list of bools, true if the clause is satisfied by the assignment
            #doe sthis get created again each time in the main loop? 
            
            clause_satisfied = [self.cnf.eval(self.assignment, c)
                                for c in self.cnf.clauses]
            num_sat = sum(clause_satisfied)
            
            for _ in range(max_flips):
                if num_sat == len(self.cnf.clauses):
                    self.model = list(self.assignment)
                    return True

                if rng_local.random() < p_random_move:
                    var_id = rng_local.randint(1, self.cnf.num_variables)
                else:
                    best_delta = None
                    best_vars = []
                    for v in range(1, self.cnf.num_variables + 1):
                        delta = self.delta4Flip(v, clause_satisfied)
                        if best_delta is None or delta > best_delta:
                            best_delta = delta
                            best_vars = [v]
                        elif delta == best_delta:
                            best_vars.append(v)
                    var_id = rng_local.choice(best_vars)

                # apply chosen flip and update cache
                delta = self.doFlip(var_id, clause_satisfied)
                num_sat += delta

        return False

    #helper function to get the delta in satisfied clauses if we flip var_id
    #we only look at the clauses where var appears
    def delta4Flip(self, v: int, clause_satisfied: List[bool]) -> int:
        affected = self.pos_occurrences[v] + self.neg_occurrences[v]
        delta = 0
        self.flip_in_place(v)
        for idx in affected:
            before = clause_satisfied[idx]
            after = self.cnf.eval(self.assignment, self.cnf.clauses[idx])
            if before != after:
                delta += 1 if after else -1
        self.flip_in_place(v)
        return delta

    #helper function to apply the flip and update the clause cache
    def doFlip(self, v: int, clause_satisfied: List[bool]) -> int:
        affected = self.pos_occurrences[v] + self.neg_occurrences[v]
        delta = 0
        self.flip_in_place(v)
        for idx in affected:
            before = clause_satisfied[idx]
            after = self.cnf.eval(self.assignment, self.cnf.clauses[idx])
            if before != after:
                clause_satisfied[idx] = after
                delta += 1 if after else -1
        return delta

    #_________#
    # WalkSAT - helpers below
    # this algo will try to find a satisfying assignment for the formula by flipping variables and scoring the changes
    # max_tries is the number of times to try to find a satisfying assignment
    # max_flips is the number of flips to try
    # p_random_move is the probability of flipping a random variable
    # seed is the random seed
    def walksat(self, max_tries=1, max_flips=20000, p_random_move=0.7, seed=None):    
       
        """
        WalkSAT pseudocode (template):
          for i in range(max_tries):
            randomize assignment
            for j in range(max_flips):
              if all clauses satisfied: store model and return True
              pick an unsatisfied clause uniformly at random
              with prob p_random_move:
                pick a variable uniformly at random from that clause and flip
              else:
                among variables in that clause, pick the one with best score (ties broken uniformly) and flip
          return False
        """
        rng_local = random.Random(seed)
        for _ in range(max_tries):
            self.random_assignment(rng_local)
            # initialize clause satisfaction cache and unsatisfied set
            clause_satisfied: List[bool] = [self.cnf.eval(self.assignment, clause) for clause in self.cnf.clauses]
            unsat_set = {i for i, sat in enumerate(clause_satisfied) if not sat}

            #main loop - try to find a satisfying assignment
            for _ in range(max_flips):
                if not unsat_set:
                    self.model = list(self.assignment)
                    return True
                clause_idx = rng_local.choice(tuple(unsat_set))
                clause = self.cnf.clauses[clause_idx]
                if rng_local.random() < p_random_move:
                    var_id = abs(rng_local.choice(clause))
                    affected = self.pos_occurrences[var_id] + self.neg_occurrences[var_id]
                    self.flip_in_place(var_id)
                    for idx in affected:
                        old = clause_satisfied[idx]
                        new = self.cnf.eval(self.assignment, self.cnf.clauses[idx])
                        if old != new:
                            clause_satisfied[idx] = new
                            if new:
                                if idx in unsat_set:
                                    unsat_set.remove(idx)
                            else:
                                unsat_set.add(idx)
                #if the clause is unsatisfied, we need to pick a variable to flip, 
                #among variables in that clause, pick the one with best score (ties broken uniformly) and flip
                else:
                    candidate_vars = [abs(lit) for lit in clause]
                    # compute break counts (how many satisfied clauses become unsatisfied)
                    best_break = None
                    best_vars: List[int] = []
                    
                    #gfet break ct for each variable in the clause
                    for v in candidate_vars:
                        affected = self.pos_occurrences[v] + self.neg_occurrences[v]
                        break_count = 0
                        # flip temporarily
                        self.flip_in_place(v)
                        for idx in affected:
                            if clause_satisfied[idx] and not self.cnf.eval(self.assignment, self.cnf.clauses[idx]):
                                break_count += 1

                        # flip back
                        self.flip_in_place(v)
                        if best_break is None or break_count < best_break:
                            best_break = break_count
                            best_vars = [v]
                        elif break_count == best_break:
                            best_vars.append(v)

                    #pick the variable with the best score 
                    var_id = rng_local.choice(best_vars)
                    affected = self.pos_occurrences[var_id] + self.neg_occurrences[var_id]
                    self.flip_in_place(var_id)
                    for idx in affected:
                        old = clause_satisfied[idx]
                        new = self.cnf.eval(self.assignment, self.cnf.clauses[idx])
                        if old != new:
                            clause_satisfied[idx] = new
                            if new:
                                if idx in unsat_set:
                                    unsat_set.remove(idx)
                            else:
                                unsat_set.add(idx)
        return False

    # MaxWalkSAT - weighted variant of WalkSAT
    #Each clause i has a weight w_i.
    #The cost of an assignment is the sum of weights of UNSATISFIED clauses.
    #We try to minimize this cost via random and greedy flips.

    # max_tries is the number of times to try to find a satisfying assignment
    # max_flips is the number of flips to try
    # p_random_move is the probability of flipping a random variable
    # seed is the random seed
    # treat_unit_clauses_as_hard is a boolean to treat unit clauses as hard
    # hard_weight is the weight to assign to unit clauses
    def maxwalksat(self, max_tries=1, max_flips=1000, p_random_move=0.5, seed=None, treat_unit_clauses_as_hard=False, hard_weight=10.0):
        rng_local = random.Random(seed)
        #build weights for each clause
        if treat_unit_clauses_as_hard:
            weights: List[float] = [
                #unit clauses get a much larger weight (distinguished by length 1 vs >1)
                (hard_weight if len(clause) == 1 else 1.0)
                for clause in self.cnf.clauses
            ]
        else:
            #all other clauses get a weight of 1.0
            weights = [1.0 for _ in self.cnf.clauses]

        #track the best assignment and cost seen so far stored outside loop
        best_assignment: Optional[List[Optional[bool]]] = None
        best_cost: float = float("inf")

        
        for _ in range(max_tries):
            #start w a random assignment
            self.random_assignment(rng_local)

            #init clause satisfaciton cache
            clause_satisfied: List[bool] = [
                self.cnf.eval(self.assignment, clause)
                for clause in self.cnf.clauses
            ]

            #curr cost = sum of weights for all unsatisfied clauses
            cost = sum(
                w for w, sat in zip(weights, clause_satisfied) if not sat
            )

            for _ in range(max_flips):
                #if sat all clauses, done.
                if cost == 0:
                    self.model = list(self.assignment)
                    return True

                #best assgt seen so far
                if cost < best_cost:
                    best_cost = cost
                    best_assignment = list(self.assignment)

                #choose unsatisfied clause at random
                unsat_indices = [
                    i for i, sat in enumerate(clause_satisfied) if not sat
                ]
                if not unsat_indices:
                    break

                #choose unsat clause at random
                clause_idx = rng_local.choice(unsat_indices)
                clause = self.cnf.clauses[clause_idx]
                #decide whether to do a random move or a greedy (min-cost) move
                if rng_local.random() < p_random_move:
                    #random move: flip a random var in this clause
                    var_id = abs(rng_local.choice(clause))
                else:
                    #greedy move: for each var in the clause,
                    #compute the change in cost if we flip it, and
                    #choose the flip that yields the smallest cost.
                    best_var = None
                    best_delta_cost = None

                    #compute break ct for each var in the clause- how many satisfied clauses become unsatisfied if we flip it
                    for lit in clause:
                        v = abs(lit)
                        affected = self.pos_occurrences[v] + self.neg_occurrences[v]

                        #temp flip v
                        self.flip_in_place(v)
                        delta_cost = 0.0

                        for idx in affected:
                            before = clause_satisfied[idx]
                            after = self.cnf.eval(self.assignment, self.cnf.clauses[idx])
                            if before != after:
                                #if clause goes from unsatisfied -> satisfied, cost decreases
                                if after:
                                    delta_cost -= weights[idx]
                                else:
                                    #if clause goes from satisfied -> unsatisfied, cost increases
                                    delta_cost += weights[idx]

                        #flip back
                        self.flip_in_place(v)

                        if best_delta_cost is None or delta_cost < best_delta_cost:
                            best_delta_cost = delta_cost
                            best_var = v

                    #if no best_var was found (should have been found above),
                    #fall back to a random var in the clause.
                    var_id = best_var if best_var is not None else abs(rng_local.choice(clause))

                #apply the chosen flip and update clause_satisfied and total cost
                affected = self.pos_occurrences[var_id] + self.neg_occurrences[var_id]
                self.flip_in_place(var_id)
                for idx in affected:
                    before = clause_satisfied[idx]
                    after = self.cnf.eval(self.assignment, self.cnf.clauses[idx])
                    if before != after:
                        clause_satisfied[idx] = after
                        if after:
                            cost -= weights[idx]
                        else:
                            cost += weights[idx]

        #if never found a fully satisfying assignment, but saw some partial
        # assignments with smaller cost, store the best one so it can be inspected.
        if best_assignment is not None:
            self.model = best_assignment

        return False


    #_________#
    #i/o
    #writes the current model as a list of literals, one per line
    def write_solution(self, path: str) -> None:
        
        if self.model is None:
            raise ValueError("No model to write.")
        with open(path, "w") as f:
            for var_id in range(1, self.cnf.num_variables + 1):
                val = self.model[var_id]
                if val is True:
                    f.write(f"{var_id}\n")
                else:
                    #write only posiitve literals 
                    pass

#build arg parser for the command line arguments - allows user to pass in the cnf path, algorithm, max tries, max flips, p random move, seed, and out path
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SAT solver implementing GSAT, WalkSAT, and MaxWalkSAT."
    )
    parser.add_argument("cnf_path", type=str, help="Path to .cnf file.")

    parser.add_argument(
        "--algo",
        choices=["gsat", "walksat", "maxwalksat"],
        default="walksat",
        help="Which local search algorithm to run.",
    )

    parser.add_argument(
        "--max-tries",
        type=int,
        default=10,
        help="Number of random restarts.",
    )

    parser.add_argument(
        "--max-flips",
        type=int,
        default=50000,
        help="Max flips per try.",
    )

    parser.add_argument(
        "--p",
        type=float,
        default=0.7,
        help="Noise probability (random move threshold).",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed.",
    )

    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Path to write .sol output. Defaults to <cnf_basename>.sol",
    )

    return parser

def _default_out_path(cnf_path: str) -> str:
    if cnf_path.lower().endswith(".cnf"):
        return cnf_path[:-4] + ".sol"
    return cnf_path + ".sol"

def _default_out_path(cnf_path: str) -> str:
    if cnf_path.lower().endswith(".cnf"):
        return cnf_path[:-4] + ".sol"
    return cnf_path + ".sol"


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    solver = SAT(args.cnf_path)
    out_path = args.out or _default_out_path(args.cnf_path)

    try:
        if args.algo == "gsat":
            ok = solver.gsat(
                max_tries=args.max_tries,
                max_flips=args.max_flips,
                p_random_move=args.p,
                seed=args.seed,
            )
        elif args.algo == "walksat":
            ok = solver.walksat(
                max_tries=args.max_tries,
                max_flips=args.max_flips,
                p_random_move=args.p,
                seed=args.seed,
            )
        else:  # maxwalksat
            ok = solver.maxwalksat(
                max_tries=args.max_tries,
                max_flips=args.max_flips,
                p_random_move=args.p,
                seed=args.seed,
            )
    except NotImplementedError as e:
        print(str(e))
        ok = False

    if ok:
        solver.write_solution(out_path)
        print(f"Satisfying assignment written to {out_path}")
    else:
        print("No satisfying assignment found (template is incomplete or instance is unsolved).")
