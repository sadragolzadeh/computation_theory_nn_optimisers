# Computation-theoretic neural network optimisation

Run everything from this folder (pure NumPy, no GPU).

| File | What it is |
|---|---|
| complexity_aware_opt.py | Certified removal of dead / affine-rank / duplicate neurons in ReLU MLPs (exact, IBP and data oracles) vs pruning and NAS |
| automata_rnn_compression.py | RNN -> minimal DFA via bounded Myhill-Nerode state merging, plus compression baselines |
| automata_rnn_extended.py | Extra automata experiments: imperfect teacher, a^n b^n, DFA -> ReLU RNN, GRU on Tomita grammars (needs the file above) |
| lambda_nn.py | Typed lambda-term architectures, beta-reduction and cost-guided sound rewriting |
| unified_comparison.py | Compares the three methods (imports the four modules above) |

Quick start:
    python unified_comparison.py          # all three tracks
    python complexity_aware_opt.py
    python automata_rnn_compression.py
    python automata_rnn_extended.py A C   # pick experiments A-D
    python lambda_nn.py
