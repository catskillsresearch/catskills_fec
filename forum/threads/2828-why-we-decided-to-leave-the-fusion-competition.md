# Why we decided to leave the Fusion competition

- Thread: https://www.codabench.org/forums/17429/2828/
- Started by: gsc-x
- Date created: Oct 07, 2026
- Pinned: no
- Posts: 1

---

## Post 1 — gsc-x

*Oct. 7, 2026, 7:40 a.m.*

First, I would like to thank the organizers for creating this competition and, especially, for giving participants the opportunity to test different control approaches on a common benchmark. For us, the competition was genuinely useful because it helped us identify not only what worked, but also where we believe an important methodological limitation of the environment lies.
Our work was carried out within the GSC-X Fusion Sandbox, where we used the competition environment as part of a broader investigation into predictive stabilization, nonlinear plasma dynamics and metastability.
During this work, we increasingly focused on the nonlinear and metastable character of plasma dynamics. Our conclusion was that continuing to optimize purely for the competition score would no longer correspond to the scientific question we wanted to investigate.
The central issue for us is the relationship between the plasma dynamics and the available control channels.
A fusion plasma is a strongly nonlinear dynamical system. Instabilities, regime transitions and loss of confinement can emerge from coupled feedback processes and can develop in a highly state-dependent way. In such a system, we believe that representing effective control authority primarily through approximately linear actuator responses introduces an important limitation.
A linear actuator model can certainly be useful for benchmarking and local control around an operating point. However, if the underlying system moves toward a nonlinear instability or a metastable boundary, the assumption that control influence continues to scale linearly may become increasingly restrictive.
From our perspective, this creates a fundamental distinction between:
optimizing a controller inside the benchmark, and
demonstrating causal stabilization of the nonlinear plasma itself.
We therefore decided not to continue optimizing our submission solely for the leaderboard. Instead, we redirected the work toward studying the nonlinear stability structure itself — particularly metastability corridors, early transitions, and the possibility of regenerative stabilization before the system crosses a critical boundary.
This work, developed through the GSC-X Fusion Sandbox, contributed to the following publications and public research outputs:
GSC-X Fusion Sandbox
https://doi.org/10.5281/zenodo.22774557
Web: https://fusion.gds-gsc-x.com
REGENERATIVE METASTABILITY
https://doi.org/10.5281/zenodo.20688368
GDS/GSC-X – Emergent Metastability Corridors
https://doi.org/10.5281/zenodo.20365379
Our decision to leave the competition should therefore not be interpreted as criticism of the competition itself. Quite the opposite: the benchmark helped us expose a scientific question that we considered more important than further leaderboard optimization.
We are grateful for that.
The main lesson we took from the work is that future plasma-control benchmarks could become even more interesting if actuator effectiveness itself were allowed to become nonlinear, state-dependent and potentially degraded near instability boundaries.
In our view, that would move the problem closer to the real challenge:
not simply controlling a nonlinear plasma with fixed linear control authority, but learning how control authority itself changes as the plasma approaches a nonlinear transition.
Thank you to the organizers for the opportunity, the benchmark, and the possibility to learn from it.
Martin Štěpánů
Founder & Lead Architect
GDS Research Institute s.r.o.
GSC-X / GSC-X Fusion Sandbox
Main website: https://gds-gsc-x.com
Email: stepanumartin@gds-gsc-x.com
