# Reward Function

The active v25 training configuration focuses on contact quality and recoverable next-ball states.

Main terms:

| Term | Purpose |
| --- | --- |
| Contact bonus | Reward useful racket-ball contact |
| Apex match reward | Push the ball toward the desired apex |
| Easy next ball reward | Prefer post-contact states that are reachable |
| Stable contact reward | Encourage contacts with sufficient apex and lateral stability |
| Stable cycle reward | Encourage repeated useful contacts |
| Trajectory match reward | Align outgoing ball trajectory with the planner target |
| Action penalty | Discourage excessive residual actions |
| Tilt penalties | Avoid unnecessary paddle tilt changes |
| Lateral velocity penalties | Reduce hard-to-recover sideways motion |
| Non-useful contact penalty | Penalize contact that does not produce a playable next ball |

Terms that are inactive in the selected run are intentionally omitted from this web document.
