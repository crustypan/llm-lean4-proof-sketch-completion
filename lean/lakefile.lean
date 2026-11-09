import Lake
open Lake DSL

package «lean_judge»

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @ "v4.15.0"

require aesop from git
  "https://github.com/JLimperg/aesop" @ "v4.15.0"

@[default_target]
lean_lib LeanJudge
