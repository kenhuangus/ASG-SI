"""ALFWorld (text) harness: game selection, expert transcripts, and sandboxed execution of skill programs.

Runs under Linux with the `alfworld` package (TextWorld does not support Windows). Game files are
addressed relative to $ALFWORLD_DATA so that recorded selections are portable.
"""
import builtins
import glob
import json
import os
import random
import signal
import traceback

TASK_TYPES = ("pick_and_place_simple", "look_at_obj_in_light", "pick_clean_then_place_in_recep",
              "pick_heat_then_place_in_recep", "pick_cool_then_place_in_recep", "pick_two_obj_and_place")
STEP_BUDGET = 50
WALL_SECONDS = 10
ALLOWED_MODULES = {"re", "collections", "itertools", "math", "string"}


def data_root():
    return os.path.expanduser(os.environ.get("ALFWORLD_DATA", "~/alfworld_data"))


def list_games(split, task_type):
    """Solvable games of one task type, excluding movable/sliced variants as in the ALFWorld loader."""
    root = data_root()
    out = []
    for path in sorted(glob.glob(os.path.join(root, "json_2.1.1", split, f"{task_type}-*", "*", "game.tw-pddl"))):
        if "movable" in path or "Sliced" in path:
            continue
        with open(path) as f:
            if not json.load(f).get("solvable"):
                continue
        out.append(os.path.relpath(path, root).replace(os.sep, "/"))
    return out


def make_env(game, expert=False):
    import textworld
    import textworld.gym
    from alfworld.agents.environment.alfred_tw_env import (AlfredDemangler, AlfredExpert, AlfredExpertType,
                                                           AlfredInfos)
    extras = ["gamefile"] + (["expert_plan"] if expert else [])
    infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=extras)
    wrappers = [AlfredDemangler(), AlfredInfos] + ([AlfredExpert(AlfredExpertType.HANDCODED)] if expert else [])
    env_id = textworld.gym.register_games([os.path.join(data_root(), game)], infos,
                                          max_episode_steps=STEP_BUDGET + 5, wrappers=wrappers)
    return textworld.gym.make(env_id)


def split_intro(obs):
    """Return (room description, goal) from the initial observation."""
    body = obs.split("-= Welcome to TextWorld, ALFRED! =-")[-1].strip()
    room, _, goal = body.partition("Your task is to:")
    return room.strip(), goal.strip()


def expert_transcript(game):
    env = make_env(game, expert=True)
    obs, info = env.reset()
    room, goal = split_intro(obs)
    steps, done = [], False
    while not done and len(steps) < STEP_BUDGET:
        cmd = info["extra.expert_plan"][0]
        obs, _, done, info = env.step(cmd)
        steps.append([cmd, obs])
    env.close()
    return {"game": game, "room": room, "goal": goal, "won": bool(info["won"]), "steps": steps}


def intro(game):
    env = make_env(game)
    obs, _ = env.reset()
    env.close()
    room, goal = split_intro(obs)
    return {"game": game, "room": room, "goal": goal}


def select(seed=0, n_discovery=4, n_evidence=8, n_verify=10):
    """Disjoint per-type pools drawn from the train split; evaluation uses every validation game."""
    rng = random.Random(seed)
    out = {}
    for tt in TASK_TYPES:
        train = list_games("train", tt)
        rng.shuffle(train)
        discovery, rest = [], iter(train)
        while len(discovery) < n_discovery:
            t = expert_transcript(next(rest))
            if t["won"]:
                discovery.append(t)
        # Continue from where the discovery scan stopped so pools stay disjoint.
        remaining = train[train.index(discovery[-1]["game"]) + 1:]
        evidence = [intro(g) for g in remaining[:n_evidence]]
        verify = remaining[n_evidence:n_evidence + n_verify]
        evaluation = list_games("valid_seen", tt) + list_games("valid_unseen", tt)
        out[tt] = {"discovery": discovery, "evidence": evidence, "verify": verify, "eval": evaluation}
    return out


class StepLimit(BaseException):
    """Raised when the step budget is exhausted; BaseException so `except Exception` in a skill cannot swallow it."""


class Won(BaseException):
    """Raised when the environment reports the task as completed."""


class WallTimeout(BaseException):
    pass


class SkillEnv:
    """The only interface a skill program sees."""

    def __init__(self, tw_env, room, info):
        self._env, self._info = tw_env, info
        self.observation = room
        self.steps = 0
        self.trace = []
        self.won = False

    def step(self, command):
        if self.won:
            raise Won()
        if self.steps >= STEP_BUDGET:
            raise StepLimit()
        command = str(command)
        obs, _, _, info = self._env.step(command)
        self._info = info
        self.steps += 1
        self.observation = obs
        self.trace.append([command, obs])
        if info["won"]:
            self.won = True
            raise Won()
        return obs

    def admissible(self):
        return list(self._info["admissible_commands"])


SAFE_BUILTINS = {name: getattr(builtins, name) for name in (
    "abs", "all", "any", "bool", "chr", "dict", "enumerate", "filter", "float", "frozenset", "hasattr", "int", "isinstance",
    "iter", "len", "ord", "repr", "round",
    "list", "map", "max", "min", "next", "range", "reversed", "set", "sorted", "str", "sum", "tuple", "zip",
    "Exception", "ValueError", "KeyError", "IndexError", "StopIteration", "RuntimeError", "TypeError",
    "AttributeError", "True", "False", "None")}


def _guarded_import(name, *args, **kwargs):
    if name.split(".")[0] not in ALLOWED_MODULES:
        raise ImportError(f"module {name} is not available")
    return __import__(name, *args, **kwargs)


def _alarm(signum, frame):
    raise WallTimeout()


def run_skill(code, game):
    """Execute `solve(env, goal)` on one game. Returns outcome, steps, end reason, and the trace."""
    env = make_env(game)
    obs, info = env.reset()
    room, goal = split_intro(obs)
    senv = SkillEnv(env, room, info)
    scope = {"__builtins__": dict(SAFE_BUILTINS, __import__=_guarded_import, print=lambda *a, **k: None),
             "__name__": "skill"}
    end, error = "returned", None
    old = signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(WALL_SECONDS)
    try:
        exec(compile(code, "<skill>", "exec"), scope)
        solve = scope.get("solve")
        if not callable(solve):
            end = "no_solve_function"
        else:
            solve(senv, goal)
    except Won:
        end = "won"
    except StepLimit:
        end = "step_budget"
    except WallTimeout:
        end = "wall_timeout"
    except SyntaxError as exc:
        end, error = "error:SyntaxError", f"SyntaxError: {exc.msg} (line {exc.lineno})"
    except Exception as exc:  # noqa: BLE001 - any skill error is a recorded outcome
        end = f"error:{type(exc).__name__}"
        lines = [f.lineno for f in traceback.extract_tb(exc.__traceback__) if f.filename == "<skill>"]
        error = f"{type(exc).__name__}: {str(exc)[:200]}" + (f" (line {lines[-1]})" if lines else "")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)
        env.close()
    if senv.won:
        end = "won"  # a skill that swallowed the completion signal still completed the task
    return {"game": game, "goal": goal, "won": senv.won, "steps": senv.steps, "end": end, "error": error,
            "trace": senv.trace}
