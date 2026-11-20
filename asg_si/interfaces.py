"""Skill-level interfaces and chain checking.

An interface is a pair (Pre, Post) of partial assignments over declared semantic tags
(for example unit = "L", validated = True). Post values may be the token INHERIT, meaning the
tag passes through unchanged from the skill's input. A chain s_1 ... s_k is accepted iff, after
propagating Post assignments left to right, every tag required by Pre(s_i) holds in the state
produced by s_{i-1}. This is the operational form of composition-ok(e): Post(A) satisfies Pre(B)
on every declared tag. It refutes incompatibilities that are declared; it does not prove
correctness for undeclared properties.
"""
INHERIT = "__inherit__"


def make_interface(pre=None, post=None):
    return {"pre": dict(pre or {}), "post": dict(post or {})}


def check_chain(interfaces, initial=None):
    state = dict(initial or {})
    for i, iface in enumerate(interfaces):
        for tag, required in iface["pre"].items():
            if state.get(tag) != required:
                return {"ok": False, "edge": (i - 1, i), "tag": tag, "required": required, "found": state.get(tag)}
        state = {tag: (state.get(tag) if value == INHERIT else value) for tag, value in iface["post"].items()}
    return {"ok": True, "state": state}
