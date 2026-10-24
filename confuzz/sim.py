"""Deterministic conference simulator: s' = F(s, delta).

simulate(world, perturbation) applies disruptions, runs the fixed operations
playbook (re-hosting displaced sessions), then evaluates every constraint and
returns a robustness vector. Nothing here references specific disruption ids:
interaction failures emerge from capacities, zones, routes, AV and travel.

Normalized margins (all hard constraints; < 0 means violated):
  capacity       (capacity - demand) / capacity            per hosted session, min taken
  transition     (gap - delay - travel) / gap               per consecutive speaker pair, min
  av, accessibility, room_overlap, speaker_overlap, hosting  +1 satisfied / -1 violated
A constraint with nothing to check has margin +1.

Severity (operator-chosen weights, documented, not a probability):
  sum over violations of weight[c] * min(1, shortfall), shortfall = -normalized margin
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .world import Room, Session, World

CONSTRAINTS = ("capacity", "room_overlap", "speaker_overlap", "av", "accessibility", "transition", "hosting")
SEVERITY_WEIGHTS = {
    "hosting": 1.0, "accessibility": 1.0, "room_overlap": 1.0, "speaker_overlap": 1.0,
    "capacity": 0.8, "av": 0.6, "transition": 0.5,
}


@dataclass
class SimResult:
    perturbation: tuple[str, ...]
    margins: dict[str, float]                 # normalized min margin per constraint
    raw: dict[str, dict]                      # per constraint: raw value, unit, where
    violations: list[dict]
    critical: bool
    severity: float
    assignment: dict[str, str | None]         # session -> room after playbook
    moved: dict[str, tuple[str, str | None]]  # session -> (from, to)

    def to_dict(self) -> dict:
        return {
            "perturbation": list(self.perturbation),
            "critical_failure": self.critical,
            "hard_violation_count": len(self.violations),
            "severity": round(self.severity, 4),
            "robustness_margins": {k: round(v, 4) for k, v in self.margins.items()},
            "raw_margins": self.raw,
            "violations": self.violations,
            "playbook_moves": {s: {"from": a, "to": b} for s, (a, b) in self.moved.items()},
        }


def _room_access_ok(r: Room, closed_zones: set[str]) -> bool:
    return r.step_free and r.zone not in closed_zones


def simulate(world: World, perturbation, demand_noise: dict[str, float] | None = None) -> SimResult:
    pert = tuple(sorted(set(perturbation)))
    dis = [world.disruption(d) for d in pert]
    lost = {d.target for d in dis if d.kind == "ROOM_LOSS"}
    closed = {d.target for d in dis if d.kind == "ROUTE_CLOSED"}
    av_failed = {d.target for d in dis if d.kind == "AV_FAIL"}
    delay: dict[str, float] = {}
    surge: dict[str, float] = {}
    for d in dis:
        if d.kind == "SPEAKER_DELAY":
            delay[d.target] = delay.get(d.target, 0) + d.magnitude
        elif d.kind == "SURGE":
            surge[d.target] = surge.get(d.target, 1.0) * d.magnitude

    demand = {}
    for s in world.sessions:
        x = s.demand * surge.get(s.id, 1.0) * (demand_noise or {}).get(s.id, 1.0)
        demand[s.id] = int(math.ceil(x - 1e-9))

    usable = {r.id: r for r in world.usable_rooms() if r.id not in lost}
    room_av = {rid: r.av and rid not in av_failed for rid, r in usable.items()}
    room_acc = {rid: _room_access_ok(r, closed) for rid, r in usable.items()}

    # ---- operations playbook: re-host displaced sessions -----------------
    assignment: dict[str, str | None] = {}
    moved: dict[str, tuple[str, str | None]] = {}
    for t in range(world.n_slots):
        in_slot = [s for s in world.sessions if s.slot == t]
        occupied = set()
        displaced = []
        for s in in_slot:
            if s.room in usable:
                assignment[s.id] = s.room
                occupied.add(s.room)
            else:
                displaced.append(s)
        for s in sorted(displaced, key=lambda s: (-demand[s.id], s.id)):
            free = [rid for rid in usable if rid not in occupied]
            if not free:
                assignment[s.id] = None
                moved[s.id] = (s.room, None)
                continue

            def key(rid: str):
                r = usable[rid]
                fits = demand[s.id] <= r.capacity
                return (
                    s.needs_access and not room_acc[rid],
                    s.needs_av and not room_av[rid],
                    not fits,
                    (r.capacity - demand[s.id]) if fits else (demand[s.id] - r.capacity),
                    rid,
                )
            best = min(free, key=key)
            assignment[s.id] = best
            occupied.add(best)
            moved[s.id] = (s.room, best)

    # ---- constraint evaluation ------------------------------------------
    per: dict[str, list[tuple[float, dict]]] = {c: [] for c in CONSTRAINTS}
    violations: list[dict] = []

    def record(c: str, norm: float, info: dict):
        per[c].append((norm, info))
        if norm < 0:
            violations.append({"constraint": c, **info, "normalized_margin": round(norm, 4)})

    room_slot: dict[tuple[str, int], list[str]] = {}
    spk_slot: dict[tuple[str, int], list[str]] = {}
    for s in world.sessions:
        rid = assignment[s.id]
        if rid is None:
            record("hosting", -1.0, {"session": s.id, "detail": "no free room in slot"})
            continue
        record("hosting", 1.0, {"session": s.id})
        r = usable[rid]
        room_slot.setdefault((rid, s.slot), []).append(s.id)
        spk_slot.setdefault((s.speaker, s.slot), []).append(s.id)
        cap_m = (r.capacity - demand[s.id]) / r.capacity
        record("capacity", cap_m, {"session": s.id, "room": rid, "raw": r.capacity - demand[s.id],
                                   "unit": "seats", "detail": f"capacity {r.capacity}, demand {demand[s.id]}"})
        if s.needs_av:
            ok = room_av[rid]
            record("av", 1.0 if ok else -1.0, {"session": s.id, "room": rid,
                                               "detail": "AV ok" if ok else f"room {rid} has no working AV"})
        if s.needs_access:
            ok = room_acc[rid]
            why = "step-free ok" if ok else (f"zone {r.zone} step-free route closed" if r.zone in closed else f"room {rid} not step-free")
            record("accessibility", 1.0 if ok else -1.0, {"session": s.id, "room": rid, "detail": why})

    for (rid, t), ss in room_slot.items():
        record("room_overlap", 1.0 if len(ss) == 1 else -1.0, {"room": rid, "slot": t, "sessions": ss})
    for (spk, t), ss in spk_slot.items():
        record("speaker_overlap", 1.0 if len(ss) == 1 else -1.0, {"speaker": spk, "slot": t, "sessions": ss})

    by_speaker: dict[str, list[Session]] = {}
    for s in world.sessions:
        by_speaker.setdefault(s.speaker, []).append(s)
    for spk, ss in by_speaker.items():
        ss = sorted(ss, key=lambda s: s.slot)
        for a, b in zip(ss, ss[1:]):
            if b.slot != a.slot + 1 or assignment[a.id] is None or assignment[b.id] is None:
                continue
            za, zb = usable[assignment[a.id]].zone, usable[assignment[b.id]].zone
            travel = world.travel_min(za, zb)
            raw = world.slot_gap_min - delay.get(spk, 0) - travel
            record("transition", raw / world.slot_gap_min, {
                "speaker": spk, "sessions": [a.id, b.id], "raw": raw, "unit": "minutes",
                "detail": f"gap {world.slot_gap_min} - delay {delay.get(spk, 0):g} - travel {za}->{zb} {travel}"})

    margins, raw = {}, {}
    for c in CONSTRAINTS:
        if per[c]:
            m, info = min(per[c], key=lambda x: x[0])
            margins[c] = m
            raw[c] = {k: v for k, v in info.items() if k in ("raw", "unit", "session", "room", "speaker", "sessions", "detail")}
        else:
            margins[c] = 1.0
            raw[c] = {}

    severity = sum(SEVERITY_WEIGHTS[v["constraint"]] * min(1.0, -v["normalized_margin"]) for v in violations)
    return SimResult(pert, margins, raw, violations, bool(violations), severity, assignment, moved)
