"""Conference world schema, the hand-designed demo world, and the seeded world generator.

A World is a schedule (sessions placed in rooms and slots) plus a finite set of
atomic disruptions. A perturbation is any subset of disruption ids, so with d
disruptions there are 2^d perturbations. d is kept small (8) on purpose: it lets
an offline evaluator enumerate every perturbation and know the true failure
structure. Testing policies never see that enumeration.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field, replace
from typing import Iterable


@dataclass(frozen=True)
class Room:
    id: str
    name: str
    capacity: int
    zone: str
    av: bool = True
    step_free: bool = True          # room itself has a step-free entrance
    standby: bool = False           # bookable spare room, unused unless activated by a repair
    standby_cost: float = 0.0


@dataclass(frozen=True)
class Session:
    id: str
    title: str
    slot: int
    room: str
    speaker: str
    demand: int
    needs_av: bool = True
    needs_access: bool = True       # has attendees who need a step-free route


@dataclass(frozen=True)
class Disruption:
    id: str
    kind: str        # SURGE | ROOM_LOSS | ROUTE_CLOSED | AV_FAIL | SPEAKER_DELAY
    target: str      # session id | room id | zone | room id | speaker id
    magnitude: float # demand factor | - | - | - | delay minutes
    label: str


KINDS = ("SURGE", "ROOM_LOSS", "ROUTE_CLOSED", "AV_FAIL", "SPEAKER_DELAY")


@dataclass(frozen=True)
class World:
    id: str
    name: str
    rooms: tuple[Room, ...]
    sessions: tuple[Session, ...]
    n_slots: int
    slot_gap_min: int                       # minutes between consecutive slots
    travel: tuple[tuple[str, str, int], ...]  # (zone_a, zone_b, minutes), symmetric
    disruptions: tuple[Disruption, ...]
    active_standby: frozenset[str] = frozenset()

    # ---- lookups -------------------------------------------------------
    def room(self, rid: str) -> Room:
        for r in self.rooms:
            if r.id == rid:
                return r
        raise KeyError(rid)

    def session(self, sid: str) -> Session:
        for s in self.sessions:
            if s.id == sid:
                return s
        raise KeyError(sid)

    def disruption(self, did: str) -> Disruption:
        for d in self.disruptions:
            if d.id == did:
                return d
        raise KeyError(did)

    @property
    def disruption_ids(self) -> list[str]:
        return [d.id for d in self.disruptions]

    def travel_min(self, za: str, zb: str) -> int:
        for a, b, m in self.travel:
            if {a, b} == {za, zb}:
                return m
        raise KeyError((za, zb))

    def usable_rooms(self) -> list[Room]:
        return [r for r in self.rooms if (not r.standby) or r.id in self.active_standby]

    def with_sessions(self, sessions: Iterable[Session]) -> "World":
        return replace(self, sessions=tuple(sessions))

    # ---- serialization (what agents may see) ----------------------------
    def public_spec(self) -> dict:
        """Everything an organizer or agent may know. Never includes evaluation results."""
        return {
            "world_id": self.id,
            "name": self.name,
            "n_slots": self.n_slots,
            "slot_gap_min": self.slot_gap_min,
            "travel_minutes": [{"zones": [a, b], "minutes": m} for a, b, m in self.travel],
            "rooms": [asdict(r) | {"active": (not r.standby) or r.id in self.active_standby} for r in self.rooms],
            "sessions": [asdict(s) for s in self.sessions],
            "disruptions": [asdict(d) for d in self.disruptions],
            "operations_playbook": PLAYBOOK_TEXT,
            "constraints": CONSTRAINT_TEXT,
        }

    def to_json(self) -> str:
        return json.dumps(self.public_spec(), indent=2)


PLAYBOOK_TEXT = (
    "When a room becomes unavailable, every session in it is re-hosted, in that slot, "
    "in a free usable room. Displaced sessions are handled largest-demand first. Among "
    "free rooms the playbook prefers, in order: rooms that meet the step-free requirement, "
    "rooms with working AV, rooms that fit the demand, then the tightest fit. If no room "
    "is free the session is unhosted."
)

CONSTRAINT_TEXT = {
    "capacity": "effective room capacity >= session demand",
    "room_overlap": "at most one session per room per slot",
    "speaker_overlap": "a speaker gives at most one session per slot",
    "av": "sessions needing AV are in a room with working AV",
    "accessibility": "sessions needing step-free access are in a step-free room whose zone route is open",
    "transition": "a speaker with sessions in consecutive slots has gap - delay - travel >= 0 minutes",
    "hosting": "every session is hosted in some room",
}


# ---------------------------------------------------------------------------
# Hand-designed demo world. Designed for legibility on stage; disclosed as such.
# Failures are NOT coded anywhere: they are derived by the simulator from
# capacities, zones, routes, AV and travel times.
# ---------------------------------------------------------------------------
def demo_world() -> World:
    rooms = (
        Room("A", "Grand Hall A", 200, "North"),
        Room("B", "Room B", 120, "North"),
        Room("C", "Room C", 90, "South"),
        Room("D", "Hall D", 220, "South"),
        Room("E", "Room E", 160, "North"),
        Room("F", "Ballroom F (standby)", 260, "North", standby=True, standby_cost=5.0),
    )
    S = Session
    sessions = (
        S("S01", "Opening Remarks", 0, "A", "Okafor", 150),
        S("S02", "Platform Engineering 101", 0, "B", "Lindqvist", 100),
        S("S03", "Rust at the Edge", 0, "C", "Haddad", 70, needs_access=False),
        S("S04", "State of Open Source", 1, "A", "Moreau", 150),
        S("S05", "GPU Scheduling Deep Dive", 1, "D", "Tanaka", 180, needs_access=False),
        S("S06", "Accessible UX Patterns", 1, "B", "Ferreira", 90),
        S("S07", "Keynote: Agents in Production", 2, "A", "Patel", 140),
        S("S08", "Observability for LLM Apps", 2, "B", "Nguyen", 110),
        S("S09", "Postgres Internals", 2, "C", "Kowalski", 60, needs_access=False),
        S("S10", "Agents Q&A Panel", 3, "E", "Patel", 120, needs_av=False),
        S("S11", "Supply-Chain Security", 3, "A", "Abara", 110),
        S("S12", "Data Contracts", 3, "D", "Svensson", 150, needs_access=False),
    )
    D = Disruption
    disruptions = (
        D("D1", "SURGE", "S07", 1.36, "Keynote demand surge (+36%)"),
        D("D2", "ROOM_LOSS", "A", 0, "Grand Hall A unavailable (HVAC failure)"),
        D("D3", "ROUTE_CLOSED", "South", 0, "South-wing step-free route closed (lift outage)"),
        D("D4", "SPEAKER_DELAY", "Patel", 5, "Keynote speaker runs 5 min late"),
        D("D5", "AV_FAIL", "E", 0, "Room E projector failure"),
        D("D6", "ROOM_LOSS", "B", 0, "Room B unavailable"),
        D("D7", "SURGE", "S05", 1.15, "GPU deep dive demand surge (+15%)"),
        D("D8", "SURGE", "S12", 1.20, "Data Contracts demand surge (+20%)"),
    )
    return World(
        id="demo",
        name="DevConf demo world (hand-designed)",
        rooms=rooms,
        sessions=sessions,
        n_slots=4,
        slot_gap_min=15,
        travel=(("North", "North", 3), ("South", "South", 3), ("North", "South", 12)),
        disruptions=disruptions,
    )


# ---------------------------------------------------------------------------
# Seeded generator for benchmark worlds.
# Acceptance rule (fixed in advance, independent of any testing method):
#   the undisrupted schedule passes, EVERY single disruption passes, and at
#   least one critical failure exists. The study is about interaction failures;
#   a world where one disruption alone breaks the schedule is caught by any
#   single-disruption check and is out of scope. (v1 of the generator omitted the
#   single-disruption rule; 19 of 20 worlds then had only first-order failures and
#   size-ordered testing was best. That result is kept in experiments/results.)
# ---------------------------------------------------------------------------
_TITLES = [
    "Kubernetes Cost Control", "WASM in Production", "Vector Search at Scale", "Incident Retros",
    "Typed Python", "Edge Caching", "Feature Flags", "LLM Evals", "Zero-Trust Networking",
    "Design Systems", "Event Sourcing", "Mobile Perf", "SRE Burnout", "Graph Databases",
    "Fuzzing 101", "CI at Monorepo Scale", "Privacy Engineering", "Streaming SQL",
]


def generate_world(seed: int) -> World | None:
    """One candidate world for this seed, or None if it fails the acceptance rule."""
    from .sim import simulate  # local import to avoid a cycle

    rng = random.Random(seed)
    zones = ["North", "South"]
    n_rooms = 5
    caps = sorted(rng.sample(range(70, 240, 10), n_rooms), reverse=True)
    rooms = []
    for i, cap in enumerate(caps):
        rooms.append(Room(
            id=chr(ord("A") + i), name=f"Room {chr(ord('A') + i)}", capacity=cap,
            zone=zones[0] if i % 2 == 0 else zones[1] if rng.random() < 0.7 else zones[0],
            av=rng.random() < 0.85, step_free=rng.random() < 0.9,
        ))
    rooms.append(Room("F", "Standby Ballroom F", rng.choice([240, 260, 280]), "North", standby=True, standby_cost=5.0))

    n_slots, per_slot = 4, 3
    speakers = [f"Spk{i:02d}" for i in range(11)]
    sessions = []
    titles = rng.sample(_TITLES, n_slots * per_slot)
    k = 0
    repeat_speaker = speakers[0]
    for t in range(n_slots):
        chosen = rng.sample([r for r in rooms if not r.standby], per_slot)
        for j, r in enumerate(chosen):
            demand = int(r.capacity * rng.uniform(0.55, 0.92))
            spk = repeat_speaker if (t in (1, 2) and j == 0) else speakers[1 + (k % 10)]
            sessions.append(Session(
                id=f"S{k + 1:02d}", title=titles[k], slot=t, room=r.id, speaker=spk, demand=demand,
                needs_av=r.av and rng.random() < 0.8, needs_access=r.step_free and rng.random() < 0.75,
            ))
            k += 1

    by_demand = sorted(sessions, key=lambda s: -s.demand)
    non_standby = [r for r in rooms if not r.standby]
    room_loss = rng.sample(non_standby[:3], 2)
    av_room = rng.choice([r for r in non_standby if r.av] or non_standby)
    closed_zone = rng.choice(zones)
    # Organizer pre-mitigation of known single points of failure (disclosed):
    # step-free attendees are not scheduled into the zone with the fragile route,
    # and the room with fragile AV hosts no AV-dependent session.
    sessions = [replace(s,
                        needs_access=s.needs_access and next(r for r in rooms if r.id == s.room).zone != closed_zone,
                        needs_av=s.needs_av and s.room != av_room.id)
                for s in sessions]
    by_demand = sorted(sessions, key=lambda s: -s.demand)
    disruptions = [
        Disruption("D1", "SURGE", by_demand[0].id, round(rng.uniform(1.15, 1.45), 2), f"Demand surge on {by_demand[0].title}"),
        Disruption("D2", "ROOM_LOSS", room_loss[0].id, 0, f"{room_loss[0].name} unavailable"),
        Disruption("D3", "ROUTE_CLOSED", closed_zone, 0, f"{closed_zone} step-free route closed"),
        Disruption("D4", "SPEAKER_DELAY", repeat_speaker, rng.choice([4, 5, 6, 8]), f"{repeat_speaker} runs late"),
        Disruption("D5", "AV_FAIL", av_room.id, 0, f"{av_room.name} AV failure"),
        Disruption("D6", "ROOM_LOSS", room_loss[1].id, 0, f"{room_loss[1].name} unavailable"),
        Disruption("D7", "SURGE", by_demand[1].id, round(rng.uniform(1.10, 1.35), 2), f"Demand surge on {by_demand[1].title}"),
        Disruption("D8", "SURGE", by_demand[2].id, round(rng.uniform(1.10, 1.35), 2), f"Demand surge on {by_demand[2].title}"),
    ]
    world = World(
        id=f"gen-{seed}", name=f"Generated world {seed}", rooms=tuple(rooms), sessions=tuple(sessions),
        n_slots=n_slots, slot_gap_min=rng.choice([12, 15, 15, 20]),
        travel=(("North", "North", 3), ("South", "South", 3), ("North", "South", rng.choice([8, 10, 12]))),
        disruptions=tuple(disruptions),
    )
    if simulate(world, frozenset()).critical:
        return None
    if any(simulate(world, frozenset([d.id])).critical for d in world.disruptions):
        return None
    return world


FROZEN_START = 0
DEV_START = 50_000_000
