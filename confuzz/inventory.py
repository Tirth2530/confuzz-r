"""Domain #2: regional inventory network (the same engine, a different system).

A distributor holds 4 SKUs at 3 sites over a 7-day incident horizon. Two SKUs are critical
(cold-chain insulin, infant formula). Disruptions change demand, supply lead times, cold storage,
inbound docks and transfers. A fixed operations playbook reacts (evacuate cold stock when a site loses
refrigeration, then rebalance critical SKUs from surplus sites to deficit sites while transfers run).
Invariants are then checked and returned as a robustness vector, exactly like the conference simulator.

Hard invariants (normalized margin < 0 = violated):
  stockout      critical SKU at a site: (available over horizon - demand over horizon) / demand
  cold_chain    insulin that could not be kept refrigerated (spoiled) -> -1, else +1
  storage       (storage capacity - units held) / capacity at each site
Severity: stockout 1.0 (capped shortfall fraction), cold_chain 1.0, storage 0.6 (operator weights).
Nothing references specific disruption ids: failures emerge from stock, demand, lead times and capacity.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, replace

from .sim import SimResult
from .world import Disruption

CONSTRAINTS = ("stockout", "cold_chain", "storage")
SEVERITY_WEIGHTS = {"stockout": 1.0, "cold_chain": 1.0, "storage": 0.6}
COLD_SKUS_KEY = "cold"


@dataclass(frozen=True)
class Site:
    id: str
    name: str
    storage: int
    cold: int


@dataclass(frozen=True)
class Sku:
    id: str
    name: str
    critical: bool
    cold: bool


@dataclass(frozen=True)
class Line:
    site: str
    sku: str
    stock: int
    demand: float          # units per day
    inbound: int           # units on the next scheduled delivery
    lead: int              # days until that delivery arrives
    supplier: str


@dataclass(frozen=True)
class InvWorld:
    id: str
    name: str
    sites: tuple[Site, ...]
    skus: tuple[Sku, ...]
    lines: tuple[Line, ...]
    horizon: int
    disruptions: tuple[Disruption, ...]
    rented_cold: tuple[tuple[str, int], ...] = ()       # (site, extra cold units, self-powered)
    dual_sourced: frozenset[str] = frozenset()          # SKUs with a second supplier (immune to supplier delay)
    expedited: tuple[tuple[str, str, int], ...] = ()    # (site, sku, units arriving day 1)
    domain: str = "inventory"

    def disruption(self, did: str) -> Disruption:
        for d in self.disruptions:
            if d.id == did:
                return d
        raise KeyError(did)

    @property
    def disruption_ids(self) -> list[str]:
        return [d.id for d in self.disruptions]

    def site(self, sid: str) -> Site:
        return next(s for s in self.sites if s.id == sid)

    def sku(self, kid: str) -> Sku:
        return next(k for k in self.skus if k.id == kid)

    def line(self, sid: str, kid: str) -> Line:
        return next(l for l in self.lines if l.site == sid and l.sku == kid)

    def public_spec(self) -> dict:
        return {
            "world_id": self.id, "name": self.name, "domain": "inventory", "horizon_days": self.horizon,
            "sites": [asdict(s) for s in self.sites], "skus": [asdict(k) for k in self.skus],
            "stock_lines": [asdict(l) for l in self.lines],
            "rented_cold_storage": [{"site": s, "units": u} for s, u in self.rented_cold],
            "dual_sourced_skus": sorted(self.dual_sourced),
            "expedited_orders": [{"site": s, "sku": k, "units": u} for s, k, u in self.expedited],
            "disruptions": [asdict(d) for d in self.disruptions],
            "operations_playbook": PLAYBOOK_TEXT, "constraints": CONSTRAINT_TEXT,
        }


PLAYBOOK_TEXT = (
    "1) If a site loses refrigeration, its insulin is moved to other sites with spare cold capacity, "
    "largest spare first; insulin that cannot be placed spoils. Transfers are impossible during a carrier "
    "strike. 2) For each critical SKU, sites short over the horizon receive stock from sites with surplus "
    "(donors keep their own horizon demand), largest deficit first, largest surplus first, unless a "
    "carrier strike stops transfers. Rebalanced insulin to a site without refrigeration is delivered "
    "daily (just in time), so it needs no cold storage at the receiver. 3) Deliveries arrive only if their lead time is within the horizon "
    "and the site's inbound dock is open."
)
CONSTRAINT_TEXT = {
    "stockout": "every critical SKU at every site covers its demand over the horizon",
    "cold_chain": "no insulin spoils for lack of refrigeration",
    "storage": "units held at a site fit its storage capacity",
}

ACTION_TYPES = ("TRANSFER_STOCK", "EXPEDITE_ORDER", "RENT_COLD_STORAGE", "DUAL_SOURCE", "RAISE_SAFETY_STOCK", "DO_NOTHING")
QTYS = (100, 200, 400)
COST = {"TRANSFER_STOCK": 1.0, "RAISE_SAFETY_STOCK": 2.0, "RENT_COLD_STORAGE": 3.0, "EXPEDITE_ORDER": 4.0,
        "DUAL_SOURCE": 5.0, "DO_NOTHING": 0.0}
RENTED_COLD_UNITS = 300
EXPEDITE_UNITS = 200


def simulate(world: InvWorld, perturbation, demand_noise: dict[str, float] | None = None) -> SimResult:
    pert = tuple(sorted(set(perturbation)))
    dis = [world.disruption(d) for d in pert]
    T = world.horizon
    demand_factor: dict[tuple[str, str], float] = {}
    supplier_delay: dict[str, int] = {}
    power_out, dock_closed, recall = set(), set(), {}
    strike = False
    for d in dis:
        if d.kind == "DEMAND_SPIKE":            # target "SITE:SKU", magnitude factor
            s, k = d.target.split(":")
            demand_factor[(s, k)] = demand_factor.get((s, k), 1.0) * d.magnitude
        elif d.kind == "REGION_DEMAND":         # target SKU, all sites
            for site in world.sites:
                demand_factor[(site.id, d.target)] = demand_factor.get((site.id, d.target), 1.0) * d.magnitude
        elif d.kind == "SUPPLIER_DELAY":        # target supplier, magnitude days
            supplier_delay[d.target] = supplier_delay.get(d.target, 0) + int(d.magnitude)
        elif d.kind == "POWER_OUTAGE":          # target site: loses its own refrigeration
            power_out.add(d.target)
        elif d.kind == "DOCK_CLOSED":           # target site: inbound deliveries stop
            dock_closed.add(d.target)
        elif d.kind == "RECALL":                # target "SITE:SKU", magnitude fraction of stock lost
            s, k = d.target.split(":")
            recall[(s, k)] = recall.get((s, k), 0.0) + d.magnitude
        elif d.kind == "CARRIER_STRIKE":
            strike = True

    rented = dict(world.rented_cold)
    cold_cap = {s.id: (0 if s.id in power_out else s.cold) + rented.get(s.id, 0) for s in world.sites}
    stock, need, inbound = {}, {}, {}
    for l in world.lines:
        key = (l.site, l.sku)
        stock[key] = max(0, int(round(l.stock * (1 - recall.get(key, 0.0)))))
        dem = l.demand * demand_factor.get(key, 1.0) * (demand_noise or {}).get(f"{l.site}:{l.sku}", 1.0)
        need[key] = int(math.ceil(dem * T - 1e-9))
        delay = 0 if l.sku in world.dual_sourced else supplier_delay.get(l.supplier, 0)
        arrives = l.lead + delay <= T and l.site not in dock_closed
        inbound[key] = l.inbound if arrives else 0
    for s, k, u in world.expedited:          # expedited air freight: arrives day 1, bypasses supplier delay
        inbound[(s, k)] = inbound.get((s, k), 0) + (u if s not in dock_closed else 0)

    moves: dict[str, tuple[str, str | None]] = {}
    spoiled: dict[str, int] = {}
    cold_skus = [k.id for k in world.skus if k.cold]
    # 1) cold evacuation
    for s in world.sites:
        held = sum(stock[(s.id, k)] for k in cold_skus)
        excess = held - cold_cap[s.id]
        if excess <= 0:
            continue
        for k in cold_skus:
            if excess <= 0:
                break
            take = min(excess, stock[(s.id, k)])
            stock[(s.id, k)] -= take
            excess -= take
            remaining = take
            if not strike:
                others = sorted((o for o in world.sites if o.id != s.id),
                                key=lambda o: -(cold_cap[o.id] - sum(stock[(o.id, c)] for c in cold_skus)))
                for o in others:
                    spare = cold_cap[o.id] - sum(stock[(o.id, c)] for c in cold_skus)
                    mv = min(spare, remaining)
                    if mv > 0:
                        stock[(o.id, k)] += mv
                        remaining -= mv
                        moves[f"evacuate {k} {s.id}->{o.id}"] = (s.id, o.id)
            if remaining > 0:
                spoiled[s.id] = spoiled.get(s.id, 0) + remaining
                moves[f"spoil {k} at {s.id}"] = (s.id, None)
    # 2) critical rebalancing
    if not strike:
        for k in [x.id for x in world.skus if x.critical]:
            avail = {s.id: stock[(s.id, k)] + inbound[(s.id, k)] for s in world.sites}
            deficits = sorted(((need[(s, k)] - avail[s], s) for s in avail if avail[s] < need[(s, k)]), reverse=True)
            for gap, s in deficits:
                donors = sorted(((avail[d] - need[(d, k)], d) for d in avail if d != s and avail[d] > need[(d, k)]), reverse=True)
                for surplus, d in donors:
                    if gap <= 0:
                        break
                    mv = min(surplus, gap, stock[(d, k)])  # only on-hand stock can move now
                    if mv <= 0:
                        continue
                    stock[(d, k)] -= mv
                    stock[(s, k)] += mv
                    avail[d] -= mv
                    avail[s] += mv
                    gap -= mv
                    moves[f"rebalance {k} {d}->{s} ({mv})"] = (d, s)

    # 3) evaluate
    items, violations = [], []

    def rec(c, m, info):
        items.append((c, m, info))
        if m < 0:
            violations.append({"constraint": c, **info, "normalized_margin": round(m, 4)})

    for s in world.sites:
        for k in world.skus:
            if not k.critical:
                continue
            key = (s.id, k.id)
            avail = stock[key] + inbound[key]
            m = (avail - need[key]) / need[key] if need[key] > 0 else 1.0
            rec("stockout", m, {"site": s.id, "sku": k.id, "raw": avail - need[key], "unit": "units",
                                "detail": f"{k.name} at {s.name}: available {avail} vs {T}-day demand {need[key]}"})
        rec("cold_chain", -1.0 if spoiled.get(s.id) else 1.0,
            {"site": s.id, "detail": f"{spoiled[s.id]} insulin units spoiled at {s.name}" if spoiled.get(s.id) else "cold chain held"})
        held = sum(stock[(s.id, k.id)] + inbound[(s.id, k.id)] for k in world.skus)
        rec("storage", (s.storage - held) / s.storage, {"site": s.id, "raw": s.storage - held, "unit": "units",
                                                         "detail": f"{s.name}: {held} units vs capacity {s.storage}"})
    margins, raw = {}, {}
    for c in CONSTRAINTS:
        ms = [(m, info) for cc, m, info in items if cc == c]
        m, info = min(ms, key=lambda x: x[0])
        margins[c] = m
        raw[c] = {k: v for k, v in info.items() if k in ("raw", "unit", "site", "sku", "detail")}
    severity = sum(SEVERITY_WEIGHTS[v["constraint"]] * min(1.0, -v["normalized_margin"]) for v in violations)
    return SimResult(pert, margins, raw, violations, bool(violations), severity, {}, moves, items)


# ---------------------------------------------------------------------------
# Repair actions (Action is the shared dataclass from repair.py; inventory uses sku/src/dst/qty/site)
# ---------------------------------------------------------------------------
def describe(world: InvWorld, a) -> str:
    t = a.type
    if t == "TRANSFER_STOCK":
        return f"transfer {a.qty} {a.sku} from {a.src} to {a.dst}"
    if t == "EXPEDITE_ORDER":
        return f"expedite {EXPEDITE_UNITS} {a.sku} to {a.site} by air (arrives day 1)"
    if t == "RENT_COLD_STORAGE":
        return f"rent a self-powered reefer at {a.site} (+{RENTED_COLD_UNITS} cold units)"
    if t == "DUAL_SOURCE":
        return f"dual-source {a.sku} (second supplier)"
    if t == "RAISE_SAFETY_STOCK":
        return f"order {a.qty} extra {a.sku} to {a.site} now"
    return "do nothing"


def action_cost(world: InvWorld, a) -> float:
    return COST[a.type]


def apply_action(world: InvWorld, a) -> InvWorld:
    t = a.type
    if t == "DO_NOTHING":
        return world
    if t == "TRANSFER_STOCK":
        if a.src == a.dst or a.qty not in QTYS:
            raise ValueError("invalid transfer")
        src = world.line(a.src, a.sku)
        if src.stock < a.qty:
            raise ValueError(f"{a.src} holds only {src.stock} {a.sku}")
        lines = tuple(replace(l, stock=l.stock - a.qty) if (l.site, l.sku) == (a.src, a.sku) else
                      replace(l, stock=l.stock + a.qty) if (l.site, l.sku) == (a.dst, a.sku) else l for l in world.lines)
        return replace(world, lines=lines)
    if t == "RAISE_SAFETY_STOCK":
        if a.qty not in QTYS:
            raise ValueError("invalid qty")
        world.line(a.site, a.sku)
        lines = tuple(replace(l, inbound=l.inbound + a.qty) if (l.site, l.sku) == (a.site, a.sku) else l for l in world.lines)
        return replace(world, lines=lines)
    if t == "EXPEDITE_ORDER":
        world.line(a.site, a.sku)
        if any(s == a.site and k == a.sku for s, k, _ in world.expedited):
            raise ValueError("already expedited")
        return replace(world, expedited=world.expedited + ((a.site, a.sku, EXPEDITE_UNITS),))
    if t == "RENT_COLD_STORAGE":
        world.site(a.site)
        if any(s == a.site for s, _ in world.rented_cold):
            raise ValueError("already rented at this site")
        return replace(world, rented_cold=world.rented_cold + ((a.site, RENTED_COLD_UNITS),))
    if t == "DUAL_SOURCE":
        world.sku(a.sku)
        if a.sku in world.dual_sourced:
            raise ValueError("already dual-sourced")
        return replace(world, dual_sourced=world.dual_sourced | {a.sku})
    raise ValueError(f"unknown action type {t}")


def concrete_actions(world: InvWorld, strategy: dict, Action) -> list:
    t = strategy.get("type")
    if t not in ACTION_TYPES:
        raise ValueError(f"type must be one of {ACTION_TYPES}")
    pick = lambda key, pool: [strategy[key]] if strategy.get(key) is not None else pool
    sites = [s.id for s in world.sites]
    skus = [k.id for k in world.skus]
    if t == "DO_NOTHING":
        out = [Action("DO_NOTHING")]
    elif t == "TRANSFER_STOCK":
        out = [Action(t, sku=k, src=s, dst=d, qty=q) for k in pick("sku", skus) for s in pick("src", sites)
               for d in pick("dst", sites) for q in pick("qty", list(QTYS))]
    elif t == "RAISE_SAFETY_STOCK":
        out = [Action(t, sku=k, site=s, qty=q) for k in pick("sku", skus) for s in pick("site", sites) for q in pick("qty", list(QTYS))]
    elif t == "EXPEDITE_ORDER":
        out = [Action(t, sku=k, site=s) for k in pick("sku", skus) for s in pick("site", sites)]
    elif t == "RENT_COLD_STORAGE":
        out = [Action(t, site=s) for s in pick("site", sites)]
    else:
        out = [Action(t, sku=k) for k in pick("sku", skus)]
    valid = []
    for a in out:
        try:
            apply_action(world, a)
            valid.append(a)
        except (ValueError, KeyError, StopIteration):
            pass
    return valid


def noise_keys(world: InvWorld) -> list[str]:
    return [f"{l.site}:{l.sku}" for l in world.lines]


# ---------------------------------------------------------------------------
# Demo world (hand-designed, disclosed). Failures are derived by the simulator.
# ---------------------------------------------------------------------------
def demo_inventory_world() -> InvWorld:
    sites = (Site("N", "North DC", 7000, 520), Site("S", "South DC", 6000, 330), Site("H", "City Hub", 2600, 160))
    skus = (Sku("INS", "Insulin (cold chain)", True, True), Sku("FOR", "Infant formula", True, False),
            Sku("WAT", "Bottled water", False, False), Sku("BAT", "Batteries", False, False))
    L = Line
    lines = (
        L("N", "INS", 300, 20, 120, 4, "ColdPharma"), L("S", "INS", 250, 18, 100, 4, "ColdPharma"), L("H", "INS", 110, 15, 80, 3, "ColdPharma"),
        L("N", "FOR", 420, 30, 200, 5, "NutriCo"), L("S", "FOR", 260, 28, 150, 5, "NutriCo"), L("H", "FOR", 160, 25, 120, 4, "NutriCo"),
        L("N", "WAT", 3000, 300, 1500, 3, "AquaPure"), L("S", "WAT", 2600, 260, 1200, 3, "AquaPure"), L("H", "WAT", 900, 120, 600, 2, "AquaPure"),
        L("N", "BAT", 800, 40, 300, 6, "VoltCell"), L("S", "BAT", 700, 35, 250, 6, "VoltCell"), L("H", "BAT", 300, 20, 120, 4, "VoltCell"),
    )
    D = Disruption
    disruptions = (
        D("D1", "DEMAND_SPIKE", "H:FOR", 1.6, "City Hub formula demand +60% (panic buying)"),
        D("D2", "POWER_OUTAGE", "S", 0, "South DC power outage (refrigeration down)"),
        D("D3", "SUPPLIER_DELAY", "ColdPharma", 5, "ColdPharma insulin shipments delayed 5 days"),
        D("D4", "CARRIER_STRIKE", "-", 0, "Regional carrier strike (no inter-site transfers)"),
        D("D5", "RECALL", "N:FOR", 0.4, "Formula batch recall at North DC (-40% stock)"),
        D("D6", "DOCK_CLOSED", "H", 0, "City Hub inbound dock closed (flooding)"),
        D("D7", "REGION_DEMAND", "WAT", 1.5, "Heatwave: water demand +50% everywhere"),
        D("D8", "SUPPLIER_DELAY", "NutriCo", 4, "NutriCo formula shipments delayed 4 days"),
    )
    return InvWorld("inventory", "Regional distributor demo network (hand-designed)", sites, skus, lines, 7, disruptions)
