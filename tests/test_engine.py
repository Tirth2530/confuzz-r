"""Integrity tests. Run: python -m unittest discover -s tests -v"""
import ast
import os
import tempfile
import unittest
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ["CONFUZZ_STATE"] = os.path.join(tempfile.mkdtemp(), "state.json")

from confuzz.certificate import clopper_pearson_upper, cvar  # noqa: E402
from confuzz.groundtruth import ground_truth, minimal_failing  # noqa: E402
from confuzz.repair import Action, apply_action, verify_exact  # noqa: E402
from confuzz.search import (AdaptiveGroupTesting, BudgetExhausted, Oracle, RandomPolicy, RandomPrunedPolicy,  # noqa: E402
                            minimize, run_episode)
from confuzz.session import RuleViolation, Session  # noqa: E402
from confuzz.sim import simulate  # noqa: E402
from confuzz.benchmark_worlds import frozen_worlds  # noqa: E402
from confuzz.world import DEV_START, FROZEN_START, demo_world  # noqa: E402


class TestSimulator(unittest.TestCase):
    def test_deterministic_replay(self):
        w = demo_world()
        a, b = simulate(w, ["D1", "D2", "D3"]).to_dict(), simulate(w, ["D3", "D2", "D1"]).to_dict()
        self.assertEqual(a, b)

    def test_base_and_singles_pass(self):
        w = demo_world()
        self.assertFalse(simulate(w, []).critical)
        for d in w.disruption_ids:
            self.assertFalse(simulate(w, [d]).critical, d)

    def test_capacity_margin_value(self):
        w = demo_world()
        r = simulate(w, ["D1"])  # keynote 140 * 1.36 = 190.4 -> 191 in 200-seat hall
        self.assertAlmostEqual(r.margins["capacity"], min(r.margins["capacity"], (200 - 191) / 200), places=6)

    def test_transition_margin_value(self):
        w = demo_world()
        r = simulate(w, ["D1", "D2", "D4"])  # keynote re-hosted South, panel North: 15 - 5 - 12 = -2
        tr = [v for v in r.violations if v["constraint"] == "transition"]
        self.assertEqual(len(tr), 1)
        self.assertAlmostEqual(tr[0]["normalized_margin"], -2 / 15, places=4)

    def test_third_order_interaction_emerges(self):
        w = demo_world()
        m = frozenset({"D1", "D2", "D3"})
        self.assertTrue(simulate(w, m).critical)
        for k in (1, 2):
            for c in combinations(sorted(m), k):
                self.assertFalse(simulate(w, c).critical, c)
        self.assertIn(m, ground_truth(w).minimal_sets)

    def test_no_disruption_id_logic_in_simulator(self):
        src = (ROOT / "confuzz" / "sim.py").read_text()
        for d in [f'"D{i}"' for i in range(1, 9)]:
            self.assertNotIn(d, src, "simulator must not reference specific disruption ids")


class TestGroundTruth(unittest.TestCase):
    def test_minimal_definition(self):
        out = {frozenset(): False, frozenset("a"): False, frozenset("b"): False,
               frozenset("ab"): True, frozenset("c"): True, frozenset("ac"): True,
               frozenset("bc"): True, frozenset("abc"): True}
        self.assertEqual(minimal_failing(out), {frozenset("ab"), frozenset("c")})

    def test_frozen_dev_disjoint(self):
        f = {w.id for w in frozen_worlds(20, FROZEN_START)}
        d = {w.id for w in frozen_worlds(20, DEV_START)}
        self.assertFalse(f & d)


class TestSearch(unittest.TestCase):
    def test_budget_enforced_including_minimization(self):
        w = demo_world()
        o = Oracle(w, 3)
        o.query(["D1", "D2", "D3", "D4"])
        with self.assertRaises(BudgetExhausted):
            minimize(o, frozenset({"D1", "D2", "D3", "D4"}))
        self.assertEqual(o.used, 3)

    def test_cached_query_free(self):
        o = Oracle(demo_world(), 2)
        o.query(["D1"]); o.query(["D1"])
        self.assertEqual(o.used, 1)

    def test_episode_never_exceeds_budget(self):
        w = demo_world()
        for pol in (RandomPolicy(0), RandomPrunedPolicy(0), AdaptiveGroupTesting(0)):
            ep = run_episode(w, pol, 12, 0)
            self.assertLessEqual(ep.queries, 12)
            self.assertTrue(all(o.index <= 12 for o in ep.log))

    def test_found_sets_are_true_failures(self):
        w = demo_world()
        ep = run_episode(w, RandomPolicy(1), 40, 1)
        for _, m in ep.found:
            self.assertTrue(simulate(w, m).critical)

    def test_baseline_fairness_same_world_same_budget(self):
        w = demo_world()
        a = run_episode(w, RandomPolicy(3), 15, 3)
        b = run_episode(w, AdaptiveGroupTesting(3), 15, 3)
        self.assertEqual((a.world_id, a.budget), (b.world_id, b.budget))


class TestRepair(unittest.TestCase):
    def test_invalid_move_rejected(self):
        w = demo_world()
        with self.assertRaises(ValueError):
            apply_action(w, Action("MOVE_SESSION", session="S07", room="B"))  # B occupied in slot 2

    def test_swap_repair_has_real_regression(self):
        w = demo_world()
        p = apply_action(w, Action("SWAP_ROOMS", session="S10", session2="S12"))
        v = verify_exact(w, p, [frozenset({"D1", "D2", "D4"})])
        self.assertEqual(v["status"], "REPAIR_REJECTED")
        self.assertIn(["D4"], v["regressions"])

    def test_same_condition_comparison(self):
        w = demo_world()
        p = apply_action(apply_action(w, Action("ACTIVATE_STANDBY", room="F")), Action("MOVE_SESSION", session="S07", room="F"))
        t = frozenset({"D1", "D2", "D3"})
        self.assertTrue(simulate(w, t).critical)
        self.assertFalse(simulate(p, t).critical)


class TestStats(unittest.TestCase):
    def test_clopper_pearson(self):
        self.assertAlmostEqual(clopper_pearson_upper(0, 400), 1 - 0.05 ** (1 / 400), places=6)
        self.assertEqual(clopper_pearson_upper(5, 5), 1.0)

    def test_cvar(self):
        v, k = cvar(list(range(100)), 0.95)
        self.assertEqual(k, 5)
        self.assertAlmostEqual(v, (99 + 98 + 97 + 96 + 95) / 5)


class TestSessionRules(unittest.TestCase):
    def setUp(self):
        self.s = Session.new("demo", budget=20)

    def test_apply_refused_unless_verified(self):
        self.s.minimize(["D1", "D2", "D3"])
        r = self.s.propose_repair(["F1"], [{"type": "ACTIVATE_STANDBY"}], "test", None)
        with self.assertRaises(RuleViolation):
            self.s.apply(r["repair_id"], "x")
        with self.assertRaises(RuleViolation):
            self.s.verify(r["repair_id"])  # no critic attack yet

    def test_revision_fields_required_after_rejection(self):
        self.s.minimize(["D1", "D2", "D4"])
        r = self.s.propose_repair(["F1"], [{"type": "SWAP_ROOMS"}], "cheapest swap", None)
        res = self.s.attack(r["repair_id"], ["D4"], "delay alone now hits a cross-zone transition", "transition")
        self.assertEqual(res["repair_status"], "REPAIR_REJECTED")
        with self.assertRaises(RuleViolation):
            self.s.propose_repair(["F1"], [{"type": "ACTIVATE_STANDBY"}], "retry", None)

    def test_budget_enforced_in_server(self):
        s = Session.new("demo", budget=1)
        s.query(["D1"], "test", "RED")
        with self.assertRaises(RuleViolation):
            s.query(["D2"], "test", "RED")


class TestIsolation(unittest.TestCase):
    AGENT_FACING = ["mcp_server.py", "session.py", "search.py", "repair.py", "certificate.py", "sim.py", "world.py", "llm.py", "space.py"]

    def test_agent_facing_modules_never_import_ground_truth(self):
        for f in self.AGENT_FACING:
            tree = ast.parse((ROOT / "confuzz" / f).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertNotIn("groundtruth", node.module or "", f"{f} imports ground truth")
                if isinstance(node, ast.Import):
                    for n in node.names:
                        self.assertNotIn("groundtruth", n.name, f"{f} imports ground truth")

    def test_world_generator_ground_truth_import_is_evaluator_only(self):
        # frozen_worlds() uses ground truth for the acceptance rule; agents only get public_spec()
        spec = demo_world().public_spec()
        self.assertNotIn("minimal", str(spec).lower())
        self.assertNotIn("failing", str(spec).lower())

    def test_red_prompt_has_no_answers(self):
        from confuzz.llm import red_prompt
        p = red_prompt(demo_world())
        self.assertNotIn("minimal_sets", p)
        self.assertNotIn("world_id", p)


if __name__ == "__main__":
    unittest.main()
