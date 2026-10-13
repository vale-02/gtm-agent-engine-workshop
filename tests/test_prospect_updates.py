import importlib
import os
import unittest
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("OPENAI_API_KEY", "test")
os.environ["LANGSMITH_TRACING"] = "false"

data_service = importlib.import_module("gtm_agent.data_service")
gtm_agent = importlib.import_module("gtm_agent.gtm_agent")
records = importlib.import_module("gtm_agent.gtm_records")
OFFERINGS = records.OFFERINGS
PROSPECTS = records.PROSPECTS

os.environ["LANGSMITH_TRACING"] = "false"


class ProspectUpdateTests(unittest.TestCase):
    prospect_id = "LEAD-71001"

    def setUp(self):
        self.original_record = deepcopy(PROSPECTS[self.prospect_id])
        self.had_cached_profile = self.prospect_id in data_service._PROFILES
        self.original_profile = data_service._PROFILES.get(self.prospect_id)
        data_service._PROFILES.pop(self.prospect_id, None)

    def tearDown(self):
        PROSPECTS[self.prospect_id] = self.original_record
        if self.had_cached_profile:
            data_service._PROFILES[self.prospect_id] = self.original_profile
        else:
            data_service._PROFILES.pop(self.prospect_id, None)

    def test_update_persists_technology_and_fetches_saved_stack(self):
        result = data_service.update_prospect_info(self.prospect_id, "Kafka")

        self.assertTrue(result["updated"])
        self.assertTrue(result["found"])
        self.assertIn("Kafka", PROSPECTS[self.prospect_id]["tech_stack"])
        self.assertEqual(result["tech_stack"], PROSPECTS[self.prospect_id]["tech_stack"])
        self.assertIn("Kafka", data_service.fetch_tech_stack(self.prospect_id))

    def test_update_invalidates_profile_and_rescoring_uses_updated_stack(self):
        initial_profile = gtm_agent.build_prospect_profile.invoke(
            {"prospect_id": self.prospect_id}
        )["prospect_profile"]
        self.assertNotIn("Kafka", initial_profile["tech_stack"])

        class ScoringStub:
            def invoke(self, messages):
                prospect_profile = messages[1]["content"].split(
                    "\n\nProspect profile:\n", 1
                )[1]
                score = 90 if '"Kafka"' in prospect_profile else 75
                return SimpleNamespace(model_dump=lambda: {"score": score})

        with patch.object(gtm_agent, "_scoring_llm", ScoringStub()):
            offering = OFFERINGS["OFFER-10005"]
            original_score = gtm_agent.score_prospect.invoke(
                {"prospect_profile": initial_profile, "offering": offering}
            )["score"]

            data_service.update_prospect_info(self.prospect_id, "Kafka")
            self.assertIsNone(
                data_service.get_profile_from_db(self.prospect_id)["prospect_profile"]
            )

            updated_profile = gtm_agent.build_prospect_profile.invoke(
                {"prospect_id": self.prospect_id}
            )["prospect_profile"]
            self.assertIn("Kafka", updated_profile["tech_stack"])
            updated_score = gtm_agent.score_prospect.invoke(
                {"prospect_profile": updated_profile, "offering": offering}
            )["score"]

        self.assertNotEqual(original_score, updated_score)

    def test_update_of_missing_prospect_does_not_create_record(self):
        prospect_id = "LEAD-MISSING"

        result = data_service.update_prospect_info(prospect_id, "Kafka")

        self.assertEqual(result, {"updated": False, "found": False})
        self.assertNotIn(prospect_id, PROSPECTS)
        self.assertNotIn(prospect_id, data_service._PROFILES)

    def test_duplicate_technology_remains_single_and_returns_persisted_stack(self):
        data_service.update_prospect_info(self.prospect_id, "Kafka")

        result = data_service.update_prospect_info(self.prospect_id, "Kafka")

        saved_stack = PROSPECTS[self.prospect_id]["tech_stack"]
        self.assertEqual(saved_stack.count("Kafka"), 1)
        self.assertEqual(result["tech_stack"], saved_stack)


if __name__ == "__main__":
    unittest.main()
