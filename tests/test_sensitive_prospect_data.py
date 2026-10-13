import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ["LANGSMITH_TRACING"] = "false"

from gtm_agent import data_service, gtm_agent


class SensitiveProspectDataTests(unittest.TestCase):
    prospect_id = "LEAD-TEST-SENSITIVE"
    billing_qualification = {
        "tax_id": "test-tax-id",
        "date_of_birth": "1990-01-01",
        "card_on_file": "test-card-number",
        "credit_check_ref": "test-credit-ref",
    }

    def setUp(self):
        data_service._PROFILES.pop(self.prospect_id, None)

    def tearDown(self):
        data_service._PROFILES.pop(self.prospect_id, None)

    def prospect_record(self):
        return {
            "name": "Test Prospect",
            "email": "test@example.com",
            "annual_revenue": 1000000,
            "billing_qualification": dict(self.billing_qualification),
            "engagement_history": ["engaged"],
            "account_details": {"industry": "software"},
            "tech_stack": ["python"],
        }

    def assert_no_billing_data(self, value):
        self.assertNotIn("billing_qualification", value)
        serialized = json.dumps(value)
        for field, data in self.billing_qualification.items():
            self.assertNotIn(field, serialized)
            self.assertNotIn(data, serialized)

    def test_get_prospect_excludes_billing_and_enrichment_fields(self):
        with patch.object(data_service, "get_prospect_record", return_value=self.prospect_record()):
            result = gtm_agent.get_prospect.func(self.prospect_id)

        prospect = result["prospect"]
        self.assertTrue(result["found"])
        self.assertEqual(prospect["name"], "Test Prospect")
        self.assertEqual(prospect["email"], "test@example.com")
        self.assertEqual(prospect["annual_revenue"], 1000000)
        self.assertNotIn("engagement_history", prospect)
        self.assertNotIn("account_details", prospect)
        self.assertNotIn("tech_stack", prospect)
        self.assert_no_billing_data(prospect)

    def test_build_profile_excludes_billing_before_return_and_cache(self):
        record = self.prospect_record()
        with (
            patch.object(data_service, "get_prospect_record", return_value=record),
            patch.object(data_service, "fetch_engagement_history", return_value=record["engagement_history"]),
            patch.object(data_service, "fetch_account_details", return_value=record["account_details"]),
            patch.object(data_service, "fetch_tech_stack", return_value=record["tech_stack"]),
        ):
            result = gtm_agent.build_prospect_profile.func(self.prospect_id)

        profile = result["prospect_profile"]
        self.assertTrue(result["found"])
        self.assertEqual(profile["name"], "Test Prospect")
        self.assert_no_billing_data(profile)
        self.assert_no_billing_data(data_service._PROFILES[self.prospect_id])

    def test_build_profile_sanitizes_existing_cached_profile(self):
        data_service.save_profile_to_db(self.prospect_id, self.prospect_record())

        result = gtm_agent.build_prospect_profile.func(self.prospect_id)

        self.assert_no_billing_data(result["prospect_profile"])
        self.assert_no_billing_data(data_service._PROFILES[self.prospect_id])

    def test_score_serializes_only_allowlisted_fields_and_returns_score(self):
        profile = {
            "prospect_id": self.prospect_id,
            "name": "Test Prospect",
            "account_details": {"industry": "software"},
            "annual_revenue": 1000000,
            "tech_stack": ["python"],
            "engagement_history": ["engaged"],
            "email": "test@example.com",
            "billing_qualification": dict(self.billing_qualification),
        }
        offering = {
            "required_tech_stack": ["python"],
            "min_annual_revenue": 500000,
            "description": "A test offering",
        }
        model_result = SimpleNamespace(model_dump=lambda: {"score": 90})
        scoring_llm = SimpleNamespace(invoke=Mock(return_value=model_result))
        with (
            patch.object(data_service, "fetch_tech_stack", return_value=["python"]),
            patch.object(gtm_agent, "_scoring_llm", scoring_llm),
        ):
            result = gtm_agent.score_prospect.func(profile, offering)

        self.assertEqual(result["score"], 90)
        serialized_profile = scoring_llm.invoke.call_args.args[0][1]["content"].split(
            "\n\nProspect profile:\n", 1
        )[1]
        self.assertEqual(
            set(json.loads(serialized_profile)),
            {
                "prospect_id",
                "name",
                "account_details",
                "annual_revenue",
                "tech_stack",
                "engagement_history",
            },
        )
        self.assert_no_billing_data(serialized_profile)


if __name__ == "__main__":
    unittest.main()
