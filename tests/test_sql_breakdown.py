import unittest

from rag.sql_service import SqlQueryService


class FakeRow(dict):
    def __getitem__(self, key):
        return super().__getitem__(key)


class SqlBreakdownTests(unittest.TestCase):
    def test_detects_violation_and_non_violation_breakdown_request(self):
        self.assertTrue(SqlQueryService._requires_commitment_breakdown(
            "查7月5日所有案例多少条，包括违法和不违法"
        ))
        self.assertFalse(SqlQueryService._requires_commitment_breakdown(
            "查7月5日所有案例多少条"
        ))

    def test_retries_when_first_sql_only_returns_total(self):
        service = SqlQueryService.__new__(SqlQueryService)
        generated_sql = iter(["SELECT COUNT(*) FROM results", "SELECT breakdown FROM results"])
        service._to_sql = lambda question, error_hint="": next(generated_sql)

        def execute(sql):
            if "COUNT" in sql:
                return [FakeRow({"COUNT(*)": 5})], ["COUNT(*)"]
            return [FakeRow({
                "total_count": 5,
                "confirmed_violation_count": 2,
                "unconfirmed_violation_count": 3,
            })], [
                "total_count",
                "confirmed_violation_count",
                "unconfirmed_violation_count",
            ]

        service._execute = execute
        payload = service.query_structured("查7月5日所有案例多少条，包括违法和不违法")
        self.assertEqual(payload["rows"][0]["total_count"], 5)
        self.assertEqual(payload["rows"][0]["confirmed_violation_count"], 2)
        self.assertEqual(payload["rows"][0]["unconfirmed_violation_count"], 3)


if __name__ == "__main__":
    unittest.main()
