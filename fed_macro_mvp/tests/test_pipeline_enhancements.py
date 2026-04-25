import unittest
from datetime import datetime, timezone
import json
from pathlib import Path
import tempfile

import pandas as pd

from core.config import PipelineConfig
from core.ingest import apply_catalog_filters
from core.observability import create_recorder
from core.validation import coerce_investor_json, validate_investor_json


class TestPipelineEnhancements(unittest.TestCase):
    def test_profile_defaults_and_switch(self):
        cfg = PipelineConfig(project_dir=Path('fed_macro_mvp'))
        self.assertEqual(cfg.profile_name, 'fast_default')
        self.assertEqual(cfg.days_back, 180)
        self.assertEqual(cfg.max_pdfs, 24)
        self.assertEqual(cfg.allowed_doc_types, ['fomc_minutes', 'mpr'])
        self.assertTrue(str(cfg.diagnostics_dir).endswith('outputs/diagnostics'))

        cfg.set_profile('full_default')
        self.assertEqual(cfg.days_back, 540)
        self.assertEqual(cfg.max_pdfs, 40)
        self.assertEqual(cfg.allowed_doc_types, ['all'])

    def test_catalog_filters_fast_profile_scope(self):
        cfg = PipelineConfig(project_dir=Path('fed_macro_mvp'))
        cfg.set_profile('fast_default')

        raw = pd.DataFrame(
            [
                {
                    'pdf_url': 'https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20260301.pdf',
                    'title': 'Minutes',
                    'date_hint': '2026-03-01',
                    'doc_type': 'fomc_minutes',
                },
                {
                    'pdf_url': 'https://www.federalreserve.gov/monetarypolicy/files/20260201_mprfullreport.pdf',
                    'title': 'MPR',
                    'date_hint': '2026-02-01',
                    'doc_type': 'mpr',
                },
                {
                    'pdf_url': 'https://www.federalreserve.gov/monetarypolicy/files/beigebook20260301.pdf',
                    'title': 'Beige',
                    'date_hint': '2026-03-01',
                    'doc_type': 'other',
                },
                {
                    'pdf_url': 'https://www.federalreserve.gov/monetarypolicy/files/fomcminutes20240101.pdf',
                    'title': 'Old minutes',
                    'date_hint': '2024-01-01',
                    'doc_type': 'fomc_minutes',
                },
            ]
        )

        filtered = apply_catalog_filters(raw, cfg, today_utc=datetime(2026, 3, 18, tzinfo=timezone.utc))
        self.assertEqual(set(filtered['doc_type'].tolist()), {'fomc_minutes', 'mpr'})
        self.assertTrue((pd.to_datetime(filtered['date_hint']) >= pd.Timestamp('2025-09-19')).all())

        cfg.set_profile('full_default')
        full_filtered = apply_catalog_filters(raw, cfg, today_utc=datetime(2026, 3, 18, tzinfo=timezone.utc))
        self.assertIn('other', set(full_filtered['doc_type'].tolist()))

    def test_evidence_minimum_and_quotes(self):
        topic_hits = {
            'inflation': pd.DataFrame([
                {'chunk_id': 'fomcminutes20260301.pdf::chunk0001', 'text': 'Inflation moderated but remained above target.'}
            ]),
            'unemployment': pd.DataFrame(),
            'growth': pd.DataFrame(),
            'policy_rates': pd.DataFrame(),
            'financial_conditions': pd.DataFrame(),
            'credit': pd.DataFrame(),
        }
        valid_ids = {'fomcminutes20260301.pdf::chunk0001'}

        parsed = {
            'executive_summary': 'summary',
            'topic_signals': [{'topic': 'inflation', 'view': 'moderate', 'confidence': '0.7', 'evidence': []}],
            'citations': [{'chunk_id': 'fomcminutes20260301.pdf::chunk0001'}],
        }

        coerced, meta = coerce_investor_json(
            parsed,
            topic_hits,
            valid_ids,
            enforce_topic_min_evidence=True,
            return_meta=True,
        )
        report = validate_investor_json(coerced, valid_ids, enforce_topic_min_evidence=True)

        for sig in coerced['topic_signals']:
            self.assertGreaterEqual(len(sig['evidence']), 1)
        self.assertEqual(report['bad_shape'], [])
        self.assertTrue(coerced['citations'][0]['quote'])
        self.assertGreaterEqual(meta['topic_fallback_injections'], 1)

    def test_observability_recorder_writes_bundle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = PipelineConfig(project_dir=Path(tmpdir))
            recorder = create_recorder(cfg, mode='analysis_only')
            recorder.emit(
                'stage_completed',
                stage='retrieval',
                status='ok',
                payload={'topic': 'inflation', 'hit_count': 2},
                duration_ms=12.5,
            )
            recorder.write_topic_retrieval(
                [{'topic': 'inflation', 'hit_count': 2, 'top_chunk_id': 'doc.pdf::chunk0001'}]
            )
            recorder.write_generation_attempts(
                pd.DataFrame([{'attempt': 1, 'status': 'ok', 'latency_s': 1.2}])
            )
            recorder.write_validation_summary({'quality': {'missing_topics': []}})
            recorder.add_artifact('events', Path(tmpdir) / 'dummy.json')
            summary = recorder.finalize_run('ok', payload={'question': 'test'})

            run_dir = Path(recorder.diagnostics_paths()['run_dir'])
            self.assertTrue((run_dir / 'events.jsonl').exists())
            self.assertTrue((run_dir / 'summary.json').exists())
            self.assertTrue((run_dir / 'topic_retrieval.csv').exists())
            self.assertTrue((run_dir / 'generation_attempts.csv').exists())
            self.assertTrue((run_dir / 'validation_summary.json').exists())
            self.assertTrue((run_dir / 'artifacts_manifest.json').exists())
            self.assertEqual(summary['run_id'], recorder.run_id)

            events = (run_dir / 'events.jsonl').read_text().strip().splitlines()
            self.assertGreaterEqual(len(events), 2)
            payload = json.loads(events[0])
            self.assertIn('event_type', payload)
            self.assertIn('run_id', payload)
            self.assertIn('payload', payload)


if __name__ == '__main__':
    unittest.main()
