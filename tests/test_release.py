"""Check that deposited displays preserve source results and release boundaries."""
import csv,hashlib,importlib.util,json,re,subprocess,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def load(name):return json.loads((ROOT/name).read_text(encoding='utf-8-sig'))
def rows(name):
    with (ROOT/name).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

class ReleaseTests(unittest.TestCase):
    def test_r4_table_display_order(self):
        tables=load('data/presentation/tables.json')['supplementary_tables']
        self.assertEqual([x['number'] for x in tables],['1','2A','2B','2C','2D','3','4','5','6'])
        self.assertTrue(tables[0]['title'].startswith('Supplementary clinical'))
        self.assertEqual([x['title'] for x in tables[1:5]],[
            'Complete calibration results: eICU, 50 labels',
            'Complete calibration results: eICU, 500 labels',
            'Complete calibration results: NWICU, 50 labels',
            'Complete calibration results: NWICU, 500 labels',
        ])
        self.assertRegex(tables[5]['legend'], r'\bData\s*22\b')
        self.assertTrue(tables[6]['title'].startswith('Calendar-separated'))
        self.assertTrue(tables[7]['title'].startswith('Exploratory SICdb'))
        self.assertTrue(tables[8]['title'].startswith('Point-only log loss'))

    def test_figure_display_renumbering_preserves_source_identity(self):
        index=load('data/presentation/figure_display_index.json')
        expected_main={'Figure1':'Figure2','Figure2':'Figure6','Figure3':'Figure1','Figure4':'Figure3','Figure5':'Figure4','Figure6':'Figure5'}
        expected_supp={'FigureS1':'FigureS3','FigureS2':'FigureS5','FigureS3':'FigureS4','FigureS4':'FigureS2','FigureS5':'FigureS6','FigureS6':'FigureS1'}
        self.assertEqual(index['display_renumbering']['main'],expected_main)
        self.assertEqual(index['display_renumbering']['supplementary'],expected_supp)
        history={'Figure1':'1','Figure2':'2','Figure3':'6','Figure4':None,'Figure5':None,'Figure6':None,
                 'FigureS1':'S1','FigureS2':'S2','FigureS3':'S3','FigureS4':'S4','FigureS5':'3','FigureS6':'4'}
        previous={x['previous_display_figure']:x for x in index['figures']}
        bindings={}
        for record in load('provenance/figure-source-bindings.json'):
            if 'previous_display_figure' in record:
                bindings[record['previous_display_figure']]=record['display_figure']
                self.assertEqual(record['release_version'],'2026.11.25-r4')
        for old,new in {**expected_main,**expected_supp}.items():
            record=previous[old]
            self.assertEqual(record['figure'],new)
            self.assertEqual(record['historical_source_figure'],history[old])
            self.assertEqual(hashlib.sha256((ROOT/record['reference_pdf']).read_bytes()).hexdigest(),record['reference_pdf_sha256'])
            self.assertEqual(bindings[old],new)
        self.assertEqual(hashlib.sha256((ROOT/'data/presentation/figures.json').read_bytes()).hexdigest(),'01f9a298cd05229a2196e1a6dbc90c3e901744edbf47553603c2c4c4fa4103c0')
        self.assertEqual(hashlib.sha256((ROOT/'data/presentation/tables_source_r2.json').read_bytes()).hexdigest(),'7c2251df2346e94b02255c075f95b611a130ff42830438a0e021211d78c94e80')

    def test_release_integrity(self):
        subprocess.run([sys.executable,str(ROOT/'scripts/reproduce.py'),'--verify-only'],check=True)

    def test_main_calibration_summary_matches_raw_and_platt_source(self):
        data=rows('data/aggregates/Data15/calibration/global_50_500_summary.csv')
        table=load('data/presentation/tables.json')['main_tables'][1]
        self.assertEqual(len(table['rows']),8)
        self.assertEqual(len(table['columns']),7)
        cohorts={'eICU':'eicu_semantic_v4_2','NWICU':'nwicu_discharge_v4_1'}
        for row in table['rows']:
            cohort,budget,model,events,failures_fallback,brier,ll=row
            raw=[x for x in data if x['cohort']==cohorts[cohort] and x['model']==model and x['method']=='no_update' and x['budget']==budget]
            platt=[x for x in data if x['cohort']==cohorts[cohort] and x['model']==model and x['method']=='platt' and x['budget']==budget]
            self.assertEqual((len(raw),len(platt)),(1,1));raw=raw[0];platt=platt[0]
            self.assertEqual(events,f"{float(platt['events_p05']):.1f}–{float(platt['events_p95']):.1f}")
            self.assertEqual(failures_fallback,f"{int(float(platt['fit_failures']))}/{int(float(platt['fallback_count']))}")
            for cell,prefix in [(brier,'brier'),(ll,'logloss')]:
                observed=[float(x) for x in re.findall(r'\d+\.\d+',cell)]
                expected=[float(f"{float(src[prefix+'_mean']):.5f}") for src in (raw,platt)]
                self.assertEqual(observed,expected)

    def test_supplementary_calibration_blocks_restore_full_24_row_grid(self):
        data=rows('data/aggregates/Data15/calibration/global_50_500_summary.csv')
        tables=load('data/presentation/tables.json')['supplementary_tables']
        by_number={table['number']:table for table in tables}
        cohorts={'eICU':'eicu_semantic_v4_2','NWICU':'nwicu_discharge_v4_1'}
        blocks={'2A':('eICU','50'),'2B':('eICU','500'),'2C':('NWICU','50'),'2D':('NWICU','500')}
        methods={'No update':'no_update','Intercept-only':'intercept','Platt':'platt'}
        seen=[]
        for number,(cohort,budget) in blocks.items():
            table=by_number[number]
            self.assertEqual(len(table['rows']),6)
            self.assertEqual(len(table['columns']),6)
            for row in table['rows']:
                model,method,attempts,events,brier,ll=row
                match=[x for x in data if x['cohort']==cohorts[cohort] and x['model']==model and x['method']==methods[method] and x['budget']==budget]
                self.assertEqual(len(match),1);src=match[0]
                seen.append((cohort,budget,model,method))
                expected_attempts='0/0/0' if method=='No update' else f"{int(float(src['fit_attempts']))}/{int(float(src['fit_failures']))}/{int(float(src['fallback_count']))}"
                self.assertEqual(attempts,expected_attempts)
                self.assertEqual(events,f"{float(src['events_p05']):.1f}–{float(src['events_p95']):.1f}")
                for cell,prefix in [(brier,'brier'),(ll,'logloss')]:
                    observed=[float(x) for x in re.findall(r'\d+\.\d+',cell)]
                    expected=[float(f"{float(src[prefix+'_'+suffix]):.5f}") for suffix in ['mean','p05','p95']]
                    self.assertEqual(observed,expected)
        self.assertEqual(len(seen),24)
        self.assertEqual(len(set(seen)),24)

    def test_new_source_only_table_matches_source(self):
        data=rows('data/aggregates/Data22/metrics.csv')
        paired=rows('data/aggregates/Data22/paired_gaps.csv')
        table=next(x for x in load('data/presentation/tables.json')['supplementary_tables'] if x['number']=='3')
        self.assertEqual(len(table['rows']),12)
        for row in table['rows']:
            cohort,seed,budget,fm,lr,bgap,lgap=row
            q=[x for x in data if x['dataset']==cohort.lower() and x['model']=='fm_seed_'+seed and x['budget']==budget]
            self.assertEqual(len(q),1);self.assertEqual(float(fm),float(f"{float(q[0]['brier']):.6f}"))
            for cell,metric in [(bgap,'brier'),(lgap,'logloss')]:
                match=[x for x in paired if x['dataset']==cohort.lower() and x['seed']==seed and x['budget']==budget and x['metric']==metric]
                self.assertEqual(len(match),1)
                got=[float(x) for x in re.findall(r'[+-]?\d+\.\d+',cell)]
                want=[float(f"{float(match[0][key]):+.7f}") for key in ['updated_gap','updated_ci_low','updated_ci_high']]
                self.assertEqual(got,want)

    def test_private_workspace_stages_code_without_clinical_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            command=[sys.executable,str(ROOT/'scripts/clinical_workspace.py')]
            subprocess.run(command+['--prepare',tmp],check=True,capture_output=True,text=True)
            result=subprocess.run(command+['--check',tmp],capture_output=True,text=True)
            self.assertEqual(result.returncode,2)
            report=json.loads(result.stdout)
            self.assertTrue(any(x['missing_inputs'] for x in report['stages']))
            self.assertTrue(all(x['script_present'] for x in report['stages']))
            self.assertFalse(any(Path(tmp).rglob('*.npz')))

    def test_release_excludes_individual_products_and_credentials(self):
        subprocess.run([sys.executable,str(ROOT/'scripts/check_release.py')],check=True,capture_output=True,text=True)

    def test_tables_only_manifest_ignores_stale_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp);(path/'figures').mkdir();(path/'figures/stale.txt').write_text('test fixture')
            subprocess.run([sys.executable,str(ROOT/'scripts/reproduce.py'),'--tables-only','--output',tmp],check=True,capture_output=True,text=True)
            result=json.loads((path/'reproduction_manifest.json').read_text())
            self.assertEqual(result['figures'],0);self.assertEqual(len(result['outputs']),22)
            self.assertTrue(all(x['path'].startswith('tables/') for x in result['outputs']))

    def test_metadata_is_editable_without_unbinding_scientific_inputs(self):
        entries=load('provenance/file_manifest.json')['files']
        self.assertNotIn('CITATION.cff',{x['path'] for x in entries})

    def test_workspace_guard_rejects_production_subdirectory(self):
        production=load('config/clinical_stages.json')['original_root']
        result=subprocess.run([sys.executable,str(ROOT/'scripts/clinical_workspace.py'),'--prepare',production+'/release_test_must_not_be_created'],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertFalse(Path(production+'/release_test_must_not_be_created').exists())

if __name__=='__main__':unittest.main()
