"""Run the wavy-channel scientific workflow."""
import argparse,subprocess,sys
from pathlib import Path
STAGES={
 'tests':['test_revision.py'],
 'experiments':['revision_experiments.py'],
 'controls':['revision_controls.py'],
 'intervals':['revision_interval_groups.py','revision_random_reference.py'],
 'diagnostics':['revision_diagnostics.py','reconsider_diagnostics.py'],
 'exports':['wavy_channel_exports.py'],
 'validate':['validate_revision_outputs.py']}
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['all']+list(STAGES),default='all');a=p.parse_args()
 for stage in (STAGES if a.stage=='all' else [a.stage]):
  for filename in STAGES[stage]:subprocess.run([sys.executable,str(Path(__file__).parent/filename)],check=True)
if __name__=='__main__':main()
