"""
Run All Alaska Demos - Batch Execution
=======================================

This script runs all three Alaska demo files and saves their outputs.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path('/Users/k.jones/Desktop/geostats/alaska_outputs/demos')
OUTPUT_DIR.mkdir(exist_ok=True)

EXAMPLES_DIR = Path('/Users/k.jones/Desktop/geostats/examples')

demos = [
 {
 'name': 'Demo 1: Gold Exploration',
 'file': 'demo_01_gold_exploration.py',
 'output': 'demo_01_output.txt',
 'description': 'Gold exploration workflow - Fairbanks district'
 },
 {
 'name': 'Demo 2: Multi-Element Cokriging',
 'file': 'demo_02_multi_element_cokriging.py',
 'output': 'demo_02_output.txt',
 'description': 'Cu-Mo-Au multi-element analysis with cokriging'
 },
 {
 'name': 'Demo 3: Environmental Assessment',
 'file': 'demo_03_environmental_assessment.py',
 'output': 'demo_03_output.txt',
 'description': 'Environmental risk assessment - As, Pb, Hg'
 }
]

logger.info("RUNNING ALL ALASKA DEMOS")
logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info(f"Output directory: {OUTPUT_DIR}")

results = []

for i, demo in enumerate(demos, 1):
 logger.info(f"Script: {demo['file']}")

 script_path = EXAMPLES_DIR / demo['file']
 output_path = OUTPUT_DIR / demo['output']

 if not script_path.exists():
 results.append({'demo': demo['name'], 'status': 'NOT FOUND', 'output': None})
 continue

 logger.info(f"Running {demo['file']}...")
 start_time = datetime.now()

 try:
 try:
 result = subprocess.run(
 [sys.executable, str(script_path)],
 cwd=EXAMPLES_DIR.parent,
 capture_output=True,
 text=True,
 timeout=300 # 5 minute timeout
 )

 end_time = datetime.now()
 duration = (end_time - start_time).total_seconds()

 # Save output
 with open(output_path, 'w') as f:
 with open(output_path, 'w') as f:
 f.write(f"Script: {demo['file']}\n")
 f.write(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
 f.write(f"Duration: {duration:.1f} seconds\n")
 f.write("\n")
 f.write("STDOUT:\n")
 f.write(result.stdout)
 f.write("\n\nSTDERR:\n")
 f.write(result.stderr)
 f.write("\n\nExit Code: " + str(result.returncode))

 if result.returncode == 0:
 logger.info(f" Output saved to: {output_path.name}")
 results.append({
 'demo': demo['name'],
 'status': 'SUCCESS',
 'duration': duration,
 'output': output_path
 })
 else:
 else:
 logger.info(f" Check output file for details: {output_path.name}")
 results.append({
 'demo': demo['name'],
 'status': 'FAILED',
 'exit_code': result.returncode,
 'output': output_path
 })

 except subprocess.TimeoutExpired:
 logger.info(f"⏱ TIMEOUT - Demo took longer than 5 minutes")
 results.append({
 'demo': demo['name'],
 'status': 'TIMEOUT',
 'output': None
 })

 except Exception as e:
 logger.error(f" ERROR - {str(e)}")
 results.append({
 'demo': demo['name'],
 'status': 'ERROR',
 'error': str(e),
 'output': None
 })

# Summary
logger.info("SUMMARY")

success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
failed_count = sum(1 for r in results if r['status'] in ['FAILED', 'ERROR', 'TIMEOUT'])

logger.info(f"Total Demos: {len(demos)}")
logger.info(f"Successful: {success_count}")
logger.error(f"Failed: {failed_count}")

logger.info("Results:")
for r in results:
 if 'duration' in r:
 else:
    pass

 else:
    pass

# Create summary file
summary_path = OUTPUT_DIR / 'SUMMARY.txt'
with open(summary_path, 'w') as f:
with open(summary_path, 'w') as f:

 for r in results:
 f.write(f"Status: {r['status']}\n")
 if 'duration' in r:
 if r['output']:
 f.write("\n")

 f.write(f"\nSuccess Rate: {success_count}/{len(demos)} ({success_count/len(demos)*100:.0f}%)\n")

logger.info(f"Summary saved to: {summary_path.name}")
logger.info("DEMO EXECUTION COMPLETE")
