"""
Run All Alaska Demos - Batch Execution
=======================================

This script runs all three Alaska demo files and saves their outputs.
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime

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

print("=" * 80)
print("RUNNING ALL ALASKA DEMOS")
print("=" * 80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Output directory: {OUTPUT_DIR}")
print()

results = []

for i, demo in enumerate(demos, 1):
    print("=" * 80)
    print(f"DEMO {i}/3: {demo['name']}")
    print("=" * 80)
    print(f"Description: {demo['description']}")
    print(f"Script: {demo['file']}")
    print()
    
    script_path = EXAMPLES_DIR / demo['file']
    output_path = OUTPUT_DIR / demo['output']
    
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        results.append({'demo': demo['name'], 'status': 'NOT FOUND', 'output': None})
        continue
    
    print(f"Running {demo['file']}...")
    start_time = datetime.now()
    
    try:
        # Run the demo and capture output
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=EXAMPLES_DIR.parent,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Save output
        with open(output_path, 'w') as f:
            f.write(f"Demo: {demo['name']}\n")
            f.write(f"Script: {demo['file']}\n")
            f.write(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Duration: {duration:.1f} seconds\n")
            f.write("=" * 80 + "\n\n")
            f.write("STDOUT:\n")
            f.write(result.stdout)
            f.write("\n\nSTDERR:\n")
            f.write(result.stderr)
            f.write("\n\nExit Code: " + str(result.returncode))
        
        if result.returncode == 0:
            print(f"✅ SUCCESS - Completed in {duration:.1f}s")
            print(f"   Output saved to: {output_path.name}")
            results.append({
                'demo': demo['name'],
                'status': 'SUCCESS',
                'duration': duration,
                'output': output_path
            })
        else:
            print(f"❌ FAILED - Exit code {result.returncode}")
            print(f"   Check output file for details: {output_path.name}")
            results.append({
                'demo': demo['name'],
                'status': 'FAILED',
                'exit_code': result.returncode,
                'output': output_path
            })
            
    except subprocess.TimeoutExpired:
        print(f"⏱️  TIMEOUT - Demo took longer than 5 minutes")
        results.append({
            'demo': demo['name'],
            'status': 'TIMEOUT',
            'output': None
        })
        
    except Exception as e:
        print(f"❌ ERROR - {str(e)}")
        results.append({
            'demo': demo['name'],
            'status': 'ERROR',
            'error': str(e),
            'output': None
        })
    
    print()

# Summary
print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
failed_count = sum(1 for r in results if r['status'] in ['FAILED', 'ERROR', 'TIMEOUT'])

print(f"Total Demos: {len(demos)}")
print(f"Successful: {success_count}")
print(f"Failed: {failed_count}")
print()

print("Results:")
for r in results:
    status_icon = "✅" if r['status'] == 'SUCCESS' else "❌"
    print(f"  {status_icon} {r['demo']}: {r['status']}", end='')
    if 'duration' in r:
        print(f" ({r['duration']:.1f}s)")
    else:
        print()

print()
print(f"All outputs saved to: {OUTPUT_DIR}")
print()

# Create summary file
summary_path = OUTPUT_DIR / 'SUMMARY.txt'
with open(summary_path, 'w') as f:
    f.write("Alaska Demos Execution Summary\n")
    f.write("=" * 80 + "\n")
    f.write(f"Executed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    for r in results:
        f.write(f"Demo: {r['demo']}\n")
        f.write(f"Status: {r['status']}\n")
        if 'duration' in r:
            f.write(f"Duration: {r['duration']:.1f}s\n")
        if r['output']:
            f.write(f"Output: {r['output'].name}\n")
        f.write("\n")
    
    f.write(f"\nSuccess Rate: {success_count}/{len(demos)} ({success_count/len(demos)*100:.0f}%)\n")

print(f"Summary saved to: {summary_path.name}")
print()
print("=" * 80)
print("DEMO EXECUTION COMPLETE")
print("=" * 80)
