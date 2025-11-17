#!/usr/bin/env python3
"""
Generic Batch Processing Orchestrator
Auto-discovers films and queues processing jobs with dependency management.

Usage:
    python batch_processor.py --config configs/batch/sam2_lama.yaml --resume

Features:
    - Auto-discovery of films under base directory
    - Dependency-aware job scheduling (SAM2 → LaMa → Clustering, etc.)
    - Completion detection (skip already-processed jobs)
    - Retry logic with exponential backoff
    - Progress tracking and resume capability
    - Detailed logging per job
"""

import argparse
import json
import yaml
import subprocess
import time
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from glob import glob

@dataclass
class Job:
    """Represents a single processing job for one film"""
    name: str
    film: str
    script: str
    conda_env: str
    args_template: List[str]
    args_named: Dict[str, Any]
    completion_check: Dict[str, Any]
    resources: Dict[str, Any]
    retry: Dict[str, Any]
    depends_on: Optional[str] = None
    status: str = "pending"  # pending, running, completed, failed, skipped
    attempts: int = 0
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_seconds: Optional[float] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dict for JSON serialization"""
        return asdict(self)


class BatchProcessor:
    """Orchestrates multi-stage batch processing across multiple films"""

    def __init__(self, config_path: str, resume: bool = True, dry_run: bool = False):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.dry_run = dry_run
        self.jobs: List[Job] = []

        # Create log directory
        self.log_dir = Path(self.config['execution']['log_dir'])
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.progress_file = self.log_dir / self.config['execution']['progress_file']

        if resume and self.progress_file.exists():
            print(f"📂 Loading progress from {self.progress_file}")
            self._load_progress()
        else:
            print("🔍 Discovering films and creating job queue...")
            self._discover_and_queue_jobs()

        self._print_summary()

    def _load_config(self) -> Dict:
        """Load and validate YAML configuration"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")

        with open(self.config_path) as f:
            config = yaml.safe_load(f)

        # Validate required keys
        required_keys = ['discovery', 'jobs', 'execution']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")

        return config

    def _discover_films(self) -> List[str]:
        """Auto-discover films in base directory"""
        base_dir = Path(self.config['discovery']['base_dir'])

        if not base_dir.exists():
            raise FileNotFoundError(f"Base directory not found: {base_dir}")

        pattern = self.config['discovery'].get('film_pattern', '*')
        exclude = set(self.config['discovery'].get('exclude', []))

        # Find all subdirectories matching pattern
        films = []
        for path in base_dir.glob(pattern):
            if path.is_dir() and path.name not in exclude:
                # Check if frames exist (either frames/ or frames_final/)
                frames_dir = path / 'frames'
                frames_final_dir = path / 'frames_final'

                if frames_dir.exists() or frames_final_dir.exists():
                    films.append(path.name)

        return sorted(films)

    def _discover_and_queue_jobs(self):
        """Discover films and create job queue"""
        films = self._discover_films()

        if not films:
            print("⚠️  No films found matching criteria!")
            return

        print(f"✅ Found {len(films)} films: {', '.join(films)}")

        # Create jobs for each film × job definition
        job_defs = self.config['jobs']

        for film in films:
            for job_def in job_defs:
                job = Job(
                    name=job_def['name'],
                    film=film,
                    script=job_def['script'],
                    conda_env=job_def.get('conda_env', 'ai_env'),
                    args_template=job_def['args'].get('template', []),
                    args_named=job_def['args'].get('named', {}),
                    completion_check=job_def['completion_check'],
                    resources=job_def.get('resources', {}),
                    retry=job_def.get('retry', {'max_attempts': 1, 'backoff_seconds': 60}),
                    depends_on=job_def.get('depends_on')
                )
                self.jobs.append(job)

        print(f"📋 Created {len(self.jobs)} jobs ({len(films)} films × {len(job_defs)} operations)")

    def _resolve_template(self, template: str, film: str) -> str:
        """Resolve template variables like {film}, {base_dir}"""
        base_dir = self.config['discovery']['base_dir']

        # Determine frames directory (frames/ or frames_final/)
        film_dir = Path(base_dir) / film
        if (film_dir / 'frames').exists():
            frames_dir = str(film_dir / 'frames')
        elif (film_dir / 'frames_final').exists():
            frames_dir = str(film_dir / 'frames_final')
        else:
            frames_dir = str(film_dir / 'frames')  # Default

        replacements = {
            '{film}': film,
            '{base_dir}': base_dir,
            '{frames_dir}': frames_dir
        }

        result = template
        for key, value in replacements.items():
            result = result.replace(key, value)

        return result

    def _check_completion(self, job: Job) -> bool:
        """Check if job has completed successfully"""
        check = job.completion_check
        check_type = check['type']

        try:
            if check_type == "backgrounds_complete":
                # Count frames in source directory
                frames_dir = Path(self._resolve_template(check['frames_dir'], job.film))
                if not frames_dir.exists():
                    print(f"⚠️  Frames directory not found: {frames_dir}")
                    return False

                total_frames = len(list(frames_dir.glob("*.jpg")))
                if total_frames == 0:
                    print(f"⚠️  No frames found in {frames_dir}")
                    return False

                # Count backgrounds produced
                backgrounds_dir = Path(self._resolve_template(check['path'], job.film))
                if not backgrounds_dir.exists():
                    return False

                backgrounds_count = len(list(backgrounds_dir.glob("*.jpg")))

                # Check if at least tolerance% of frames processed
                tolerance = check.get('tolerance', 0.95)
                required_count = int(total_frames * tolerance)

                is_complete = backgrounds_count >= required_count

                if is_complete:
                    print(f"✅ {job.name}/{job.film}: {backgrounds_count}/{total_frames} backgrounds ({backgrounds_count*100/total_frames:.1f}%)")
                else:
                    print(f"⏳ {job.name}/{job.film}: {backgrounds_count}/{total_frames} backgrounds ({backgrounds_count*100/total_frames:.1f}%, need {required_count})")

                return is_complete

            elif check_type == "directory_exists":
                path = Path(self._resolve_template(check['path'], job.film))
                if not path.exists():
                    return False

                min_files = check.get('min_files', 0)
                if min_files > 0:
                    file_count = len(list(path.iterdir()))
                    return file_count >= min_files
                return True

            elif check_type == "file_exists":
                path = Path(self._resolve_template(check['path'], job.film))
                return path.exists()

            elif check_type == "metadata_key":
                path = Path(self._resolve_template(check['path'], job.film))
                if not path.exists():
                    return False

                with open(path) as f:
                    data = json.load(f)

                key = check['key']
                return key in data and data[key] > 0

            else:
                print(f"⚠️  Unknown completion check type: {check_type}")
                return False

        except Exception as e:
            print(f"⚠️  Error checking completion for {job.name}/{job.film}: {e}")
            return False

    def _build_command(self, job: Job) -> List[str]:
        """Build command line for executing job"""
        cmd = ['conda', 'run', '-n', job.conda_env, 'python', job.script]

        # Add positional arguments (template args)
        if isinstance(job.args_template, str):
            resolved = self._resolve_template(job.args_template, job.film)
            cmd.append(resolved)
        elif isinstance(job.args_template, list):
            for arg in job.args_template:
                resolved = self._resolve_template(arg, job.film)
                cmd.append(resolved)

        # Add named arguments
        for key, value in job.args_named.items():
            cmd.append(key)
            if value is not None:
                resolved = self._resolve_template(str(value), job.film)
                cmd.append(resolved)

        return cmd

    def _execute_job(self, job: Job) -> bool:
        """Execute a single job, returns True on success"""
        cmd = self._build_command(job)

        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f"{job.name}_{job.film}_{timestamp}.log"

        print(f"\n{'='*70}")
        print(f"🚀 Starting: {job.name} for {job.film}")
        print(f"   Command: {' '.join(cmd)}")
        print(f"   Log: {log_file}")
        print(f"{'='*70}\n")

        if self.dry_run:
            print("   [DRY RUN] Would execute command")
            return True

        job.status = "running"
        job.start_time = datetime.now().isoformat()
        job.attempts += 1

        start = time.time()

        try:
            with open(log_file, 'w') as log:
                result = subprocess.run(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    text=True
                )

            end = time.time()
            job.end_time = datetime.now().isoformat()
            job.duration_seconds = end - start

            # Check both exit code AND completion check
            exit_success = result.returncode == 0

            # Even if exit code is 0, verify actual completion
            completion_verified = self._check_completion(job)

            if exit_success and completion_verified:
                job.status = "completed"
                print(f"✅ Completed: {job.name} for {job.film} ({job.duration_seconds:.1f}s)")
                success = True
            elif exit_success and not completion_verified:
                job.status = "failed"
                job.error_message = "Process exited successfully but completion check failed"
                print(f"⚠️  Warning: {job.name} for {job.film} exited OK but output incomplete")
                success = False
            else:
                job.status = "failed"
                job.error_message = f"Exit code {result.returncode}"
                print(f"❌ Failed: {job.name} for {job.film} (exit code {result.returncode})")
                success = False

            self._save_progress()
            return success

        except Exception as e:
            job.status = "failed"
            job.end_time = datetime.now().isoformat()
            job.error_message = str(e)
            print(f"❌ Exception: {job.name} for {job.film}: {e}")
            self._save_progress()
            return False

    def _dependency_met(self, job: Job) -> bool:
        """Check if job dependencies are satisfied"""
        if not job.depends_on:
            return True

        # Find dependency job for same film
        dep_jobs = [
            j for j in self.jobs
            if j.name == job.depends_on and j.film == job.film
        ]

        if not dep_jobs:
            print(f"⚠️  Dependency not found: {job.depends_on} for {job.film}")
            return False

        dep_job = dep_jobs[0]
        return dep_job.status == "completed"

    def _save_progress(self):
        """Save current job status to progress file"""
        progress = {
            "timestamp": datetime.now().isoformat(),
            "config_file": str(self.config_path),
            "jobs": [job.to_dict() for job in self.jobs]
        }

        with open(self.progress_file, 'w') as f:
            json.dump(progress, f, indent=2)

    def _load_progress(self):
        """Load job status from progress file"""
        with open(self.progress_file) as f:
            progress = json.load(f)

        # Reconstruct jobs from saved state
        for job_data in progress['jobs']:
            job = Job(**{k: v for k, v in job_data.items() if k in Job.__annotations__})
            self.jobs.append(job)

        print(f"   Loaded {len(self.jobs)} jobs from progress file")

    def _print_summary(self):
        """Print current job status summary"""
        if not self.jobs:
            return

        total = len(self.jobs)
        completed = sum(1 for j in self.jobs if j.status == "completed")
        failed = sum(1 for j in self.jobs if j.status == "failed")
        pending = sum(1 for j in self.jobs if j.status == "pending")
        skipped = sum(1 for j in self.jobs if j.status == "skipped")

        print(f"\n{'='*70}")
        print(f"📊 Job Queue Summary")
        print(f"{'='*70}")
        print(f"   Total jobs:     {total}")
        print(f"   ✅ Completed:    {completed}")
        print(f"   ⏸️  Pending:      {pending}")
        print(f"   ❌ Failed:       {failed}")
        print(f"   ⏭️  Skipped:      {skipped}")
        print(f"{'='*70}\n")

    def run(self):
        """Execute all jobs with dependency resolution and retry logic"""
        if not self.jobs:
            print("❌ No jobs to execute!")
            return

        print(f"🎬 Starting batch processing...")
        if self.dry_run:
            print("   [DRY RUN MODE - No actual execution]")
        print()

        iteration = 0
        max_iterations = 1000  # Prevent infinite loops

        while iteration < max_iterations:
            iteration += 1

            # Find next runnable job
            runnable_jobs = [
                j for j in self.jobs
                if j.status in ["pending", "failed"]
                and self._dependency_met(j)
                and j.attempts < j.retry['max_attempts']
            ]

            if not runnable_jobs:
                break  # All done or stuck

            # Process ONE job at a time (sequential mode for GPU constraint)
            if runnable_jobs:
                job = runnable_jobs[0]  # Take first runnable job only

                # Check if already completed (for resume/skip)
                if self._check_completion(job):
                    job.status = "skipped"
                    print(f"⏭️  Skipping {job.name} for {job.film} (already completed)")
                    self._save_progress()
                    continue

                # Execute
                success = self._execute_job(job)

                # Retry logic
                if not success and job.attempts < job.retry['max_attempts']:
                    backoff = job.retry['backoff_seconds']
                    print(f"⏳ Retry {job.attempts}/{job.retry['max_attempts']} in {backoff}s...")
                    if not self.dry_run:
                        time.sleep(backoff)
                else:
                    # Job completed or exhausted retries, move to next
                    continue

        # Final summary
        self._print_summary()

        # Check for failures
        failed_jobs = [j for j in self.jobs if j.status == "failed"]
        if failed_jobs:
            print(f"\n⚠️  {len(failed_jobs)} jobs failed:")
            for job in failed_jobs:
                print(f"   - {job.name} for {job.film}: {job.error_message}")
            print(f"\n💡 Check logs in {self.log_dir} for details")

        completed = sum(1 for j in self.jobs if j.status in ["completed", "skipped"])
        total = len(self.jobs)

        if completed == total:
            print(f"\n🎉 All {total} jobs completed successfully!")
        else:
            print(f"\n⚠️  {completed}/{total} jobs completed")


def main():
    parser = argparse.ArgumentParser(
        description="Generic Batch Processing Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run SAM2 + LaMa batch processing with resume
  python batch_processor.py --config configs/batch/sam2_lama.yaml --resume

  # Dry run to test configuration
  python batch_processor.py --config configs/batch/sam2_lama.yaml --dry-run

  # Fresh start (ignore progress file)
  python batch_processor.py --config configs/batch/sam2_lama.yaml --no-resume
        """
    )

    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )

    parser.add_argument(
        '--resume',
        action='store_true',
        default=True,
        help='Resume from previous progress (default: True)'
    )

    parser.add_argument(
        '--no-resume',
        action='store_false',
        dest='resume',
        help='Start fresh, ignore progress file'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Test configuration without executing jobs'
    )

    args = parser.parse_args()

    try:
        processor = BatchProcessor(
            config_path=args.config,
            resume=args.resume,
            dry_run=args.dry_run
        )
        processor.run()

    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
