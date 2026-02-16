#!/usr/bin/env python3
"""
Belle II DIRAC Grid Job Monitor
================================

Real-time monitoring and management of Belle II grid jobs submitted via gbasf2.

Features:
- Monitor job status across all grid sites
- Automatic job resubmission on failure
- Progress tracking and ETA estimation
- Email/Slack notifications
- Resource utilization statistics
- Log file parsing and error detection

Usage:
    python grid_job_monitor.py --job-id 12345678 --check-interval 300

Author: [Your Name]
Date: February 2026
"""

import argparse
import subprocess
import time
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)


@dataclass
class JobStatus:
    """Container for job status information."""
    job_id: str
    timestamp: str
    total_jobs: int
    done: int
    running: int
    waiting: int
    failed: int
    killed: int
    stalled: int
    completion_rate: float
    estimated_time_remaining: Optional[str] = None
    sites_active: List[str] = None


class GridJobMonitor:
    """
    Monitor Belle II DIRAC grid jobs.
    
    Polls gbasf2 job status and provides real-time updates.
    """
    
    def __init__(self, job_id: str, check_interval: int = 300,
                 auto_resubmit: bool = False, max_retries: int = 3):
        """
        Args:
            job_id: DIRAC job ID to monitor
            check_interval: Status check interval in seconds
            auto_resubmit: Automatically resubmit failed jobs
            max_retries: Maximum number of resubmission attempts
        """
        self.job_id = job_id
        self.check_interval = check_interval
        self.auto_resubmit = auto_resubmit
        self.max_retries = max_retries
        
        self.status_history: List[JobStatus] = []
        self.retry_count: Dict[str, int] = defaultdict(int)
        self.start_time = datetime.now()
        
        # Load configuration
        self.config = self._load_config()
        
        logger.info(f"Initialized monitor for job {job_id}")
        logger.info(f"Check interval: {check_interval}s, Auto-resubmit: {auto_resubmit}")
    
    def _load_config(self) -> Dict:
        """Load configuration from file or defaults."""
        config_file = Path.home() / '.belle2' / 'monitor_config.json'
        
        default_config = {
            'email_notifications': False,
            'email_address': '',
            'slack_webhook': '',
            'alert_on_failure_rate': 0.1,  # Alert if >10% failures
            'log_dir': str(Path.cwd() / 'logs'),
        }
        
        if config_file.exists():
            with open(config_file) as f:
                config = json.load(f)
            logger.info(f"Loaded config from {config_file}")
        else:
            config = default_config
            logger.info("Using default configuration")
        
        # Create log directory
        Path(config['log_dir']).mkdir(parents=True, exist_ok=True)
        
        return config
    
    def get_job_status(self) -> Optional[JobStatus]:
        """
        Query current job status from DIRAC.
        
        Returns:
            JobStatus object or None on error
        """
        try:
            # Execute gb2_job_status command
            cmd = ['gb2_job_status', str(self.job_id)]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                logger.error(f"gb2_job_status failed: {result.stderr}")
                return None
            
            # Parse output
            status = self._parse_job_status(result.stdout)
            
            # Store in history
            self.status_history.append(status)
            
            return status
        
        except subprocess.TimeoutExpired:
            logger.error("Job status query timed out")
            return None
        except Exception as e:
            logger.error(f"Error getting job status: {e}")
            return None
    
    def _parse_job_status(self, output: str) -> JobStatus:
        """
        Parse gb2_job_status output.
        
        Expected format:
        JobID: 12345678
        Status counts:
          Done: 850
          Running: 120
          Waiting: 20
          Failed: 10
        """
        status_counts = {
            'Done': 0,
            'Running': 0,
            'Waiting': 0,
            'Failed': 0,
            'Killed': 0,
            'Stalled': 0,
        }
        
        lines = output.split('\n')
        
        for line in lines:
            for state in status_counts.keys():
                if state in line:
                    try:
                        count = int(line.split()[-1])
                        status_counts[state] = count
                    except (ValueError, IndexError):
                        pass
        
        total = sum(status_counts.values())
        
        if total == 0:
            total = 1  # Avoid division by zero
        
        completion_rate = status_counts['Done'] / total
        
        # Estimate time remaining
        eta = self._estimate_eta(status_counts, completion_rate)
        
        # Extract active sites (simplified)
        sites_active = self._extract_active_sites(output)
        
        return JobStatus(
            job_id=self.job_id,
            timestamp=datetime.now().isoformat(),
            total_jobs=total,
            done=status_counts['Done'],
            running=status_counts['Running'],
            waiting=status_counts['Waiting'],
            failed=status_counts['Failed'],
            killed=status_counts['Killed'],
            stalled=status_counts['Stalled'],
            completion_rate=completion_rate,
            estimated_time_remaining=eta,
            sites_active=sites_active
        )
    
    def _estimate_eta(self, status_counts: Dict, completion_rate: float) -> Optional[str]:
        """
        Estimate time remaining based on completion rate.
        
        Returns:
            Human-readable ETA string
        """
        if len(self.status_history) < 2:
            return None
        
        # Compute completion velocity (jobs per second)
        recent_history = self.status_history[-5:]  # Last 5 checks
        
        if len(recent_history) < 2:
            return None
        
        time_elapsed = (
            datetime.fromisoformat(recent_history[-1].timestamp) -
            datetime.fromisoformat(recent_history[0].timestamp)
        ).total_seconds()
        
        jobs_completed = recent_history[-1].done - recent_history[0].done
        
        if time_elapsed <= 0 or jobs_completed <= 0:
            return None
        
        velocity = jobs_completed / time_elapsed  # jobs/sec
        
        # Jobs remaining
        remaining = status_counts['Waiting'] + status_counts['Running']
        
        if velocity > 0:
            eta_seconds = remaining / velocity
            eta_timedelta = timedelta(seconds=int(eta_seconds))
            
            return str(eta_timedelta)
        
        return None
    
    def _extract_active_sites(self, output: str) -> List[str]:
        """Extract list of active grid sites from status output."""
        sites = []
        
        # Simplified: look for common site patterns
        site_patterns = ['LCG.', 'CLOUD.', 'T2_', 'T3_']
        
        for line in output.split('\n'):
            for pattern in site_patterns:
                if pattern in line:
                    # Extract site name (heuristic)
                    parts = line.split()
                    for part in parts:
                        if pattern in part:
                            sites.append(part)
                            break
        
        return list(set(sites))  # Unique sites
    
    def monitor_loop(self, max_iterations: Optional[int] = None):
        """
        Main monitoring loop.
        
        Args:
            max_iterations: Maximum iterations (None = infinite)
        """
        iteration = 0
        
        logger.info("="*70)
        logger.info(f"Starting monitoring for job {self.job_id}")
        logger.info("="*70)
        
        try:
            while True:
                iteration += 1
                
                # Get status
                status = self.get_job_status()
                
                if status is None:
                    logger.warning("Failed to get job status, retrying...")
                    time.sleep(30)
                    continue
                
                # Display status
                self._display_status(status)
                
                # Check for alerts
                self._check_alerts(status)
                
                # Auto-resubmit failed jobs
                if self.auto_resubmit and status.failed > 0:
                    self._handle_failed_jobs(status)
                
                # Check if complete
                if self._is_complete(status):
                    logger.info("="*70)
                    logger.info("All jobs completed!")
                    logger.info("="*70)
                    self._generate_summary()
                    break
                
                # Check iteration limit
                if max_iterations and iteration >= max_iterations:
                    logger.info(f"Reached maximum iterations ({max_iterations})")
                    break
                
                # Wait for next check
                logger.info(f"Next check in {self.check_interval}s...")
                time.sleep(self.check_interval)
        
        except KeyboardInterrupt:
            logger.info("\nMonitoring interrupted by user")
            self._generate_summary()
        except Exception as e:
            logger.error(f"Monitoring error: {e}", exc_info=True)
    
    def _display_status(self, status: JobStatus):
        """Display formatted status update."""
        logger.info("\n" + "="*70)
        logger.info(f"Job Status Update - {status.timestamp}")
        logger.info("="*70)
        logger.info(f"Job ID: {status.job_id}")
        logger.info(f"Total Jobs: {status.total_jobs}")
        logger.info(f"  ✓ Done:     {status.done:6d} ({status.done/status.total_jobs*100:5.1f}%)")
        logger.info(f"  ▶ Running:  {status.running:6d} ({status.running/status.total_jobs*100:5.1f}%)")
        logger.info(f"  ⏸ Waiting:  {status.waiting:6d} ({status.waiting/status.total_jobs*100:5.1f}%)")
        logger.info(f"  ✗ Failed:   {status.failed:6d} ({status.failed/status.total_jobs*100:5.1f}%)")
        logger.info(f"  ⊗ Killed:   {status.killed:6d}")
        logger.info(f"  ⚠ Stalled:  {status.stalled:6d}")
        
        if status.estimated_time_remaining:
            logger.info(f"\nEstimated time remaining: {status.estimated_time_remaining}")
        
        if status.sites_active:
            logger.info(f"\nActive sites ({len(status.sites_active)}): {', '.join(status.sites_active[:5])}")
        
        # Progress bar
        self._display_progress_bar(status.completion_rate)
    
    def _display_progress_bar(self, completion_rate: float, width: int = 50):
        """Display ASCII progress bar."""
        filled = int(width * completion_rate)
        bar = '█' * filled + '░' * (width - filled)
        
        logger.info(f"\nProgress: |{bar}| {completion_rate*100:.1f}%")
    
    def _check_alerts(self, status: JobStatus):
        """Check for alert conditions and send notifications."""
        failure_rate = status.failed / status.total_jobs
        
        if failure_rate > self.config['alert_on_failure_rate']:
            message = (
                f"⚠️ High failure rate detected!\n"
                f"Job {status.job_id}: {status.failed}/{status.total_jobs} failed "
                f"({failure_rate*100:.1f}%)"
            )
            
            logger.warning(message)
            self._send_notification(message)
    
    def _send_notification(self, message: str):
        """Send notification via email or Slack."""
        # Email notification
        if self.config['email_notifications'] and self.config['email_address']:
            try:
                self._send_email(message)
            except Exception as e:
                logger.error(f"Failed to send email: {e}")
        
        # Slack notification
        if REQUESTS_AVAILABLE and self.config.get('slack_webhook'):
            try:
                self._send_slack(message)
            except Exception as e:
                logger.error(f"Failed to send Slack notification: {e}")
    
    def _send_email(self, message: str):
        """Send email notification."""
        import smtplib
        from email.mime.text import MIMEText
        
        msg = MIMEText(message)
        msg['Subject'] = f'Belle II Grid Job Alert - {self.job_id}'
        msg['From'] = 'grid-monitor@belle2.org'
        msg['To'] = self.config['email_address']
        
        # Note: Requires SMTP configuration
        # This is a placeholder - configure with your SMTP server
        logger.info(f"Would send email to {self.config['email_address']}")
    
    def _send_slack(self, message: str):
        """Send Slack notification."""
        webhook_url = self.config['slack_webhook']
        
        payload = {
            'text': message,
            'username': 'Belle II Grid Monitor',
            'icon_emoji': ':bell:'
        }
        
        response = requests.post(webhook_url, json=payload)
        
        if response.status_code == 200:
            logger.info("Slack notification sent")
        else:
            logger.error(f"Slack notification failed: {response.status_code}")
    
    def _handle_failed_jobs(self, status: JobStatus):
        """Attempt to resubmit failed jobs."""
        if status.failed == 0:
            return
        
        logger.info(f"\nAttempting to resubmit {status.failed} failed jobs...")
        
        try:
            # Get list of failed job IDs
            cmd = ['gb2_job_status', str(self.job_id), '--failed']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            # Parse failed job IDs (simplified)
            # In practice, you'd need to parse the output properly
            
            # Resubmit command
            resubmit_cmd = ['gb2_job_resubmit', str(self.job_id), '--failed']
            
            result = subprocess.run(resubmit_cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                logger.info(f"✓ Resubmitted {status.failed} failed jobs")
            else:
                logger.error(f"✗ Resubmission failed: {result.stderr}")
        
        except Exception as e:
            logger.error(f"Error resubmitting jobs: {e}")
    
    def _is_complete(self, status: JobStatus) -> bool:
        """Check if all jobs are complete."""
        active = status.running + status.waiting
        
        return active == 0 and status.done > 0
    
    def _generate_summary(self):
        """Generate final summary report."""
        if not self.status_history:
            logger.warning("No status history to summarize")
            return
        
        final_status = self.status_history[-1]
        total_time = datetime.now() - self.start_time
        
        logger.info("\n" + "="*70)
        logger.info("Final Summary")
        logger.info("="*70)
        logger.info(f"Job ID: {self.job_id}")
        logger.info(f"Total execution time: {total_time}")
        logger.info(f"\nFinal counts:")
        logger.info(f"  Done:     {final_status.done}")
        logger.info(f"  Failed:   {final_status.failed}")
        logger.info(f"  Killed:   {final_status.killed}")
        logger.info(f"\nSuccess rate: {final_status.done/final_status.total_jobs*100:.2f}%")
        
        # Save summary to JSON
        summary_file = Path(self.config['log_dir']) / f'summary_{self.job_id}.json'
        
        summary_data = {
            'job_id': self.job_id,
            'start_time': self.start_time.isoformat(),
            'end_time': datetime.now().isoformat(),
            'total_time_seconds': total_time.total_seconds(),
            'final_status': asdict(final_status),
            'status_history': [asdict(s) for s in self.status_history],
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        logger.info(f"\nSummary saved to {summary_file}")
        
        # Generate plots if pandas available
        if PANDAS_AVAILABLE:
            self._plot_history()
    
    def _plot_history(self):
        """Generate plots of job progress over time."""
        try:
            import matplotlib.pyplot as plt
            
            df = pd.DataFrame([asdict(s) for s in self.status_history])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
            
            # Plot 1: Job counts over time
            ax1.plot(df['timestamp'], df['done'], label='Done', linewidth=2)
            ax1.plot(df['timestamp'], df['running'], label='Running', linewidth=2)
            ax1.plot(df['timestamp'], df['waiting'], label='Waiting', linewidth=2)
            ax1.plot(df['timestamp'], df['failed'], label='Failed', linewidth=2)
            
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Number of Jobs')
            ax1.set_title(f'Job Progress for {self.job_id}')
            ax1.legend()
            ax1.grid(alpha=0.3)
            
            # Plot 2: Completion rate
            ax2.plot(df['timestamp'], df['completion_rate']*100, 
                    linewidth=2, color='green')
            ax2.set_xlabel('Time')
            ax2.set_ylabel('Completion Rate (%)')
            ax2.set_title('Completion Rate Over Time')
            ax2.grid(alpha=0.3)
            
            plt.tight_layout()
            
            plot_file = Path(self.config['log_dir']) / f'progress_{self.job_id}.png'
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            logger.info(f"Progress plot saved to {plot_file}")
            
        except Exception as e:
            logger.warning(f"Could not generate plots: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Monitor Belle II DIRAC grid jobs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--job-id', type=str, required=True,
                       help='DIRAC job ID to monitor')
    parser.add_argument('--check-interval', type=int, default=300,
                       help='Status check interval in seconds')
    parser.add_argument('--auto-resubmit', action='store_true',
                       help='Automatically resubmit failed jobs')
    parser.add_argument('--max-retries', type=int, default=3,
                       help='Maximum resubmission attempts')
    parser.add_argument('--max-iterations', type=int, default=None,
                       help='Maximum monitoring iterations (None=infinite)')
    parser.add_argument('--log-file', type=str, default=None,
                       help='Log file path')
    
    args = parser.parse_args()
    
    # Setup file logging if requested
    if args.log_file:
        file_handler = logging.FileHandler(args.log_file)
        file_handler.setFormatter(
            logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        )
        logger.addHandler(file_handler)
    
    # Create monitor
    monitor = GridJobMonitor(
        job_id=args.job_id,
        check_interval=args.check_interval,
        auto_resubmit=args.auto_resubmit,
        max_retries=args.max_retries
    )
    
    # Run monitoring loop
    monitor.monitor_loop(max_iterations=args.max_iterations)


if __name__ == '__main__':
    main()
