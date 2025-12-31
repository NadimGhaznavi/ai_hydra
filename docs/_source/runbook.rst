Operations Runbook
==================

This runbook provides operational procedures for managing the AI Hydra project lifecycle using Kiro IDE. It includes token tracking usage, version management, deployment procedures, and SDLC management tasks.

Token Tracking Operations
-------------------------

The token tracking system provides comprehensive monitoring of AI token usage within the Kiro IDE environment. This section covers operational procedures for managing and analyzing token usage data.

Setting Up Token Tracking
~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Enable Token Tracking Hook**

   Create or update the token tracking hook configuration:

   .. code-block:: bash

      # Create hook configuration directory
      mkdir -p .kiro/hooks
      
      # Create token tracking hook file
      cat > .kiro/hooks/token-tracking.kiro.hook << EOF
      {
        "name": "token-tracker-hook",
        "triggers": [
          "agentExecutionCompleted",
          "agentExecutionStarted"
        ],
        "enabled": true,
        "configuration": {
          "track_tokens": true,
          "include_metadata": true,
          "error_handling": "graceful"
        }
      }
      EOF

2. **Verify Hook Installation**

   .. code-block:: bash

      # Check that the hook file exists and is properly formatted
      ls -la .kiro/hooks/token-tracking.kiro.hook
      cat .kiro/hooks/token-tracking.kiro.hook | python -m json.tool

3. **Test Token Tracking**

   .. code-block:: bash

      # Trigger an AI interaction and check for CSV file creation
      # The CSV file should be created at .kiro/token_transactions.csv
      ls -la .kiro/token_transactions.csv
      
      # Check the first few lines to verify data format
      head -n 5 .kiro/token_transactions.csv

Monitoring Token Usage
~~~~~~~~~~~~~~~~~~~~~~

1. **Daily Usage Reports**

   .. code-block:: bash

      # Generate daily usage summary
      python -c "
      import pandas as pd
      from datetime import datetime, timedelta
      
      df = pd.read_csv('.kiro/token_transactions.csv')
      df['timestamp'] = pd.to_datetime(df['timestamp'])
      
      # Today's usage
      today = datetime.now().date()
      today_usage = df[df['timestamp'].dt.date == today]['tokens_used'].sum()
      print(f'Today\\'s token usage: {today_usage:,} tokens')
      
      # This week's usage
      week_ago = today - timedelta(days=7)
      week_usage = df[df['timestamp'].dt.date >= week_ago]['tokens_used'].sum()
      print(f'This week\\'s token usage: {week_usage:,} tokens')
      "

2. **Usage by Workspace**

   .. code-block:: bash

      # Analyze usage patterns by workspace
      python -c "
      import pandas as pd
      
      df = pd.read_csv('.kiro/token_transactions.csv')
      workspace_usage = df.groupby('workspace_folder')['tokens_used'].sum().sort_values(ascending=False)
      
      print('Token usage by workspace:')
      for workspace, tokens in workspace_usage.head(10).items():
          print(f'  {workspace}: {tokens:,} tokens')
      "

3. **Error Analysis**

   .. code-block:: bash

      # Check for errors in token tracking
      python -c "
      import pandas as pd
      
      df = pd.read_csv('.kiro/token_transactions.csv')
      errors = df[df['error_occurred'] == True]
      
      if len(errors) > 0:
          print(f'Found {len(errors)} errors:')
          for _, row in errors.iterrows():
              print(f'  {row[\"timestamp\"]}: {row[\"error_message\"]}')
      else:
          print('No errors found in token tracking data')
      "

Data Maintenance
~~~~~~~~~~~~~~~~

1. **Backup Token Data**

   .. code-block:: bash

      # Create timestamped backup
      BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
      cp .kiro/token_transactions.csv .kiro/token_transactions_backup_${BACKUP_DATE}.csv
      
      # Compress old backups
      find .kiro -name "token_transactions_backup_*.csv" -mtime +7 -exec gzip {} \;

2. **Archive Old Data**

   .. code-block:: bash

      # Archive data older than 90 days
      python -c "
      import pandas as pd
      from datetime import datetime, timedelta
      
      df = pd.read_csv('.kiro/token_transactions.csv')
      df['timestamp'] = pd.to_datetime(df['timestamp'])
      
      cutoff_date = datetime.now() - timedelta(days=90)
      recent_data = df[df['timestamp'] >= cutoff_date]
      old_data = df[df['timestamp'] < cutoff_date]
      
      # Save archived data
      if len(old_data) > 0:
          archive_file = f'.kiro/token_transactions_archive_{cutoff_date.strftime(\"%Y%m%d\")}.csv'
          old_data.to_csv(archive_file, index=False)
          print(f'Archived {len(old_data)} old records to {archive_file}')
          
          # Keep only recent data in main file
          recent_data.to_csv('.kiro/token_transactions.csv', index=False)
          print(f'Kept {len(recent_data)} recent records in main file')
      else:
          print('No old data to archive')
      "

3. **Validate Data Integrity**

   .. code-block:: bash

      # Check CSV file integrity
      python -c "
      import csv
      import pandas as pd
      
      try:
          # Test CSV parsing
          df = pd.read_csv('.kiro/token_transactions.csv')
          print(f'CSV file is valid with {len(df)} records')
          
          # Check for required columns
          required_cols = ['timestamp', 'prompt_text', 'tokens_used', 'elapsed_time']
          missing_cols = [col for col in required_cols if col not in df.columns]
          
          if missing_cols:
              print(f'WARNING: Missing required columns: {missing_cols}')
          else:
              print('All required columns present')
              
          # Check for data quality issues
          null_counts = df.isnull().sum()
          if null_counts.sum() > 0:
              print('Data quality issues found:')
              for col, count in null_counts[null_counts > 0].items():
                  print(f'  {col}: {count} null values')
          else:
              print('No data quality issues found')
              
      except Exception as e:
          print(f'ERROR: CSV file validation failed: {e}')
      "

4. **Test Token Tracking Components**

   .. code-block:: bash

      # Run comprehensive token tracking validation
      python debug_file_patterns.py
      
      # Test specific components if needed
      python -c "
      from ai_hydra.token_tracker import TokenTracker
      from ai_hydra.token_tracker.models import TrackerConfig
      import tempfile
      from pathlib import Path
      
      # Test with temporary file
      with tempfile.TemporaryDirectory() as temp_dir:
          config = TrackerConfig(csv_file_path=Path(temp_dir) / 'test.csv')
          tracker = TokenTracker(config)
          
          # Test transaction recording
          success = tracker.record_transaction(
              prompt_text='Test transaction',
              tokens_used=100,
              elapsed_time=1.0,
              context={'workspace_folder': 'test'}
          )
          
          print(f'Token tracker test: {\"PASSED\" if success else \"FAILED\"}')
      "
      
      # Validate Unicode handling
      python -c "
      from ai_hydra.token_tracker import TokenTracker
      from ai_hydra.token_tracker.models import TrackerConfig
      
      config = TrackerConfig.create_for_testing()
      tracker = TokenTracker(config)
      
      # Test Unicode compatibility
      results = tracker.test_unicode_compatibility()
      print(f'Unicode support: {\"PASSED\" if results[\"unicode_support_verified\"] else \"FAILED\"}')
      print(f'Special chars: {\"PASSED\" if results[\"special_chars_handled\"] else \"FAILED\"}')
      "

Version Management
------------------

This section covers procedures for managing project versions, releases, and updates using the existing version management infrastructure.

Version Update Procedure
~~~~~~~~~~~~~~~~~~~~~~~~

The project includes an automated version update script that should be used for all version changes:

1. **Update Version Numbers**

   .. code-block:: bash

      # Update to a new version (e.g., 1.2.3)
      ./update_version.sh 1.2.3
      
      # The script will update:
      # - pyproject.toml version
      # - __init__.py version strings
      # - Documentation version references
      # - CHANGELOG.md entries

2. **Verify Version Update**

   .. code-block:: bash

      # Check that all version references were updated
      grep -r "version.*=" pyproject.toml
      grep -r "__version__" ai_hydra/__init__.py
      
      # Verify no old version references remain
      git diff --name-only | xargs grep -l "old_version_number" || echo "No old version references found"

3. **Test Version Changes**

   .. code-block:: bash

      # Run tests to ensure version changes don't break functionality
      python -m pytest tests/ -v
      
      # Test package installation
      pip install -e .
      python -c "import ai_hydra; print(ai_hydra.__version__)"

4. **Commit Version Changes**

   .. code-block:: bash

      # Commit version update changes
      git add -A
      git commit -m "chore(version): update to version 1.2.3"
      
      # Create version tag
      git tag -a v1.2.3 -m "Release version 1.2.3"
      
      # Push changes and tags
      git push origin main
      git push origin v1.2.3

Release Management
~~~~~~~~~~~~~~~~~

1. **Prepare Release**

   .. code-block:: bash

      # Ensure all tests pass
      python -m pytest tests/ --cov=ai_hydra --cov-report=term-missing
      
      # Update documentation
      cd docs
      make clean
      make html
      
      # Verify documentation builds without errors
      echo "Documentation build status: $?"

2. **Create Release Package**

   .. code-block:: bash

      # Build distribution packages
      python -m build
      
      # Verify package contents
      tar -tzf dist/ai_hydra-*.tar.gz | head -20
      
      # Test package installation
      pip install dist/ai_hydra-*.whl --force-reinstall

3. **Release Validation**

   .. code-block:: bash

      # Run comprehensive test suite
      python -m pytest tests/ --maxfail=1 -v
      
      # Test CLI commands
      ai-hydra --help
      ai-hydra-tui --help
      ai-hydra-router --help
      
      # Verify token tracking functionality
      python -c "from ai_hydra.token_tracker import TokenTracker; print('Token tracker import successful')"

Deployment Procedures
--------------------

Development Environment Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Clone and Setup Repository**

   .. code-block:: bash

      # Clone repository
      git clone <repository-url>
      cd ai-hydra
      
      # Create virtual environment
      python -m venv hydra_venv
      source hydra_venv/bin/activate  # On Windows: hydra_venv\Scripts\activate
      
      # Install dependencies
      pip install -r requirements.txt
      pip install -e .

2. **Configure Development Environment**

   .. code-block:: bash

      # Set up pre-commit hooks
      pre-commit install
      
      # Configure git hooks for automated testing
      cp .git/hooks/pre-commit.sample .git/hooks/pre-commit
      chmod +x .git/hooks/pre-commit

3. **Verify Development Setup**

   .. code-block:: bash

      # Run test suite
      python -m pytest tests/ -v
      
      # Test documentation build
      cd docs && make html
      
      # Test CLI functionality
      ai-hydra --version
      ai-hydra-tui --help

Production Deployment
~~~~~~~~~~~~~~~~~~~~

1. **Server Environment Preparation**

   .. code-block:: bash

      # Update system packages
      sudo apt update && sudo apt upgrade -y
      
      # Install Python 3.11+
      sudo apt install python3.11 python3.11-venv python3.11-dev
      
      # Install system dependencies
      sudo apt install build-essential libzmq3-dev

2. **Application Deployment**

   .. code-block:: bash

      # Create application directory
      sudo mkdir -p /opt/ai-hydra
      sudo chown $USER:$USER /opt/ai-hydra
      cd /opt/ai-hydra
      
      # Deploy application
      git clone <repository-url> .
      python3.11 -m venv venv
      source venv/bin/activate
      pip install -r requirements.txt
      pip install -e .

3. **Service Configuration**

   .. code-block:: bash

      # Create systemd service file
      sudo tee /etc/systemd/system/ai-hydra.service << EOF
      [Unit]
      Description=AI Hydra Service
      After=network.target
      
      [Service]
      Type=simple
      User=$USER
      WorkingDirectory=/opt/ai-hydra
      Environment=PATH=/opt/ai-hydra/venv/bin
      ExecStart=/opt/ai-hydra/venv/bin/python -m ai_hydra.headless_server
      Restart=always
      RestartSec=10
      
      [Install]
      WantedBy=multi-user.target
      EOF
      
      # Enable and start service
      sudo systemctl daemon-reload
      sudo systemctl enable ai-hydra
      sudo systemctl start ai-hydra

4. **Deployment Verification**

   .. code-block:: bash

      # Check service status
      sudo systemctl status ai-hydra
      
      # Test service connectivity
      python -c "
      import zmq
      context = zmq.Context()
      socket = context.socket(zmq.REQ)
      socket.connect('tcp://localhost:5555')
      socket.send_json({'type': 'status'})
      response = socket.recv_json(zmq.NOBLOCK)
      print(f'Service response: {response}')
      "

SDLC Management
--------------

Development Workflow
~~~~~~~~~~~~~~~~~~~

1. **Feature Development**

   .. code-block:: bash

      # Create feature branch
      git checkout -b feature/new-feature-name
      
      # Make changes and commit regularly
      git add .
      git commit -m "feat(component): implement new feature"
      
      # Run tests before pushing
      python -m pytest tests/ -v
      
      # Push feature branch
      git push origin feature/new-feature-name

2. **Code Review Process**

   .. code-block:: bash

      # Before creating pull request, ensure:
      # - All tests pass
      python -m pytest tests/ --cov=ai_hydra
      
      # - Documentation is updated
      cd docs && make html
      
      # - Code follows style guidelines
      black ai_hydra/ tests/
      isort ai_hydra/ tests/
      flake8 ai_hydra/ tests/

3. **Integration and Deployment**

   .. code-block:: bash

      # After PR approval, merge to main
      git checkout main
      git pull origin main
      git merge feature/new-feature-name
      
      # Run full test suite
      python -m pytest tests/ --maxfail=1
      
      # Update version if needed
      ./update_version.sh 1.2.4
      
      # Deploy to production
      git push origin main

Quality Assurance
~~~~~~~~~~~~~~~~

1. **Automated Testing**

   .. code-block:: bash

      # Run comprehensive test suite
      python -m pytest tests/ -v --cov=ai_hydra --cov-report=html
      
      # Run property-based tests
      python -m pytest tests/property/ --maxfail=1
      
      # Run integration tests
      python -m pytest tests/integration/ -v

2. **Performance Testing**

   .. code-block:: bash

      # Run performance benchmarks
      python -m pytest tests/performance/ -v
      
      # Profile memory usage
      python -m memory_profiler examples/demo_headless_server.py
      
      # Test with different configurations
      python -c "
      from ai_hydra import HydraMgr
      from ai_hydra.config import SimulationConfig
      
      # Test with various budget sizes
      for budget in [50, 100, 200]:
          config = SimulationConfig(move_budget=budget)
          mgr = HydraMgr(config)
          print(f'Budget {budget}: Initialization successful')
      "

3. **Documentation Quality**

   .. code-block:: bash

      # Validate documentation
      python test_documentation.py
      
      # Check for broken links
      cd docs && make linkcheck
      
      # Verify API documentation completeness
      python -c "
      import ai_hydra
      import inspect
      
      # Check that all public classes have docstrings
      for name, obj in inspect.getmembers(ai_hydra):
          if inspect.isclass(obj) and not name.startswith('_'):
              if not obj.__doc__:
                  print(f'WARNING: {name} missing docstring')
      "

Monitoring and Maintenance
~~~~~~~~~~~~~~~~~~~~~~~~~

1. **System Health Monitoring**

   .. code-block:: bash

      # Check system resource usage
      python -c "
      import psutil
      import time
      
      # Monitor for 60 seconds
      for i in range(12):
          cpu = psutil.cpu_percent(interval=1)
          memory = psutil.virtual_memory().percent
          print(f'CPU: {cpu:5.1f}% | Memory: {memory:5.1f}%')
          time.sleep(5)
      "

2. **Log Analysis**

   .. code-block:: bash

      # Analyze application logs
      tail -f /var/log/ai-hydra/application.log | grep ERROR
      
      # Check for performance issues
      grep -i "slow\|timeout\|performance" /var/log/ai-hydra/application.log
      
      # Monitor token tracking logs
      grep "token_tracker" /var/log/ai-hydra/application.log | tail -20

3. **Maintenance Tasks**

   .. code-block:: bash

      # Update dependencies
      pip list --outdated
      pip install --upgrade -r requirements.txt
      
      # Clean up old files
      find . -name "*.pyc" -delete
      find . -name "__pycache__" -type d -exec rm -rf {} +
      
      # Rotate logs
      sudo logrotate /etc/logrotate.d/ai-hydra

Troubleshooting
--------------

Common Issues and Solutions
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Token Tracking Not Working**

.. code-block:: bash

   # Check hook configuration
   cat .kiro/hooks/token-tracking.kiro.hook
   
   # Verify file permissions
   ls -la .kiro/
   
   # Test file patterns preservation with debug script
   python debug_file_patterns.py
   
   # Check for error messages
   grep -i "token\|error" ~/.kiro/logs/kiro.log

**File Patterns Not Being Preserved**

.. code-block:: bash

   # Run comprehensive file patterns debug test
   python debug_file_patterns.py
   
   # If debug script fails, check specific components:
   
   # Test MetadataCollector directly
   python -c "
   from ai_hydra.token_tracker.metadata_collector import MetadataCollector
   from ai_hydra.token_tracker.error_handler import TokenTrackerErrorHandler
   
   collector = MetadataCollector(TokenTrackerErrorHandler())
   context = {'file_patterns': ['*.py', '*.md']}
   result = collector.get_hook_context(context)
   print(f'Hook context result: {result}')
   "
   
   # Test full metadata collection
   python -c "
   from ai_hydra.token_tracker.metadata_collector import MetadataCollector
   from ai_hydra.token_tracker.error_handler import TokenTrackerErrorHandler
   
   collector = MetadataCollector(TokenTrackerErrorHandler())
   context = {'file_patterns': ['*.py'], 'trigger_type': 'test'}
   metadata = collector.collect_execution_metadata(context)
   print(f'Metadata file_patterns: {metadata.get(\"file_patterns\", \"NOT FOUND\")}')
   "

**Performance Issues**

.. code-block:: bash

   # Profile application performance
   python -m cProfile -o profile.stats examples/demo_headless_server.py
   python -c "import pstats; pstats.Stats('profile.stats').sort_stats('cumulative').print_stats(20)"
   
   # Monitor resource usage
   top -p $(pgrep -f ai_hydra)

**Service Startup Failures**

.. code-block:: bash

   # Check service logs
   sudo journalctl -u ai-hydra -f
   
   # Verify configuration
   python -c "from ai_hydra.config import SimulationConfig; print('Config validation passed')"
   
   # Test manual startup
   cd /opt/ai-hydra && source venv/bin/activate && python -m ai_hydra.headless_server

Emergency Procedures
~~~~~~~~~~~~~~~~~~~

**Service Recovery**

.. code-block:: bash

   # Stop service
   sudo systemctl stop ai-hydra
   
   # Backup current state
   cp -r /opt/ai-hydra /opt/ai-hydra.backup.$(date +%Y%m%d_%H%M%S)
   
   # Restore from known good state
   git checkout main
   git pull origin main
   
   # Restart service
   sudo systemctl start ai-hydra
   sudo systemctl status ai-hydra

**Data Recovery**

.. code-block:: bash

   # Restore token tracking data from backup
   cp .kiro/token_transactions_backup_*.csv .kiro/token_transactions.csv
   
   # Validate restored data
   python -c "
   import pandas as pd
   df = pd.read_csv('.kiro/token_transactions.csv')
   print(f'Restored {len(df)} transaction records')
   "

This runbook provides comprehensive operational procedures for managing the AI Hydra project. Regular use of these procedures will ensure smooth operation, proper maintenance, and quick resolution of issues.