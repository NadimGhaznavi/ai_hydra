Token Tracking System Usage Guide
===================================

The Token Tracking System provides comprehensive monitoring and analysis of AI token usage within the Kiro IDE environment. This system automatically captures token consumption data from AI interactions and stores it in a structured format for analysis and optimization.

This guide provides complete instructions for configuring, using, and troubleshooting the token tracking system.

Quick Start Guide
-----------------

Getting Started in 5 Minutes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Step 1: Verify Installation**

Check if the token tracking system is already set up:

.. code-block:: bash

   # Check if hook files exist
   ls -la .kiro/hooks/token-tracking*.kiro.hook
   
   # Check if CSV file is being created
   ls -la .kiro/token_transactions.csv

**Step 2: Enable Automatic Tracking**

If hook files don't exist, the system will create them automatically on first use. To manually verify:

.. code-block:: bash

   # Verify hooks are enabled
   cat .kiro/hooks/token-tracking.kiro.hook | grep '"enabled": true'

**Step 3: Test the System**

Perform any AI interaction in Kiro IDE (ask a question, run an agent task), then check:

.. code-block:: bash

   # View recent transactions
   tail -5 .kiro/token_transactions.csv
   
   # Count total transactions
   wc -l .kiro/token_transactions.csv

**Step 4: Basic Analysis**

.. code-block:: python

   import pandas as pd
   
   # Load and analyze your data
   df = pd.read_csv('.kiro/token_transactions.csv')
   print(f"Total transactions: {len(df)}")
   print(f"Total tokens used: {df['tokens_used'].sum()}")
   print(f"Average tokens per interaction: {df['tokens_used'].mean():.1f}")

**Step 5: Configure (Optional)**

.. code-block:: python

   from ai_hydra.token_tracker.models import TrackerConfig
   
   # Create custom configuration
   config = TrackerConfig(
       max_prompt_length=2000,  # Increase prompt storage
       backup_enabled=True,     # Enable daily backups
       log_level="DEBUG"        # More detailed logging
   )
   
   # Apply configuration (requires restart)
   config.save_to_file(".kiro/token_tracker_config.json")

That's it! Your token tracking system is now monitoring all AI interactions automatically.

Overview
--------

The Token Tracking System consists of several key components:

* **Token Tracker Service**: Core service that processes and stores token usage data
* **Agent Hook Integration**: Automatic triggering system for seamless token tracking
* **CSV Data Store**: Persistent storage for transaction history with concurrent access safety
* **Metadata Collector**: Contextual information gathering from Kiro IDE environment
* **Error Handler**: Robust error management and recovery mechanisms

Key Features
------------

**Implemented Features:**

* **Core Data Models**: Robust TokenTransaction and TrackerConfig classes with validation
* **Thread-Safe CSV Operations**: Concurrent access protection with file locking mechanisms
* **Transaction Persistence**: Reliable CSV-based storage with data integrity validation
* **Special Character Handling**: Comprehensive Unicode and CSV special character support
* **Error Resilience**: Graceful error handling with recovery mechanisms
* **Configurable Operation**: Flexible configuration with validation and defaults
* **Property-Based Testing**: Comprehensive test coverage with universal correctness properties
* **Metadata Collection**: Complete context gathering from Kiro IDE environment
* **Agent Hook Integration**: Automatic token tracking with comprehensive configuration management
* **Runtime Configuration**: Dynamic configuration updates without system restart
* **Kiro Hook Configuration**: Complete .kiro.hook file integration with comprehensive configuration

**Planned Features:**

* **Concurrent Access Safety**: Advanced multi-process coordination (in development)

Architecture
------------

Current Implementation Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The implemented core system consists of:

.. mermaid::

   graph TD
       A[TokenTransaction] --> B[TokenTracker Service]
       B --> C[CSVWriter]
       C --> D[CSV File Storage]
       
       E[TrackerConfig] --> B
       F[ErrorHandler] --> B
       F --> C
       
       G[Property Tests] --> A
       G --> B
       G --> C

**Implemented Components:**

**TokenTransaction Model**
    Immutable data class representing a single token usage event with comprehensive validation and CSV serialization.

**TokenTracker Service**
    Core service that processes token transactions, validates data, and coordinates CSV operations with error handling.

**CSVWriter Component**
    Thread-safe CSV operations with file locking, transaction serialization, and Unicode/special character handling.

**TrackerConfig**
    Configuration management with validation, defaults, and environment-specific settings.

**ErrorHandler**
    Comprehensive error management with recovery mechanisms and graceful degradation.

Planned Integration Architecture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The complete system will include:

.. mermaid::

   graph TD
       A[Kiro IDE Event] --> B[Agent Hook Trigger]
       B --> C[MetadataCollector]
       C --> D[TokenTracker Service]
       D --> E[CSVWriter]
       E --> F[Transaction History]
       
       G[ErrorHandler] --> D
       G --> E
       
       H[Configuration] --> B
       H --> D

**Planned Components:**

**MetadataCollector** (In Development)
    Gathers contextual information from the Kiro IDE environment including workspace details and execution metadata.

**Agent Hook Integration** (In Development)
    Monitors Kiro IDE events and automatically triggers token tracking when AI interactions occur.

Data Model
----------

Token Transaction Structure
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each token transaction record contains the following fields:

.. list-table::
   :header-rows: 1
   :widths: 20 15 65

   * - Field
     - Type
     - Description
   * - timestamp
     - datetime
     - ISO 8601 timestamp of the transaction
   * - prompt_text
     - string
     - The prompt text sent to the AI (truncated if necessary)
   * - tokens_used
     - integer
     - Number of tokens consumed in the interaction
   * - elapsed_time
     - float
     - Time taken for the AI interaction in seconds
   * - session_id
     - string
     - Unique identifier for the Kiro IDE session
   * - workspace_folder
     - string
     - Name of the active workspace folder
   * - hook_trigger_type
     - string
     - Type of event that triggered the interaction
   * - agent_execution_id
     - string
     - Unique identifier for the agent execution
   * - file_patterns
     - string
     - File patterns that triggered the interaction (optional)
   * - hook_name
     - string
     - Name of the specific hook that initiated tracking
   * - error_occurred
     - boolean
     - Whether an error occurred during the interaction
   * - error_message
     - string
     - Error message if an error occurred (optional)

CSV Schema
~~~~~~~~~~

The CSV file uses the following structure:

.. code-block:: csv

   timestamp,prompt_text,tokens_used,elapsed_time,session_id,workspace_folder,hook_trigger_type,agent_execution_id,file_patterns,hook_name,error_occurred,error_message
   2024-01-15T10:30:45.123Z,"Implement token tracking system",1250,2.34,sess_abc123,ai_hydra,agentExecutionCompleted,exec_def456,"*.py;*.md",token-tracker-hook,false,

Configuration Guide
------------------

Complete Configuration Instructions
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The token tracking system offers flexible configuration options for different use cases. Here's how to configure it for your needs:

**Basic Configuration (Recommended for Most Users)**

.. code-block:: python

   from ai_hydra.token_tracker.models import TrackerConfig
   
   # Standard configuration for daily development
   config = TrackerConfig(
       enabled=True,                           # Enable tracking
       csv_file_path=".kiro/token_transactions.csv",  # Default location
       max_prompt_length=1000,                 # Reasonable prompt size
       backup_enabled=True,                    # Daily backups
       retention_days=90,                      # 3 months retention
       log_level="INFO"                        # Standard logging
   )

**Development Configuration**

.. code-block:: python

   # Configuration for active development with detailed logging
   dev_config = TrackerConfig(
       enabled=True,
       max_prompt_length=2000,                 # Longer prompts for debugging
       backup_enabled=False,                   # Skip backups during dev
       retention_days=30,                      # Shorter retention
       log_level="DEBUG",                      # Detailed logging
       enable_validation=True,                 # Extra validation
       file_lock_timeout_seconds=10.0          # Longer timeout for debugging
   )

**Production Configuration**

.. code-block:: python

   # Configuration for production environments
   prod_config = TrackerConfig(
       enabled=True,
       csv_file_path="/var/log/kiro/token_transactions.csv",  # System location
       max_prompt_length=500,                  # Limit prompt size
       backup_enabled=True,
       backup_interval_hours=12,               # Twice daily backups
       retention_days=365,                     # Full year retention
       compression_enabled=True,               # Compress old data
       log_level="WARNING"                     # Minimal logging
   )

**Privacy-Focused Configuration**

.. code-block:: python

   # Configuration for sensitive environments
   privacy_config = TrackerConfig(
       enabled=True,
       max_prompt_length=100,                  # Minimal prompt storage
       backup_enabled=True,
       retention_days=7,                       # Short retention
       log_level="ERROR"                       # Minimal logging
   )

**Configuration File Management**

Save and load configurations from files:

.. code-block:: python

   # Save configuration
   config.save_to_file(".kiro/token_tracker_config.json")
   
   # Load configuration
   config = TrackerConfig.load_from_file(".kiro/token_tracker_config.json")
   
   # Validate configuration
   issues = config.validate()
   if issues:
       print(f"Configuration issues: {issues}")
   else:
       print("Configuration is valid")

**Environment-Specific Setup**

.. code-block:: bash

   # Development environment
   export KIRO_TOKEN_TRACKER_LOG_LEVEL=DEBUG
   export KIRO_TOKEN_TRACKER_MAX_PROMPT_LENGTH=2000
   
   # Production environment
   export KIRO_TOKEN_TRACKER_LOG_LEVEL=WARNING
   export KIRO_TOKEN_TRACKER_RETENTION_DAYS=365
   export KIRO_TOKEN_TRACKER_BACKUP_ENABLED=true

**Runtime Configuration Updates**

Update configuration without restarting:

.. code-block:: python

   from ai_hydra.token_tracker.hook import TokenTrackingHook
   
   # Get current hook instance
   hook = TokenTrackingHook.get_instance()
   
   # Update specific settings
   changes = {
       "max_prompt_length": 1500,
       "log_level": "DEBUG",
       "backup_enabled": True
   }
   
   success = hook.apply_configuration_changes(changes)
   if success:
       print("Configuration updated successfully")
   
   # Or update entire configuration
   new_config = TrackerConfig(max_prompt_length=2000, log_level="INFO")
   hook.update_configuration(new_config)

Token Tracker Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The token tracking system is configured through the ``TrackerConfig`` class:

.. code-block:: python

   from ai_hydra.token_tracker.models import TrackerConfig
   
   # Basic configuration
   config = TrackerConfig(
       enabled=True,                           # Enable/disable tracking
       csv_file_path=".kiro/token_transactions.csv",  # CSV file location
       max_prompt_length=1000,                 # Maximum prompt text length
       backup_enabled=True,                    # Enable automatic backups
       backup_interval_hours=24,               # Backup frequency
       compression_enabled=False,              # Compress archived files
       retention_days=365,                     # Data retention period
       auto_create_directories=True,           # Create directories if needed
       file_lock_timeout_seconds=5.0,          # File locking timeout
       enable_validation=True,                 # Enable data validation
       log_level="INFO"                        # Logging level
   )
   
   # Environment-specific configurations
   
   # Development configuration
   dev_config = TrackerConfig.create_for_testing()
   
   # Production configuration
   prod_config = TrackerConfig.create_for_production()
   
   # Validate configuration
   issues = config.validate()
   if issues:
       print(f"Configuration issues: {issues}")

**Configuration Options:**

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Parameter
     - Type
     - Description
   * - enabled
     - bool
     - Enable or disable token tracking
   * - csv_file_path
     - Path
     - Location of the CSV file for storing transactions
   * - max_prompt_length
     - int
     - Maximum length of prompt text to store (truncated if longer)
   * - backup_enabled
     - bool
     - Whether to create automatic backups
   * - backup_interval_hours
     - int
     - Hours between automatic backups
   * - retention_days
     - int
     - Number of days to retain transaction data
   * - auto_create_directories
     - bool
     - Automatically create directories if they don't exist
   * - file_lock_timeout_seconds
     - float
     - Timeout for acquiring file locks
   * - enable_validation
     - bool
     - Enable data validation before storing transactions
   * - log_level
     - str
     - Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

Agent Hook Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

The agent hook system provides comprehensive configuration management with runtime updates:

**Hook Configuration Management:**

.. code-block:: python

   from ai_hydra.token_tracker.hook import TokenTrackingHook
   from ai_hydra.token_tracker.models import TrackerConfig
   
   # Initialize hook with configuration
   config = TrackerConfig(
       enabled=True,
       max_prompt_length=1000,
       backup_enabled=True,
       log_level="INFO"
   )
   hook = TokenTrackingHook(config)
   
   # Runtime configuration updates
   new_config = TrackerConfig(
       enabled=True,
       max_prompt_length=2000,  # Increased limit
       backup_enabled=False,    # Disabled backups
       log_level="DEBUG"        # More verbose logging
   )
   
   success = hook.update_configuration(new_config)
   if success:
       print("Configuration updated successfully")
   
   # Partial configuration changes
   changes = {
       "max_prompt_length": 1500,
       "log_level": "WARNING"
   }
   hook.apply_configuration_changes(changes)
   
   # Enable/disable hook dynamically
   hook.disable()  # Temporarily disable tracking
   hook.enable()   # Re-enable tracking
   
   # Configuration persistence
   hook.save_configuration_to_file(".kiro/token_tracker_config.json")
   hook.reload_configuration_from_file(".kiro/token_tracker_config.json")
   
   # Configuration validation
   validation_result = hook.validate_configuration()
   if not validation_result["valid"]:
       print(f"Configuration issues: {validation_result['issues']}")

**Configuration State Management Features:**

* **Runtime Updates**: Change configuration without restarting the system
* **Partial Changes**: Update only specific configuration parameters
* **State Consistency**: Maintain consistent state across enable/disable cycles
* **File Persistence**: Save and reload configuration from JSON files
* **Validation**: Comprehensive configuration validation with error reporting
* **Error Handling**: Graceful handling of invalid configuration attempts

**Hook Configuration File Format:**

The hook configuration can be saved to and loaded from JSON files:

.. code-block:: json

   {
     "enabled": true,
     "csv_file_path": ".kiro/token_transactions.csv",
     "max_prompt_length": 1000,
     "backup_enabled": true,
     "backup_interval_hours": 24,
     "retention_days": 365,
     "file_lock_timeout_seconds": 5.0,
     "max_concurrent_writes": 10,
     "enable_validation": true,
     "log_level": "INFO",
     "hook_enabled": true,
     "hook_name": "token-tracking-hook"
   }

**Configuration Schema Validation:**

The system provides a complete configuration schema for validation:

.. code-block:: python

   # Get configuration schema
   schema = hook.get_configuration_schema()
   
   # Schema includes type information, constraints, and defaults
   print(f"Max prompt length range: {schema['max_prompt_length']['minimum']}-{schema['max_prompt_length']['maximum']}")
   print(f"Valid log levels: {schema['log_level']['enum']}")

Agent Hook Integration (Complete)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The agent hook system is fully implemented and configured through comprehensive ``.kiro.hook`` files:

**Primary Hook Configuration (.kiro/hooks/token-tracking.kiro.hook):**

.. code-block:: json

   {
     "enabled": true,
     "name": "Token Tracker Hook",
     "description": "Automatically tracks AI token usage during agent executions",
     "version": "1.0.0",
     "when": {
       "type": "agentExecutionCompleted",
       "patterns": ["*"]
     },
     "then": {
       "type": "askAgent",
       "prompt": "Track token usage for this execution..."
     },
     "configuration": {
       "enabled": true,
       "track_tokens": true,
       "include_metadata": true,
       "error_handling": "graceful",
       "max_prompt_length": 1000,
       "csv_file_path": ".kiro/token_transactions.csv",
       "backup_enabled": true,
       "auto_create_directories": true,
       "enable_validation": true,
       "log_level": "INFO"
     },
     "error_handling": {
       "on_file_error": "log_and_continue",
       "on_validation_error": "log_and_skip",
       "on_unexpected_error": "log_and_continue",
       "retry_attempts": 3,
       "fallback_behavior": "graceful_degradation"
     }
   }

**Execution Start Hook (.kiro/hooks/token-tracking-start.kiro.hook):**

.. code-block:: json

   {
     "enabled": true,
     "name": "Token Tracker Start Hook",
     "description": "Tracks the start of AI agent executions",
     "version": "1.0.0",
     "when": {
       "type": "agentExecutionStarted",
       "patterns": ["*"]
     },
     "configuration": {
       "enabled": true,
       "track_execution_start": true,
       "error_handling": "graceful",
       "log_level": "INFO"
     }
   }

**Hook Features:**

* **Automatic Triggering**: Responds to agentExecutionCompleted and agentExecutionStarted events
* **Comprehensive Configuration**: Full configuration management with validation and error handling
* **Error Resilience**: Graceful degradation with retry logic and fallback mechanisms
* **Performance Optimization**: Configurable timeouts and resource limits
* **Metadata Capture**: Complete context gathering including workspace, execution ID, and file patterns
* **Concurrent Safety**: Handles multiple simultaneous executions safely

**Hook Configuration Management:**

The hooks support comprehensive configuration management:

.. code-block:: python

   # Hook configuration is managed through the .kiro.hook files
   # Runtime configuration updates are handled by the TokenTrackingHook class
   
   from ai_hydra.token_tracker.hook import TokenTrackingHook
   
   # The hook automatically loads configuration from .kiro.hook files
   # and provides runtime configuration management capabilities

Usage Examples and Data Analysis
--------------------------------

Practical Usage Scenarios
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Scenario 1: Daily Development Monitoring**

Track your daily token usage to understand patterns:

.. code-block:: python

   import pandas as pd
   import matplotlib.pyplot as plt
   from datetime import datetime, timedelta
   
   # Load recent data
   df = pd.read_csv('.kiro/token_transactions.csv')
   df['timestamp'] = pd.to_datetime(df['timestamp'])
   
   # Filter last 7 days
   week_ago = datetime.now() - timedelta(days=7)
   recent_df = df[df['timestamp'] >= week_ago]
   
   # Daily usage summary
   daily_usage = recent_df.groupby(recent_df['timestamp'].dt.date).agg({
       'tokens_used': ['sum', 'count', 'mean'],
       'elapsed_time': 'sum'
   }).round(2)
   
   print("Daily Token Usage Summary:")
   print(daily_usage)

**Scenario 2: Project-Specific Analysis**

Analyze token usage by workspace/project:

.. code-block:: python

   # Group by workspace
   workspace_analysis = df.groupby('workspace_folder').agg({
       'tokens_used': ['sum', 'mean', 'count'],
       'elapsed_time': ['sum', 'mean']
   }).round(2)
   
   print("Token Usage by Project:")
   print(workspace_analysis.sort_values(('tokens_used', 'sum'), ascending=False))
   
   # Find most expensive interactions
   expensive_interactions = df.nlargest(10, 'tokens_used')[
       ['timestamp', 'workspace_folder', 'tokens_used', 'prompt_text']
   ]
   print("\nMost Expensive Interactions:")
   print(expensive_interactions)

**Scenario 3: Performance Analysis**

Identify slow or inefficient interactions:

.. code-block:: python

   # Calculate tokens per second
   df['tokens_per_second'] = df['tokens_used'] / df['elapsed_time']
   
   # Find slow interactions (low tokens per second)
   slow_interactions = df.nsmallest(10, 'tokens_per_second')[
       ['timestamp', 'tokens_used', 'elapsed_time', 'tokens_per_second', 'workspace_folder']
   ]
   
   print("Slowest Interactions (tokens/second):")
   print(slow_interactions)
   
   # Performance trends over time
   daily_performance = df.groupby(df['timestamp'].dt.date)['tokens_per_second'].mean()
   
   plt.figure(figsize=(12, 6))
   daily_performance.plot(kind='line')
   plt.title('Token Processing Performance Over Time')
   plt.ylabel('Tokens per Second')
   plt.xlabel('Date')
   plt.show()

**Scenario 4: Cost Estimation**

Estimate costs based on token usage:

.. code-block:: python

   # Define cost per token (example rates)
   COST_PER_TOKEN = {
       'gpt-4': 0.00003,      # $0.03 per 1K tokens
       'gpt-3.5-turbo': 0.000002,  # $0.002 per 1K tokens
       'claude': 0.000015,    # $0.015 per 1K tokens
   }
   
   # Assume GPT-4 for cost calculation
   df['estimated_cost'] = df['tokens_used'] * COST_PER_TOKEN['gpt-4']
   
   # Daily cost analysis
   daily_cost = df.groupby(df['timestamp'].dt.date)['estimated_cost'].sum()
   
   print(f"Total estimated cost: ${df['estimated_cost'].sum():.2f}")
   print(f"Average daily cost: ${daily_cost.mean():.2f}")
   print(f"Highest daily cost: ${daily_cost.max():.2f}")
   
   # Monthly projection
   monthly_projection = daily_cost.mean() * 30
   print(f"Monthly projection: ${monthly_projection:.2f}")

**Scenario 5: Hook Trigger Analysis**

Understand which types of interactions consume the most tokens:

.. code-block:: python

   # Analyze by hook trigger type
   trigger_analysis = df.groupby('hook_trigger_type').agg({
       'tokens_used': ['sum', 'mean', 'count'],
       'elapsed_time': 'mean'
   }).round(2)
   
   print("Usage by Trigger Type:")
   print(trigger_analysis)
   
   # Visualize trigger distribution
   trigger_counts = df['hook_trigger_type'].value_counts()
   
   plt.figure(figsize=(10, 6))
   trigger_counts.plot(kind='bar')
   plt.title('Interactions by Trigger Type')
   plt.ylabel('Number of Interactions')
   plt.xticks(rotation=45)
   plt.tight_layout()
   plt.show()

**Scenario 6: Error Analysis**

Identify and analyze failed interactions:

.. code-block:: python

   # Filter error transactions
   error_df = df[df['error_occurred'] == True]
   
   if len(error_df) > 0:
       print(f"Error rate: {len(error_df) / len(df) * 100:.1f}%")
       
       # Group errors by type
       error_summary = error_df.groupby('error_message').size().sort_values(ascending=False)
       print("\nMost Common Errors:")
       print(error_summary.head(10))
       
       # Error trends over time
       error_daily = error_df.groupby(error_df['timestamp'].dt.date).size()
       print(f"\nDaily error counts:")
       print(error_daily.tail(7))
   else:
       print("No errors found in transaction history")

Advanced Data Analysis
~~~~~~~~~~~~~~~~~~~~~~

**Custom Reporting Functions**

Create reusable analysis functions:

.. code-block:: python

   def generate_usage_report(csv_path='.kiro/token_transactions.csv', days=30):
       """Generate comprehensive usage report."""
       df = pd.read_csv(csv_path)
       df['timestamp'] = pd.to_datetime(df['timestamp'])
       
       # Filter by date range
       cutoff_date = datetime.now() - timedelta(days=days)
       df = df[df['timestamp'] >= cutoff_date]
       
       report = {
           'period': f"Last {days} days",
           'total_transactions': len(df),
           'total_tokens': df['tokens_used'].sum(),
           'average_tokens_per_interaction': df['tokens_used'].mean(),
           'total_time_spent': df['elapsed_time'].sum() / 3600,  # hours
           'most_active_workspace': df['workspace_folder'].mode().iloc[0] if len(df) > 0 else 'N/A',
           'error_rate': (df['error_occurred'].sum() / len(df) * 100) if len(df) > 0 else 0
       }
       
       return report
   
   # Generate report
   report = generate_usage_report(days=7)
   print("Weekly Usage Report:")
   for key, value in report.items():
       print(f"  {key}: {value}")

**Export and Sharing**

Export data for external analysis:

.. code-block:: python

   from ai_hydra.token_tracker import TokenTracker
   
   # Initialize tracker
   tracker = TokenTracker()
   
   # Export filtered data
   success = tracker.export_data(
       output_path="weekly_report.csv",
       format="csv",
       filters={
           "start_date": "2024-01-01",
           "end_date": "2024-01-07",
           "workspace_folder": "my_project"
       }
   )
   
   if success:
       print("Data exported successfully")

**Integration with Business Intelligence Tools**

Connect to BI tools for advanced analytics:

.. code-block:: python

   # Example: Export to Excel with multiple sheets
   with pd.ExcelWriter('token_analysis.xlsx') as writer:
       # Summary sheet
       summary_df = df.groupby('workspace_folder').agg({
           'tokens_used': ['sum', 'mean', 'count']
       })
       summary_df.to_excel(writer, sheet_name='Summary')
       
       # Daily trends sheet
       daily_df = df.groupby(df['timestamp'].dt.date)['tokens_used'].sum()
       daily_df.to_excel(writer, sheet_name='Daily_Trends')
       
       # Raw data sheet (last 1000 records)
       df.tail(1000).to_excel(writer, sheet_name='Raw_Data', index=False)

Current Usage (Core System)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The core token tracking system can be used programmatically:

.. code-block:: python

   from ai_hydra.token_tracker import TokenTracker
   from ai_hydra.token_tracker.models import TokenTransaction, TrackerConfig
   from datetime import datetime
   
   # Initialize tracker with configuration
   config = TrackerConfig(
       csv_file_path=".kiro/token_transactions.csv",
       max_prompt_length=1000,
       backup_enabled=True,
       enable_validation=True
   )
   tracker = TokenTracker(config)
   
   # Record a transaction manually
   success = tracker.record_transaction(
       prompt_text="Analyze this code for improvements",
       tokens_used=150,
       elapsed_time=1.5,
       context={
           "workspace_folder": "my_project",
           "hook_trigger_type": "manual",
           "session_id": "session_123"
       }
   )
   
   if success:
       print("Transaction recorded successfully")
   
   # Retrieve transaction history
   history = tracker.get_transaction_history(limit=10)
   for transaction in history:
       print(f"Tokens: {transaction.tokens_used}, Time: {transaction.elapsed_time}s")
   
   # Get system statistics
   stats = tracker.get_statistics()
   print(f"Total transactions: {stats['transactions_recorded']}")
   print(f"Total tokens tracked: {stats['total_tokens_tracked']}")

**Special Character and Unicode Support:**

.. code-block:: python

   # The system handles Unicode and special characters automatically
   success = tracker.record_transaction(
       prompt_text="Analyze this: 中文, émojis 😀, and \"quotes\"",
       tokens_used=75,
       elapsed_time=0.8,
       context={"workspace_folder": "unicode_test"}
   )
   
   # Test Unicode compatibility
   unicode_results = tracker.test_unicode_compatibility()
   print(f"Unicode support verified: {unicode_results['unicode_support_verified']}")

**CSV File Validation:**

.. code-block:: python

   # Validate CSV file integrity
   integrity_results = tracker.validate_csv_integrity()
   print(f"File exists: {integrity_results['file_exists']}")
   print(f"Valid rows: {integrity_results['valid_rows']}")
   print(f"Total rows: {integrity_results['total_rows']}")

Automatic Usage (Hook Integration Complete)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The token tracking system now operates automatically through the implemented agent hooks:

**Automatic Operation:**

The system automatically tracks token usage from all AI interactions in the Kiro IDE through two configured hooks:

1. **Token Tracking Hook** (`.kiro/hooks/token-tracking.kiro.hook`):
   - Triggers on `agentExecutionCompleted` events
   - Records complete transaction data to CSV
   - Handles error recovery and retry logic

2. **Token Tracking Start Hook** (`.kiro/hooks/token-tracking-start.kiro.hook`):
   - Triggers on `agentExecutionStarted` events  
   - Initializes tracking context for execution monitoring

**Hook Status Verification:**

.. code-block:: bash

   # Check if hooks are properly configured
   ls -la .kiro/hooks/token-tracking*.kiro.hook
   
   # Verify CSV file is being created
   ls -la .kiro/token_transactions.csv
   
   # Check recent transactions
   tail -5 .kiro/token_transactions.csv

**Configuration Management:**

The hooks are configured with comprehensive settings for production use:

- **Error Handling**: Graceful degradation with retry logic
- **Performance**: 5-second timeout with memory limits
- **Data Integrity**: File locking and validation
- **Backup**: Automatic backup creation every 24 hours
- **Monitoring**: Detailed logging and error reporting

**Troubleshooting Hook Integration:**

If automatic tracking isn't working:

1. **Verify Hook Files**: Ensure `.kiro/hooks/token-tracking*.kiro.hook` files exist
2. **Check Permissions**: Verify write permissions for `.kiro/` directory
3. **Review Logs**: Check Kiro IDE logs for hook execution errors
4. **Test Manually**: Use the programmatic API to verify core functionality works

Data Analysis
~~~~~~~~~~~~~

The CSV format makes it easy to analyze token usage patterns:

.. code-block:: python

   import pandas as pd
   import matplotlib.pyplot as plt
   
   # Load transaction data
   df = pd.read_csv('.kiro/token_transactions.csv')
   df['timestamp'] = pd.to_datetime(df['timestamp'])
   
   # Analyze usage patterns
   daily_usage = df.groupby(df['timestamp'].dt.date)['tokens_used'].sum()
   
   # Plot usage over time
   plt.figure(figsize=(12, 6))
   daily_usage.plot(kind='line')
   plt.title('Daily Token Usage')
   plt.xlabel('Date')
   plt.ylabel('Tokens Used')
   plt.show()
   
   # Analyze by workspace
   workspace_usage = df.groupby('workspace_folder')['tokens_used'].sum()
   print("Token usage by workspace:")
   print(workspace_usage.sort_values(ascending=False))

Error Handling
--------------

The token tracking system is designed to be resilient and non-intrusive:

Common Error Scenarios
~~~~~~~~~~~~~~~~~~~~~~

**File System Errors**
    When CSV file cannot be written due to permissions or disk space issues, the system logs the error and continues operation without interrupting the user workflow.

**Data Validation Errors**
    Invalid transaction data is logged and rejected, but the system continues processing other transactions.

**Concurrent Access Issues**
    File locking mechanisms prevent data corruption when multiple processes attempt to write simultaneously.

**Configuration Errors**
    Invalid configuration is detected at startup with clear error messages and fallback to default values.

Error Recovery
~~~~~~~~~~~~~~

The system implements several recovery mechanisms:

.. code-block:: python

   # Example error handling in the tracker
   try:
       self.csv_writer.write_transaction(transaction)
   except FileSystemError as e:
       self.error_handler.handle_file_error(e, transaction)
       # Transaction queued for retry
   except ValidationError as e:
       self.error_handler.log_validation_error(e, transaction)
       # Continue with next transaction
   except Exception as e:
       self.error_handler.handle_unexpected_error(e)
       # Graceful degradation

Comprehensive Troubleshooting Guide
-----------------------------------

Step-by-Step Troubleshooting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Problem: Token tracking not working at all**

1. **Check System Status**

   .. code-block:: bash

      # Verify hook files exist and are configured
      ls -la .kiro/hooks/token-tracking*.kiro.hook
      
      # Check if CSV file exists
      ls -la .kiro/token_transactions.csv
      
      # Verify directory permissions
      ls -ld .kiro/

2. **Verify Hook Configuration**

   .. code-block:: bash

      # Check hook is enabled
      cat .kiro/hooks/token-tracking.kiro.hook | grep '"enabled"'
      
      # Validate JSON syntax
      python -m json.tool .kiro/hooks/token-tracking.kiro.hook

3. **Test Core Functionality**

   .. code-block:: python

      from ai_hydra.token_tracker import TokenTracker
      from ai_hydra.token_tracker.models import TrackerConfig
      
      # Test basic functionality
      config = TrackerConfig()
      tracker = TokenTracker(config)
      
      # Record test transaction
      success = tracker.record_transaction(
          prompt_text="Test transaction",
          tokens_used=10,
          elapsed_time=1.0,
          context={"workspace_folder": "test"}
      )
      
      print(f"Test transaction successful: {success}")

4. **Check Logs**

   .. code-block:: bash

      # Check Kiro IDE logs for errors
      tail -f ~/.kiro/logs/kiro.log | grep -i token
      
      # Check system logs
      journalctl -u kiro --since "1 hour ago" | grep -i token

**Problem: CSV file not being created**

1. **Directory Permissions**

   .. code-block:: bash

      # Check .kiro directory exists and is writable
      mkdir -p .kiro
      touch .kiro/test_file && rm .kiro/test_file
      
      # Fix permissions if needed
      chmod 755 .kiro

2. **Manual CSV Creation**

   .. code-block:: python

      from ai_hydra.token_tracker.csv_writer import CSVWriter
      from ai_hydra.token_tracker.models import TrackerConfig
      
      config = TrackerConfig(csv_file_path=".kiro/token_transactions.csv")
      writer = CSVWriter(config)
      
      # Test CSV creation
      test_result = writer.test_csv_operations()
      print(f"CSV operations test: {test_result}")

**Problem: Missing transaction data**

1. **Verify Hook Triggers**

   .. code-block:: bash

      # Check recent Kiro IDE activity
      grep -i "agent.*execution" ~/.kiro/logs/kiro.log | tail -5

2. **Test Hook Integration**

   .. code-block:: python

      from ai_hydra.token_tracker.hook import TokenTrackingHook
      
      # Test hook functionality
      hook = TokenTrackingHook()
      test_result = hook.test_hook_functionality()
      
      print(f"Hook test results: {test_result}")

3. **Manual Transaction Recording**

   .. code-block:: python

      # Manually record a transaction to test the pipeline
      from ai_hydra.token_tracker import TokenTracker
      
      tracker = TokenTracker()
      success = tracker.record_transaction(
          prompt_text="Manual test transaction",
          tokens_used=25,
          elapsed_time=2.0,
          context={
              "workspace_folder": "test_workspace",
              "hook_trigger_type": "manual",
              "session_id": "test_session"
          }
      )
      
      if success:
          print("Manual recording works - issue is with automatic triggering")
      else:
          print("Core recording functionality has issues")

**Problem: Performance issues**

1. **Check File Size**

   .. code-block:: bash

      # Check CSV file size
      ls -lh .kiro/token_transactions.csv
      
      # Count transactions
      wc -l .kiro/token_transactions.csv

2. **Optimize Configuration**

   .. code-block:: python

      # Performance-optimized configuration
      config = TrackerConfig(
          max_prompt_length=500,        # Reduce prompt storage
          backup_enabled=False,         # Disable backups temporarily
          enable_validation=False,      # Skip validation for speed
          file_lock_timeout_seconds=1.0 # Reduce timeout
      )

3. **File Rotation**

   .. code-block:: bash

      # Rotate large CSV file
      mv .kiro/token_transactions.csv .kiro/token_transactions_$(date +%Y%m%d).csv
      
      # System will create new file automatically

**Problem: Data corruption or invalid CSV**

1. **Validate CSV Integrity**

   .. code-block:: python

      from ai_hydra.token_tracker import TokenTracker
      
      tracker = TokenTracker()
      integrity_results = tracker.validate_csv_integrity()
      
      print(f"File exists: {integrity_results['file_exists']}")
      print(f"Valid headers: {integrity_results['valid_headers']}")
      print(f"Valid rows: {integrity_results['valid_rows']}")
      print(f"Total rows: {integrity_results['total_rows']}")
      
      if integrity_results['issues']:
          print(f"Issues found: {integrity_results['issues']}")

2. **Repair CSV File**

   .. code-block:: python

      # Attempt to repair CSV file
      import pandas as pd
      
      try:
          # Read with error handling
          df = pd.read_csv('.kiro/token_transactions.csv', error_bad_lines=False)
          
          # Save clean version
          df.to_csv('.kiro/token_transactions_clean.csv', index=False)
          print("Clean CSV file created")
          
      except Exception as e:
          print(f"CSV repair failed: {e}")

3. **Restore from Backup**

   .. code-block:: bash

      # Find backup files
      ls -la .kiro/token_transactions_backup_*.csv
      
      # Restore from most recent backup
      cp .kiro/token_transactions_backup_$(ls .kiro/token_transactions_backup_*.csv | tail -1 | cut -d'_' -f3-).csv .kiro/token_transactions.csv

**Problem: Unicode or special character issues**

1. **Test Unicode Support**

   .. code-block:: python

      from ai_hydra.token_tracker import TokenTracker
      
      tracker = TokenTracker()
      
      # Test Unicode handling
      test_strings = [
          "Hello 世界",
          "Émojis: 😀🎉🚀",
          "Quotes: \"Hello\" and 'World'",
          "Newlines:\nLine 1\nLine 2",
          "Tabs:\tTabbed\tContent"
      ]
      
      unicode_results = tracker.test_unicode_compatibility(test_strings)
      print(f"Unicode support: {unicode_results['unicode_support_verified']}")
      
      if not unicode_results['unicode_support_verified']:
          print(f"Failed tests: {unicode_results['failed_tests']}")

2. **Fix Encoding Issues**

   .. code-block:: python

      # Re-encode CSV file with proper UTF-8
      import pandas as pd
      
      # Read with explicit encoding
      df = pd.read_csv('.kiro/token_transactions.csv', encoding='utf-8')
      
      # Save with UTF-8 BOM for Excel compatibility
      df.to_csv('.kiro/token_transactions_utf8.csv', index=False, encoding='utf-8-sig')

Diagnostic Tools and Scripts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**System Health Check Script**

Create a comprehensive health check script:

.. code-block:: python

   #!/usr/bin/env python3
   """Token Tracker System Health Check"""
   
   import os
   import json
   from pathlib import Path
   from ai_hydra.token_tracker import TokenTracker
   from ai_hydra.token_tracker.models import TrackerConfig
   
   def health_check():
       """Perform comprehensive system health check."""
       results = {
           'overall_status': 'UNKNOWN',
           'checks': {}
       }
       
       # Check 1: Hook files exist
       hook_files = [
           '.kiro/hooks/token-tracking.kiro.hook',
           '.kiro/hooks/token-tracking-start.kiro.hook'
       ]
       
       for hook_file in hook_files:
           exists = Path(hook_file).exists()
           results['checks'][f'hook_file_{Path(hook_file).stem}'] = {
               'status': 'PASS' if exists else 'FAIL',
               'message': f"Hook file {'exists' if exists else 'missing'}: {hook_file}"
           }
       
       # Check 2: CSV file accessibility
       csv_path = '.kiro/token_transactions.csv'
       csv_exists = Path(csv_path).exists()
       csv_writable = os.access('.kiro', os.W_OK) if Path('.kiro').exists() else False
       
       results['checks']['csv_file'] = {
           'status': 'PASS' if csv_exists and csv_writable else 'WARN' if csv_writable else 'FAIL',
           'message': f"CSV file: exists={csv_exists}, writable={csv_writable}"
       }
       
       # Check 3: Core functionality
       try:
           config = TrackerConfig()
           tracker = TokenTracker(config)
           
           # Test transaction recording
           success = tracker.record_transaction(
               prompt_text="Health check test",
               tokens_used=1,
               elapsed_time=0.1,
               context={'workspace_folder': 'health_check'}
           )
           
           results['checks']['core_functionality'] = {
               'status': 'PASS' if success else 'FAIL',
               'message': f"Core functionality test: {'passed' if success else 'failed'}"
           }
           
       except Exception as e:
           results['checks']['core_functionality'] = {
               'status': 'FAIL',
               'message': f"Core functionality error: {str(e)}"
           }
       
       # Check 4: Configuration validation
       try:
           config = TrackerConfig()
           issues = config.validate()
           
           results['checks']['configuration'] = {
               'status': 'PASS' if not issues else 'WARN',
               'message': f"Configuration: {len(issues)} issues found" if issues else "Configuration valid"
           }
           
       except Exception as e:
           results['checks']['configuration'] = {
               'status': 'FAIL',
               'message': f"Configuration error: {str(e)}"
           }
       
       # Determine overall status
       statuses = [check['status'] for check in results['checks'].values()]
       if 'FAIL' in statuses:
           results['overall_status'] = 'FAIL'
       elif 'WARN' in statuses:
           results['overall_status'] = 'WARN'
       else:
           results['overall_status'] = 'PASS'
       
       return results
   
   if __name__ == '__main__':
       results = health_check()
       
       print(f"Token Tracker Health Check - Overall Status: {results['overall_status']}")
       print("=" * 50)
       
       for check_name, check_result in results['checks'].items():
           status_icon = "✓" if check_result['status'] == 'PASS' else "⚠" if check_result['status'] == 'WARN' else "✗"
           print(f"{status_icon} {check_name}: {check_result['message']}")
       
       print("=" * 50)
       
       if results['overall_status'] == 'FAIL':
           print("❌ System has critical issues that need attention")
           exit(1)
       elif results['overall_status'] == 'WARN':
           print("⚠️  System is working but has warnings")
           exit(0)
       else:
           print("✅ System is healthy")
           exit(0)

Save this as ``health_check.py`` and run it regularly:

.. code-block:: bash

   python health_check.py

**Performance Monitoring Script**

.. code-block:: python

   #!/usr/bin/env python3
   """Token Tracker Performance Monitor"""
   
   import time
   import psutil
   from ai_hydra.token_tracker import TokenTracker
   from ai_hydra.token_tracker.models import TrackerConfig
   
   def performance_test(num_transactions=100):
       """Test performance with multiple transactions."""
       config = TrackerConfig()
       tracker = TokenTracker(config)
       
       # Monitor system resources
       process = psutil.Process()
       initial_memory = process.memory_info().rss / 1024 / 1024  # MB
       
       start_time = time.time()
       
       # Record multiple transactions
       for i in range(num_transactions):
           success = tracker.record_transaction(
               prompt_text=f"Performance test transaction {i}",
               tokens_used=50 + (i % 100),
               elapsed_time=1.0 + (i % 5),
               context={'workspace_folder': 'performance_test'}
           )
           
           if not success:
               print(f"Transaction {i} failed")
       
       end_time = time.time()
       final_memory = process.memory_info().rss / 1024 / 1024  # MB
       
       # Calculate metrics
       total_time = end_time - start_time
       transactions_per_second = num_transactions / total_time
       memory_increase = final_memory - initial_memory
       
       print(f"Performance Test Results:")
       print(f"  Transactions: {num_transactions}")
       print(f"  Total time: {total_time:.2f} seconds")
       print(f"  Transactions/second: {transactions_per_second:.2f}")
       print(f"  Memory increase: {memory_increase:.2f} MB")
       print(f"  Average time per transaction: {total_time/num_transactions*1000:.2f} ms")
   
   if __name__ == '__main__':
       performance_test()

Common Issues and Solutions
~~~~~~~~~~~~~

**CSV File Not Created**
    - Check file permissions in the target directory
    - Verify the hook is properly configured and enabled
    - Check Kiro IDE logs for error messages

**Missing Transaction Data**
    - Verify the agent hook is triggering correctly
    - Check that token usage information is available in the execution context
    - Review error logs for data collection issues

**File Patterns Not Preserved**
    - Use the debug script to test metadata collection: ``python debug_file_patterns.py``
    - Check that hook context includes file_patterns field
    - Verify MetadataCollector is handling context correctly

**Performance Issues**
    - Consider reducing the maximum prompt length
    - Enable compression for large CSV files
    - Implement file rotation for long-running systems

**Data Corruption**
    - Check for concurrent access issues
    - Verify CSV file integrity using validation tools
    - Restore from backup if available

Debug and Validation Tools
~~~~~~~~~~~~~~~~~~~~~~~~~~

**File Patterns Debug Script**

The project includes a specialized debug script for testing file patterns preservation:

.. code-block:: bash

    # Test file patterns preservation through the entire metadata collection chain
    python debug_file_patterns.py

This script tests three levels of the system:

1. **MetadataCollector.get_hook_context()**: Direct hook context processing
2. **MetadataCollector.collect_execution_metadata()**: Full metadata collection
3. **TokenTracker integration**: End-to-end transaction recording

**Expected output when working correctly:**

.. code-block:: text

    Testing file patterns preservation...
    Input context: {'trigger_type': 'agentExecutionCompleted', 'hook_name': 'test-hook', 'file_patterns': ['*.py', '*.md', '*.txt']}

    1. Testing MetadataCollector.get_hook_context()...
    ✓ file_patterns found: ['*.py', '*.md', '*.txt']
    ✓ file_patterns match input

    2. Testing MetadataCollector.collect_execution_metadata()...
    ✓ file_patterns found in metadata: ['*.py', '*.md', '*.txt']
    ✓ file_patterns match input in full metadata

    3. Testing TokenTracker integration...
    ✓ file_patterns preserved in transaction

    ✓ All tests passed!

**Using the debug script for troubleshooting:**

.. code-block:: bash

    # Run debug script and capture output
    python debug_file_patterns.py > debug_output.txt 2>&1
    
    # Check for specific issues
    if python debug_file_patterns.py; then
        echo "File patterns preservation working correctly"
    else
        echo "File patterns preservation has issues - check debug output"
    fi

**Integration with testing workflow:**

.. code-block:: bash

    # Include in test suite
    pytest tests/ && python debug_file_patterns.py

Diagnostic Commands
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Check CSV file integrity
   python -c "
   import csv
   with open('.kiro/token_transactions.csv', 'r') as f:
       reader = csv.DictReader(f)
       count = sum(1 for row in reader)
       print(f'Valid rows: {count}')
   "
   
   # Validate recent transactions
   tail -n 10 .kiro/token_transactions.csv
   
   # Check file permissions
   ls -la .kiro/token_transactions.csv

Frequently Asked Questions (FAQ)
---------------------------------

**Q: How much overhead does token tracking add to AI interactions?**

A: Token tracking adds minimal overhead (typically < 50ms per interaction). The system is designed to be non-intrusive and uses asynchronous processing where possible.

**Q: Can I disable token tracking temporarily?**

A: Yes, you can disable tracking in several ways:

.. code-block:: python

   # Method 1: Update configuration
   from ai_hydra.token_tracker.hook import TokenTrackingHook
   hook = TokenTrackingHook.get_instance()
   hook.disable()
   
   # Method 2: Edit hook file
   # Set "enabled": false in .kiro/hooks/token-tracking.kiro.hook
   
   # Method 3: Environment variable
   export KIRO_TOKEN_TRACKER_ENABLED=false

**Q: How much disk space will token tracking use?**

A: Disk usage depends on your activity level:

- Light usage (10 interactions/day): ~1MB/month
- Moderate usage (50 interactions/day): ~5MB/month  
- Heavy usage (200 interactions/day): ~20MB/month

Enable compression and set retention policies to manage disk usage.

**Q: Is my prompt data secure?**

A: Yes, prompt data is stored locally in CSV files with configurable retention. You can:

- Limit prompt text length (``max_prompt_length`` setting)
- Set short retention periods (``retention_days`` setting)
- Disable prompt storage entirely (set ``max_prompt_length=0``)

**Q: Can I export data to other tools?**

A: Yes, the CSV format is compatible with Excel, Google Sheets, Tableau, and other analytics tools. You can also use the export API:

.. code-block:: python

   tracker.export_data("report.xlsx", format="excel", filters={"workspace": "my_project"})

**Q: What happens if the CSV file gets corrupted?**

A: The system includes several recovery mechanisms:

- Automatic backups (if enabled)
- CSV integrity validation
- Repair utilities for common corruption issues
- Graceful handling of partial corruption

**Q: Can I track tokens across multiple Kiro IDE instances?**

A: Each Kiro IDE instance maintains its own CSV file. To consolidate data:

.. code-block:: python

   import pandas as pd
   
   # Combine multiple CSV files
   files = ['instance1.csv', 'instance2.csv', 'instance3.csv']
   combined_df = pd.concat([pd.read_csv(f) for f in files])
   combined_df.to_csv('combined_tokens.csv', index=False)

**Q: How do I set up automated reporting?**

A: Create a scheduled script:

.. code-block:: bash

   #!/bin/bash
   # daily_token_report.sh
   
   python3 << EOF
   import pandas as pd
   from datetime import datetime, timedelta
   
   # Generate daily report
   df = pd.read_csv('.kiro/token_transactions.csv')
   df['timestamp'] = pd.to_datetime(df['timestamp'])
   
   yesterday = datetime.now() - timedelta(days=1)
   daily_df = df[df['timestamp'].dt.date == yesterday.date()]
   
   if len(daily_df) > 0:
       total_tokens = daily_df['tokens_used'].sum()
       total_interactions = len(daily_df)
       avg_tokens = daily_df['tokens_used'].mean()
       
       print(f"Daily Token Report - {yesterday.date()}")
       print(f"Total tokens: {total_tokens}")
       print(f"Total interactions: {total_interactions}")
       print(f"Average tokens per interaction: {avg_tokens:.1f}")
   else:
       print(f"No token usage recorded for {yesterday.date()}")
   EOF

Then schedule it with cron:

.. code-block:: bash

   # Add to crontab (crontab -e)
   0 9 * * * /path/to/daily_token_report.sh

**Q: Can I integrate with cost tracking systems?**

A: Yes, you can calculate costs and integrate with expense tracking:

.. code-block:: python

   # Cost calculation example
   def calculate_costs(df, model_costs):
       """Calculate costs based on token usage."""
       df['cost'] = df['tokens_used'] * model_costs.get('default', 0.00003)
       
       return {
           'total_cost': df['cost'].sum(),
           'daily_average': df.groupby(df['timestamp'].dt.date)['cost'].sum().mean(),
           'monthly_projection': df['cost'].sum() / len(df['timestamp'].dt.date.unique()) * 30
       }

**Q: How do I troubleshoot missing transactions?**

A: Follow this checklist:

1. Check if hooks are enabled: ``cat .kiro/hooks/token-tracking.kiro.hook | grep enabled``
2. Verify CSV file permissions: ``ls -la .kiro/token_transactions.csv``
3. Check Kiro IDE logs: ``tail -f ~/.kiro/logs/kiro.log | grep -i token``
4. Test manual recording: Use the programmatic API to verify core functionality
5. Run health check: Use the provided health check script

**Q: Can I customize the CSV format?**

A: The CSV format is standardized for compatibility, but you can:

- Export to custom formats using the export API
- Add custom metadata through the context parameter
- Create derived CSV files with additional calculated fields

**Q: How do I handle large CSV files?**

A: For large files (>100MB):

1. Enable compression: ``compression_enabled=True``
2. Implement rotation: Move old data to archive files
3. Use streaming processing: Process data in chunks with pandas
4. Set retention policies: Automatically delete old data

.. code-block:: python

   # Process large CSV in chunks
   chunk_size = 10000
   for chunk in pd.read_csv('large_file.csv', chunksize=chunk_size):
       # Process each chunk
       process_chunk(chunk)

Best Practices and Tips
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Enable Backups**: Always enable automatic backups for production use
* **Set Retention Policies**: Configure appropriate data retention to manage disk space
* **Monitor File Size**: Implement file rotation for long-running systems
* **Secure Sensitive Data**: Be mindful of prompt content that may contain sensitive information

Development Best Practices
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Test Error Scenarios**: Verify error handling works correctly
* **Validate Data Integrity**: Regularly check CSV file integrity
* **Monitor Performance**: Track the impact of token tracking on system performance
* **Document Configuration**: Maintain clear documentation of hook configurations

Integration Best Practices
~~~~~~~~~~~~~~~~~~~~~~~~~~~

* **Graceful Degradation**: Ensure token tracking failures don't impact normal workflows
* **Minimal Overhead**: Keep tracking overhead as low as possible
* **Clear Logging**: Provide informative log messages for troubleshooting
* **Version Compatibility**: Ensure compatibility with different Kiro IDE versions

API Reference
-------------

The current implementation provides the following API components:

Core Classes
~~~~~~~~~~~~

**TokenTracker**
    Main service class for recording and retrieving token transactions.
    
    Key methods:
    
    * ``record_transaction(prompt_text, tokens_used, elapsed_time, context=None) -> bool``
    * ``get_transaction_history(filters=None, limit=None) -> List[TokenTransaction]``
    * ``get_statistics() -> Dict[str, Any]``
    * ``validate_csv_integrity() -> Dict[str, Any]``
    * ``test_unicode_compatibility() -> Dict[str, Any]``
    * ``create_backup() -> Optional[Path]``
    * ``export_data(output_path, format='csv', filters=None) -> bool``

**TokenTransaction**
    Immutable data class representing a single token usage event.
    
    Key methods:
    
    * ``create_new(prompt_text, tokens_used, elapsed_time, workspace_folder, hook_trigger_type, **kwargs) -> TokenTransaction``
    * ``to_csv_row() -> List[str]``
    * ``from_csv_row(row: List[str]) -> TokenTransaction``
    * ``get_summary() -> Dict[str, Any]``

**TrackerConfig**
    Configuration class with validation and environment-specific presets.
    
    Key methods:
    
    * ``create_default() -> TrackerConfig``
    * ``create_for_testing() -> TrackerConfig``
    * ``create_for_production() -> TrackerConfig``
    * ``validate() -> List[str]``
    * ``to_dict() -> Dict[str, Any]``
    * ``from_dict(config_dict: Dict[str, Any]) -> TrackerConfig``

**CSVWriter**
    Thread-safe CSV operations with file locking and Unicode support.
    
    Key methods:
    
    * ``write_transaction(transaction: TokenTransaction) -> bool``
    * ``write_transactions_batch(transactions: List[TokenTransaction]) -> int``
    * ``read_transactions(limit: Optional[int] = None) -> List[TokenTransaction]``
    * ``validate_csv_integrity() -> Dict[str, Any]``
    * ``validate_unicode_handling(test_strings: List[str]) -> Dict[str, Any]``
    * ``create_backup() -> Optional[Path]``

**ErrorHandler**
    Comprehensive error management with recovery mechanisms.
    
    Key methods:
    
    * ``handle_csv_write_error(error, file_path, transaction)``
    * ``handle_csv_read_error(error, file_path)``
    * ``handle_validation_error(issues, transaction)``
    * ``get_error_statistics() -> Dict[str, Any]``

**TokenTrackingHook**
    Agent hook for automatic token tracking integration with Kiro IDE.
    
    Key methods:
    
    * ``on_agent_execution_start(context: Dict[str, Any]) -> None``
    * ``on_agent_execution_complete(context: Dict[str, Any], result: Dict[str, Any]) -> None``
    * ``update_configuration(new_config: TrackerConfig) -> bool``
    * ``get_configuration() -> Dict[str, Any]``
    * ``apply_configuration_changes(changes: Dict[str, Any]) -> bool``
    * ``save_configuration_to_file(config_file_path: Optional[Path] = None) -> bool``
    * ``reload_configuration_from_file(config_file_path: Optional[Path] = None) -> bool``
    * ``reset_to_default_configuration() -> bool``
    * ``validate_configuration() -> Dict[str, Any]``
    * ``get_configuration_schema() -> Dict[str, Any]``
    * ``enable() -> None``
    * ``disable() -> None``
    * ``get_statistics() -> Dict[str, Any]``
    * ``test_hook_functionality() -> Dict[str, Any]``
    * ``cleanup() -> None``

Property-Based Testing
~~~~~~~~~~~~~~~~~~~~~~

The system includes comprehensive property-based tests that validate:

* **Property 1: CSV Transaction Persistence** - All valid transactions can be stored and retrieved
* **Property 2: Data Append Safety** - Concurrent append operations preserve data integrity  
* **Property 4: Hook-Tracker Integration** - Agent hook automatically triggers token tracking and records complete metadata
* **Property 5: Error Recovery Resilience** - System recovers gracefully from various error conditions
* **Property 6: Configuration State Management** - Hook maintains consistent configuration state and supports runtime updates
* **Property 7: Special Character Handling** - Unicode and CSV special characters are handled correctly
* **Property 8: Data Validation Integrity** - All data validation rules are enforced consistently

**Configuration Management Property Testing:**

The configuration management system is validated through comprehensive property-based tests that ensure:

* **State Consistency**: For any valid configuration parameters, the hook maintains consistent internal state
* **Runtime Updates**: Configuration changes are applied correctly without requiring system restart
* **Enable/Disable Cycles**: Configuration values are preserved across enable/disable operations
* **Partial Updates**: Only specified configuration parameters are changed while others remain unchanged
* **File Persistence**: Configuration can be saved to and loaded from JSON files correctly
* **Error Handling**: Invalid configurations are rejected gracefully without corrupting system state
* **Validation**: Configuration validation provides meaningful error messages and recovery guidance

Example property test for configuration management:

.. code-block:: python

    @given(
        enabled=st.booleans(),
        max_prompt_length=st.integers(min_value=10, max_value=5000),
        backup_enabled=st.booleans(),
        log_level=st.sampled_from(["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]),
    )
    @settings(max_examples=50, deadline=3000)
    def test_configuration_state_consistency_property(
        self, enabled, max_prompt_length, backup_enabled, log_level
    ):
        """
        **Property 6: Configuration State Management**
        **Validates: Requirements 2.5**
        
        For any valid configuration parameters, the hook should:
        1. Accept the configuration update
        2. Maintain consistent internal state
        3. Reflect changes in subsequent operations
        4. Preserve configuration across enable/disable cycles
        """
        # Property test implementation validates all aspects of configuration management

This property-based approach ensures that configuration management works correctly across all possible valid input combinations, providing confidence in the system's reliability and robustness.

For complete API documentation with detailed method signatures and examples, see the auto-generated API reference.

See Also
--------

* :doc:`requirements` - Complete requirements specification
* :doc:`design` - System design and architecture details
* :doc:`testing` - Testing strategies and validation
* :doc:`troubleshooting` - General troubleshooting guide