Token Tracking System
====================

The Token Tracking System provides comprehensive monitoring and analysis of AI token usage within the Kiro IDE environment. This system automatically captures token consumption data from AI interactions and stores it in a structured format for analysis and optimization.

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

* **Automatic Token Tracking**: Seamlessly captures token usage without manual intervention
* **Comprehensive Metadata**: Records detailed context including workspace, hook triggers, and execution IDs
* **CSV-Based Storage**: Standard format compatible with spreadsheet and analysis tools
* **Concurrent Access Safety**: Thread-safe operations for multi-user environments
* **Error Resilience**: Graceful error handling that doesn't interrupt normal workflows
* **Configurable Operation**: Enable/disable tracking and customize behavior

Architecture
------------

Token Transaction Flow
~~~~~~~~~~~~~~~~~~~~~~~

.. mermaid::

   graph TD
       A[Kiro IDE Event] --> B[Agent Hook Trigger]
       B --> C[Metadata Collector]
       C --> D[Token Tracker Service]
       D --> E[CSV Writer]
       E --> F[Transaction History]
       
       G[Error Handler] --> D
       G --> E
       
       H[Configuration] --> B
       H --> D

Component Responsibilities
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Token Tracker Service**
    Core service responsible for processing token transactions, validating data, and coordinating with other components.

**Agent Hook Integration**
    Monitors Kiro IDE events and automatically triggers token tracking when AI interactions occur.

**CSV Writer**
    Thread-safe component that handles writing transaction data to CSV files with proper locking mechanisms.

**Metadata Collector**
    Gathers contextual information from the Kiro IDE environment including workspace details, hook context, and execution metadata.

**Error Handler**
    Manages system failures and provides fallback mechanisms to ensure reliable operation.

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

Configuration
-------------

Token Tracker Configuration
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The token tracking system can be configured through the ``TrackerConfig`` class:

.. code-block:: python

   from ai_hydra.token_tracker.models import TrackerConfig
   
   config = TrackerConfig(
       enabled=True,                           # Enable/disable tracking
       csv_file_path=".kiro/token_transactions.csv",  # CSV file location
       max_prompt_length=1000,                 # Maximum prompt text length
       backup_enabled=True,                    # Enable automatic backups
       backup_interval_hours=24,               # Backup frequency
       compression_enabled=False,              # Compress archived files
       retention_days=365                      # Data retention period
   )

Agent Hook Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

The agent hook can be configured through a ``.kiro.hook`` file:

.. code-block:: json

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

Usage
-----

Basic Usage
~~~~~~~~~~~

The token tracking system operates automatically once configured. Here's how to set it up:

1. **Install the Token Tracker**:

   .. code-block:: bash

      # The token tracker is included with ai_hydra
      pip install -e .

2. **Configure the Hook**:

   Create a ``.kiro/hooks/token-tracking.kiro.hook`` file with the appropriate configuration.

3. **Verify Operation**:

   Check that the CSV file is being created and populated:

   .. code-block:: bash

      ls -la .kiro/token_transactions.csv
      head .kiro/token_transactions.csv

Programmatic Usage
~~~~~~~~~~~~~~~~~~

You can also use the token tracker programmatically:

.. code-block:: python

   from ai_hydra.token_tracker import TokenTracker
   from ai_hydra.token_tracker.models import TokenTransaction, TrackerConfig
   from datetime import datetime
   
   # Initialize tracker
   config = TrackerConfig(csv_file_path="my_tokens.csv")
   tracker = TokenTracker(config)
   
   # Record a transaction
   transaction = TokenTransaction(
       timestamp=datetime.now(),
       prompt_text="Analyze this code",
       tokens_used=150,
       elapsed_time=1.5,
       session_id="session_123",
       workspace_folder="my_project",
       hook_trigger_type="manual",
       agent_execution_id="exec_456",
       hook_name="manual-tracking"
   )
   
   success = tracker.record_transaction(transaction)
   if success:
       print("Transaction recorded successfully")

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

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**CSV File Not Created**
    - Check file permissions in the target directory
    - Verify the hook is properly configured and enabled
    - Check Kiro IDE logs for error messages

**Missing Transaction Data**
    - Verify the agent hook is triggering correctly
    - Check that token usage information is available in the execution context
    - Review error logs for data collection issues

**Performance Issues**
    - Consider reducing the maximum prompt length
    - Enable compression for large CSV files
    - Implement file rotation for long-running systems

**Data Corruption**
    - Check for concurrent access issues
    - Verify CSV file integrity using validation tools
    - Restore from backup if available

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

Best Practices
--------------

Configuration Best Practices
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

For detailed API documentation, see the :doc:`api_reference` section covering:

* ``TokenTracker`` class methods and properties
* ``TokenTransaction`` data model
* ``TrackerConfig`` configuration options
* ``MetadataCollector`` interface
* ``ErrorHandler`` methods
* ``CSVWriter`` functionality

See Also
--------

* :doc:`requirements` - Complete requirements specification
* :doc:`design` - System design and architecture details
* :doc:`testing` - Testing strategies and validation
* :doc:`troubleshooting` - General troubleshooting guide