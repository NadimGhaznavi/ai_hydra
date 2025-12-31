Token Tracking System
====================

The Token Tracking System provides comprehensive monitoring and analysis of AI token usage within the Kiro IDE environment. This system automatically captures token consumption data from AI interactions and stores it in a structured format for analysis and optimization.

.. note::
   **Implementation Status**: Core functionality is complete and operational. The system includes:
   
   * ✅ Core data models (TokenTransaction, TrackerConfig)
   * ✅ Thread-safe CSV operations with file locking
   * ✅ Token tracker service with validation and error handling
   * ✅ Special character and Unicode handling
   * ✅ Comprehensive property-based testing
   * 🔄 Metadata collection system (in progress)
   * 🔄 Agent hook integration (in progress)

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

**Planned Features:**

* **Automatic Token Tracking**: Seamless capture via Kiro IDE agent hooks (in development)
* **Comprehensive Metadata**: Detailed context from Kiro IDE environment (in development)
* **Agent Hook Integration**: Automatic triggering system (in development)
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

Configuration
-------------

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

Agent Hook Configuration (Coming Soon)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The agent hook will be configured through a ``.kiro.hook`` file:

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

Planned Usage (Full System)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the agent hook integration is complete, the system will operate automatically:

1. **Install and Configure**:

   .. code-block:: bash

      # The token tracker is included with ai_hydra
      pip install -e .

2. **Configure the Hook** (Coming Soon):

   Create a ``.kiro/hooks/token-tracking.kiro.hook`` file with the appropriate configuration.

3. **Automatic Operation**:

   The system will automatically track token usage from all AI interactions in the Kiro IDE.

4. **Monitor and Analyze**:

   .. code-block:: bash

      # Check the CSV file
      ls -la .kiro/token_transactions.csv
      head .kiro/token_transactions.csv

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

Property-Based Testing
~~~~~~~~~~~~~~~~~~~~~~

The system includes comprehensive property-based tests that validate:

* **Property 1: CSV Transaction Persistence** - All valid transactions can be stored and retrieved
* **Property 2: Data Append Safety** - Concurrent append operations preserve data integrity  
* **Property 5: Error Recovery Resilience** - System recovers gracefully from various error conditions
* **Property 7: Special Character Handling** - Unicode and CSV special characters are handled correctly
* **Property 8: Data Validation Integrity** - All data validation rules are enforced consistently

For complete API documentation with detailed method signatures and examples, see the auto-generated API reference.

See Also
--------

* :doc:`requirements` - Complete requirements specification
* :doc:`design` - System design and architecture details
* :doc:`testing` - Testing strategies and validation
* :doc:`troubleshooting` - General troubleshooting guide