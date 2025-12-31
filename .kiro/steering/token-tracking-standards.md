# Token Tracking Standards

## Overview

This document defines the standards and practices for token tracking within the AI Hydra project using the Kiro IDE. These standards ensure consistent, reliable, and maintainable token usage monitoring across all AI interactions.

## Token Tracking Philosophy

### Core Principles

- **Automatic and Transparent**: Token tracking should happen automatically without user intervention
- **Non-Intrusive**: Tracking must not interfere with normal development workflows
- **Comprehensive**: Capture all relevant metadata for analysis and optimization
- **Reliable**: System must be resilient to errors and continue operation
- **Privacy-Aware**: Handle sensitive information in prompts appropriately

### Data Quality Standards

- **Accuracy**: All token counts and timing information must be precise
- **Completeness**: Every AI interaction should be tracked with full metadata
- **Consistency**: Data format and structure must remain stable over time
- **Integrity**: Concurrent access must not corrupt data
- **Traceability**: Each transaction must be uniquely identifiable

## CSV Format Specifications

### Standard Column Structure

The CSV file MUST use the following column structure with exact header names:

```csv
timestamp,prompt_text,tokens_used,elapsed_time,session_id,workspace_folder,hook_trigger_type,agent_execution_id,file_patterns,hook_name,error_occurred,error_message
```

### Column Specifications

| Column | Type | Required | Description | Format/Constraints |
|--------|------|----------|-------------|-------------------|
| `timestamp` | datetime | Yes | ISO 8601 timestamp of transaction | `YYYY-MM-DDTHH:MM:SS.sssZ` |
| `prompt_text` | string | Yes | The prompt sent to AI (truncated if needed) | Max 1000 chars, CSV-escaped |
| `tokens_used` | integer | Yes | Number of tokens consumed | Positive integer |
| `elapsed_time` | float | Yes | Time taken for interaction in seconds | Positive float, 2 decimal places |
| `session_id` | string | Yes | Unique Kiro IDE session identifier | Alphanumeric, max 50 chars |
| `workspace_folder` | string | Yes | Name of active workspace folder | Valid folder name |
| `hook_trigger_type` | string | Yes | Type of event that triggered tracking | Enum: see Hook Trigger Types |
| `agent_execution_id` | string | Yes | Unique agent execution identifier | Alphanumeric, max 50 chars |
| `file_patterns` | string | No | File patterns that triggered interaction | Semicolon-separated patterns |
| `hook_name` | string | Yes | Name of hook that initiated tracking | Valid hook name |
| `error_occurred` | boolean | Yes | Whether an error occurred | `true` or `false` |
| `error_message` | string | No | Error message if error occurred | CSV-escaped, max 500 chars |

### Hook Trigger Types

Standard hook trigger types that MUST be used:

- `agentExecutionCompleted`: When an AI agent execution completes
- `agentExecutionStarted`: When an AI agent execution begins
- `fileEdited`: When a file is edited and triggers AI interaction
- `userMessage`: When user sends a message to AI
- `manual`: Manual token tracking invocation
- `scheduled`: Scheduled or automated AI interaction

### Data Validation Rules

**Timestamp Validation**
- Must be valid ISO 8601 format
- Must not be in the future (allow 5-minute tolerance)
- Must not be older than 1 year

**Token Count Validation**
- Must be positive integer
- Must be reasonable (1 to 1,000,000 tokens)
- Must not be zero unless error occurred

**Elapsed Time Validation**
- Must be positive float
- Must be reasonable (0.001 to 3600 seconds)
- Must correlate with token count (more tokens = more time)

**Text Field Validation**
- Must be properly CSV-escaped
- Must handle Unicode characters correctly
- Must truncate long content appropriately
- Must preserve essential information when truncating

## Agent Hook Configuration Guidelines

### Hook File Structure

Agent hooks MUST be configured using `.kiro.hook` files with the following structure:

```json
{
  "name": "token-tracker-hook",
  "version": "1.0.0",
  "description": "Automatic token usage tracking for AI interactions",
  "triggers": [
    "agentExecutionCompleted",
    "agentExecutionStarted"
  ],
  "enabled": true,
  "configuration": {
    "track_tokens": true,
    "include_metadata": true,
    "error_handling": "graceful",
    "max_prompt_length": 1000,
    "csv_file_path": ".kiro/token_transactions.csv"
  },
  "error_handling": {
    "on_file_error": "log_and_continue",
    "on_validation_error": "log_and_skip",
    "on_unexpected_error": "log_and_continue"
  }
}
```

### Hook Configuration Standards

**Naming Convention**
- Hook names MUST use kebab-case: `token-tracker-hook`
- Hook files MUST use `.kiro.hook` extension
- Hook files MUST be placed in `.kiro/hooks/` directory

**Trigger Configuration**
- MUST include at least `agentExecutionCompleted` trigger
- MAY include additional triggers based on use case
- MUST NOT include triggers that would cause excessive overhead

**Error Handling Configuration**
- MUST specify error handling strategy for each error type
- MUST use `graceful` error handling to avoid workflow interruption
- MUST log errors for troubleshooting

### Hook Implementation Standards

**Performance Requirements**
- Hook execution MUST complete within 5 seconds
- Hook MUST NOT block the main UI thread
- Hook MUST handle concurrent executions safely

**Resource Management**
- Hook MUST clean up resources after execution
- Hook MUST handle file locking appropriately
- Hook MUST limit memory usage to reasonable bounds

**Error Resilience**
- Hook MUST continue operation after non-critical errors
- Hook MUST provide meaningful error messages
- Hook MUST implement retry logic for transient failures

## Configuration Management

### Tracker Configuration Standards

The `TrackerConfig` class MUST be used for all configuration:

```python
from ai_hydra.token_tracker.models import TrackerConfig

# Standard configuration
config = TrackerConfig(
    enabled=True,                           # Enable/disable tracking
    csv_file_path=".kiro/token_transactions.csv",  # CSV file location
    max_prompt_length=1000,                 # Maximum prompt text length
    backup_enabled=True,                    # Enable automatic backups
    backup_interval_hours=24,               # Backup frequency
    compression_enabled=False,              # Compress archived files
    retention_days=365                      # Data retention period
)
```

### Configuration Validation

**File Path Validation**
- CSV file path MUST be writable
- Directory MUST exist or be creatable
- Path MUST be relative to workspace root

**Parameter Validation**
- All numeric parameters MUST be positive
- String parameters MUST not be empty
- Boolean parameters MUST be explicitly set

**Environment-Specific Configuration**
- Development: Enable verbose logging, shorter retention
- Production: Enable backups, longer retention, compression
- Testing: Use temporary files, disable backups

## Data Privacy and Security

### Sensitive Information Handling

**Prompt Text Sanitization**
- MUST truncate prompts longer than configured maximum
- MAY implement keyword-based sanitization for sensitive data
- MUST preserve enough context for analysis while protecting privacy

**Access Control**
- CSV files MUST have appropriate file permissions (600 or 644)
- Backup files MUST inherit same permissions as original
- Log files MUST not contain sensitive prompt content

**Data Retention**
- MUST implement configurable data retention policies
- MUST provide secure deletion of expired data
- MUST support data export for compliance requirements

### Compliance Considerations

**GDPR Compliance**
- Support data subject access requests
- Implement right to erasure (data deletion)
- Maintain data processing records

**Enterprise Security**
- Support encryption at rest for sensitive environments
- Implement audit logging for data access
- Support integration with enterprise security tools

## Error Handling Standards

### Error Categories

**File System Errors**
- Disk space exhaustion
- Permission denied
- File corruption
- Network storage issues

**Data Validation Errors**
- Invalid token counts
- Malformed timestamps
- Missing required fields
- Data type mismatches

**Integration Errors**
- Hook execution failures
- Metadata collection failures
- Kiro IDE communication issues
- Concurrent access conflicts

### Error Handling Strategies

**Graceful Degradation**
```python
try:
    tracker.record_transaction(transaction)
except FileSystemError as e:
    logger.error(f"File system error: {e}")
    # Queue transaction for retry
    retry_queue.add(transaction)
except ValidationError as e:
    logger.warning(f"Validation error: {e}")
    # Skip invalid transaction, continue processing
    continue
except Exception as e:
    logger.error(f"Unexpected error: {e}", exc_info=True)
    # Continue operation, don't crash
    pass
```

**Error Recovery Mechanisms**
- Automatic retry with exponential backoff
- Fallback to alternative storage locations
- Transaction queuing for temporary failures
- Health check and self-healing capabilities

**Error Reporting**
- Structured error logging with context
- Error metrics and alerting
- User-friendly error messages
- Troubleshooting guidance

## Testing Standards

### Unit Testing Requirements

**Test Coverage**
- MUST achieve 90%+ code coverage for token tracking components
- MUST test all error conditions and edge cases
- MUST validate all configuration options
- MUST test concurrent access scenarios

**Test Data Management**
- Use deterministic test data for reproducible results
- Test with various prompt lengths and content types
- Test with different metadata combinations
- Test with malformed and edge-case data

### Property-Based Testing

**Universal Properties**
- For any valid transaction, recording should succeed
- For any sequence of transactions, data integrity should be maintained
- For any error condition, system should recover gracefully
- For any configuration change, system should adapt correctly

**Test Implementation**
```python
from hypothesis import given, strategies as st

@given(
    prompt_text=st.text(min_size=1, max_size=2000),
    tokens_used=st.integers(min_value=1, max_value=100000),
    elapsed_time=st.floats(min_value=0.001, max_value=3600.0)
)
def test_transaction_recording_property(self, prompt_text, tokens_used, elapsed_time):
    """For any valid transaction data, recording should succeed."""
    transaction = TokenTransaction(
        timestamp=datetime.now(),
        prompt_text=prompt_text,
        tokens_used=tokens_used,
        elapsed_time=elapsed_time,
        session_id="test_session",
        workspace_folder="test_workspace",
        hook_trigger_type="manual",
        agent_execution_id="test_exec",
        hook_name="test_hook"
    )
    
    result = self.tracker.record_transaction(transaction)
    assert result is True
```

### Integration Testing

**End-to-End Testing**
- Test complete workflow from hook trigger to CSV storage
- Test with real Kiro IDE integration
- Test error scenarios and recovery
- Test performance under load

**Mock-Based Testing**
- Mock external dependencies (file system, Kiro IDE)
- Test component interactions
- Validate error propagation
- Test configuration changes

## Performance Standards

### Performance Requirements

**Response Time**
- Token tracking MUST complete within 1 second for normal operations
- CSV writing MUST complete within 500ms for single transactions
- Metadata collection MUST complete within 200ms

**Throughput**
- System MUST handle 100 transactions per minute
- CSV file MUST support concurrent access from 10 processes
- Memory usage MUST remain below 50MB for tracking components

**Scalability**
- System MUST handle CSV files up to 100MB without performance degradation
- System MUST support workspaces with 1000+ files
- System MUST handle 24/7 operation without memory leaks

### Performance Monitoring

**Metrics Collection**
- Track transaction processing time
- Monitor CSV file size and growth rate
- Measure memory usage over time
- Monitor error rates and types

**Performance Optimization**
- Use buffered I/O for CSV writing
- Implement connection pooling for database operations
- Cache frequently accessed metadata
- Optimize data structures for common operations

## Troubleshooting Guidelines

### Common Issues

**CSV File Not Created**
1. Check file permissions in target directory
2. Verify hook configuration is correct
3. Check Kiro IDE logs for error messages
4. Validate workspace folder structure

**Missing Transaction Data**
1. Verify hook triggers are configured correctly
2. Check that AI interactions are actually occurring
3. Review error logs for data collection failures
4. Validate metadata collection is working

**Performance Issues**
1. Check CSV file size and consider rotation
2. Monitor system resource usage
3. Review concurrent access patterns
4. Optimize hook execution time

**Data Corruption**
1. Validate CSV file integrity
2. Check for concurrent access issues
3. Review file locking mechanisms
4. Restore from backup if necessary

### Diagnostic Tools

**CSV Validation Script**
```bash
python -c "
import csv
import pandas as pd

try:
    df = pd.read_csv('.kiro/token_transactions.csv')
    print(f'CSV file is valid with {len(df)} records')
    
    # Check required columns
    required_cols = ['timestamp', 'prompt_text', 'tokens_used']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print(f'Missing columns: {missing}')
    else:
        print('All required columns present')
        
except Exception as e:
    print(f'CSV validation failed: {e}')
"
```

**Hook Status Check**
```bash
# Check hook configuration
cat .kiro/hooks/token-tracking.kiro.hook | python -m json.tool

# Verify hook file permissions
ls -la .kiro/hooks/token-tracking.kiro.hook

# Check for hook execution in logs
grep -i "token.*hook" ~/.kiro/logs/kiro.log | tail -10
```

**System Health Check**
```bash
# Check disk space
df -h .kiro/

# Check file permissions
ls -la .kiro/token_transactions.csv

# Check recent transactions
tail -5 .kiro/token_transactions.csv

# Validate data integrity
python -c "
import pandas as pd
df = pd.read_csv('.kiro/token_transactions.csv')
print(f'Records: {len(df)}')
print(f'Date range: {df[\"timestamp\"].min()} to {df[\"timestamp\"].max()}')
print(f'Total tokens: {df[\"tokens_used\"].sum():,}')
"
```

## Best Practices

### Development Best Practices

**Code Organization**
- Keep token tracking code modular and testable
- Use dependency injection for external dependencies
- Implement proper logging throughout the system
- Follow established coding standards and patterns

**Configuration Management**
- Use environment-specific configuration files
- Validate configuration at startup
- Provide sensible defaults for all parameters
- Document all configuration options

**Error Handling**
- Implement comprehensive error handling
- Use structured logging for troubleshooting
- Provide user-friendly error messages
- Implement graceful degradation strategies

### Operational Best Practices

**Monitoring and Alerting**
- Monitor CSV file growth and rotation
- Alert on error rate increases
- Track performance metrics over time
- Monitor system resource usage

**Data Management**
- Implement regular backup procedures
- Set up data retention policies
- Monitor data quality and integrity
- Plan for data migration and upgrades

**Security**
- Regularly review access permissions
- Implement data encryption where required
- Monitor for unauthorized access
- Keep security patches up to date

### User Experience Best Practices

**Transparency**
- Clearly communicate what data is being tracked
- Provide easy access to tracking data
- Allow users to control tracking preferences
- Explain the benefits of token tracking

**Performance**
- Minimize impact on normal workflows
- Provide fast access to tracking data
- Optimize for common use cases
- Handle errors gracefully without user impact

**Privacy**
- Respect user privacy preferences
- Implement data minimization principles
- Provide data export and deletion capabilities
- Follow applicable privacy regulations

This document establishes the standards for token tracking within the AI Hydra project. All token tracking implementations must comply with these standards to ensure consistency, reliability, and maintainability.