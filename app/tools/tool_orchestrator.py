"""
Tool Orchestrator - Workflow Engine for Multi-Agent Tool Execution
Part of the Multi-Agent Voice AI System Transformation

This orchestrator manages:
- Sequential and parallel tool execution workflows
- Tool dependency resolution and error handling
- Result aggregation and context propagation
- Performance monitoring and latency optimization
- Retry logic and failure recovery

Supports complex business workflows like:
- Payment processing + confirmation + CRM update
- Ticket creation + notification + calendar scheduling
- Multi-step troubleshooting with conditional branching
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Callable, Tuple
from datetime import datetime, timedelta
from enum import Enum
import json
import traceback
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from app.core.latency_optimizer import latency_monitor
from app.tools.external_apis import ExternalAPIManager
from app.tools.payment_tools import PaymentProcessor
from app.tools.communication_tools import CommunicationManager
from app.tools.scheduling_tools import SchedulingManager

# Configure logging
logger = logging.getLogger(__name__)

class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PARTIAL_SUCCESS = "partial_success"

class ToolExecutionMode(Enum):
    """Tool execution modes"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    RETRY_ON_FAILURE = "retry_on_failure"

@dataclass
class ToolResult:
    """Result of a single tool execution"""
    tool_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0
    retry_count: int = 0
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()

@dataclass
class WorkflowStep:
    """Individual step in a workflow"""
    step_id: str
    tool_name: str
    tool_function: Callable
    parameters: Dict[str, Any]
    dependencies: List[str] = None
    retry_attempts: int = 3
    timeout_seconds: int = 30
    on_success: Optional[Callable] = None
    on_failure: Optional[Callable] = None
    conditional_logic: Optional[Callable] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

@dataclass
class WorkflowDefinition:
    """Complete workflow definition"""
    workflow_id: str
    name: str
    description: str
    steps: List[WorkflowStep]
    execution_mode: ToolExecutionMode = ToolExecutionMode.SEQUENTIAL
    max_execution_time: int = 300  # 5 minutes default
    rollback_on_failure: bool = False
    context_propagation: bool = True

class ToolOrchestrator:
    """
    Advanced workflow engine for orchestrating complex multi-tool operations
    Handles sequential, parallel, and conditional tool execution patterns
    """
    
    def __init__(self):
        self.external_apis = ExternalAPIManager()
        self.payment_processor = PaymentProcessor()
        self.communication_manager = CommunicationManager()
        self.scheduling_manager = SchedulingManager()
        
        # Workflow tracking
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self.workflow_history: List[Dict[str, Any]] = []
        
        # Performance metrics
        self.execution_metrics = {
            "total_workflows": 0,
            "successful_workflows": 0,
            "failed_workflows": 0,
            "average_execution_time": 0,
            "tool_usage_stats": {}
        }
        
        # Thread pool for parallel execution
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Pre-defined workflow templates
        self._register_workflow_templates()
        
    def _register_workflow_templates(self):
        """Register common workflow templates"""
        self.workflow_templates = {
            "payment_processing": self._create_payment_workflow_template(),
            "support_ticket_creation": self._create_support_ticket_template(),
            "customer_onboarding": self._create_onboarding_template(),
            "billing_dispute_resolution": self._create_dispute_resolution_template(),
            "emergency_response": self._create_emergency_response_template()
        }

    @latency_monitor("orchestrator_execute_workflow")
    async def execute_workflow(self, 
                             workflow_def: WorkflowDefinition,
                             initial_context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a complete workflow with all steps and error handling
        
        Args:
            workflow_def: Workflow definition with steps and configuration
            initial_context: Initial context data for workflow execution
            
        Returns:
            Workflow execution results with detailed step outcomes
        """
        if initial_context is None:
            initial_context = {}
            
        workflow_id = f"{workflow_def.workflow_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initialize workflow tracking
        workflow_state = {
            "workflow_id": workflow_id,
            "definition": workflow_def,
            "status": WorkflowStatus.PENDING,
            "context": initial_context.copy(),
            "step_results": {},
            "start_time": datetime.now(),
            "end_time": None,
            "total_execution_time": 0,
            "errors": [],
            "rollback_performed": False
        }
        
        self.active_workflows[workflow_id] = workflow_state
        
        try:
            logger.info(f"Starting workflow execution: {workflow_def.name} ({workflow_id})")
            workflow_state["status"] = WorkflowStatus.RUNNING
            
            # Execute workflow based on execution mode
            if workflow_def.execution_mode == ToolExecutionMode.SEQUENTIAL:
                results = await self._execute_sequential_workflow(workflow_def, workflow_state)
            elif workflow_def.execution_mode == ToolExecutionMode.PARALLEL:
                results = await self._execute_parallel_workflow(workflow_def, workflow_state)
            elif workflow_def.execution_mode == ToolExecutionMode.CONDITIONAL:
                results = await self._execute_conditional_workflow(workflow_def, workflow_state)
            else:
                raise ValueError(f"Unsupported execution mode: {workflow_def.execution_mode}")
            
            # Finalize workflow
            workflow_state["end_time"] = datetime.now()
            workflow_state["total_execution_time"] = (
                workflow_state["end_time"] - workflow_state["start_time"]
            ).total_seconds() * 1000  # Convert to milliseconds
            
            # Determine final status
            successful_steps = sum(1 for r in results.values() if r.success)
            total_steps = len(results)
            
            if successful_steps == total_steps:
                workflow_state["status"] = WorkflowStatus.COMPLETED
            elif successful_steps > 0:
                workflow_state["status"] = WorkflowStatus.PARTIAL_SUCCESS
            else:
                workflow_state["status"] = WorkflowStatus.FAILED
            
            # Update metrics
            self._update_execution_metrics(workflow_state)
            
            # Move to history
            self.workflow_history.append(workflow_state.copy())
            del self.active_workflows[workflow_id]
            
            logger.info(f"Workflow completed: {workflow_id} - Status: {workflow_state['status']}")
            
            return {
                "workflow_id": workflow_id,
                "status": workflow_state["status"].value,
                "execution_time_ms": workflow_state["total_execution_time"],
                "step_results": {k: self._serialize_tool_result(v) for k, v in results.items()},
                "final_context": workflow_state["context"],
                "success_rate": f"{successful_steps}/{total_steps}",
                "errors": workflow_state["errors"]
            }
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {workflow_id} - {str(e)}")
            workflow_state["status"] = WorkflowStatus.FAILED
            workflow_state["errors"].append({
                "type": "workflow_execution_error",
                "message": str(e),
                "traceback": traceback.format_exc()
            })
            
            # Attempt rollback if configured
            if workflow_def.rollback_on_failure:
                await self._perform_rollback(workflow_state)
            
            return {
                "workflow_id": workflow_id,
                "status": WorkflowStatus.FAILED.value,
                "error": str(e),
                "partial_results": workflow_state.get("step_results", {})
            }

    async def _execute_sequential_workflow(self, 
                                         workflow_def: WorkflowDefinition,
                                         workflow_state: Dict[str, Any]) -> Dict[str, ToolResult]:
        """Execute workflow steps sequentially"""
        results = {}
        context = workflow_state["context"]
        
        for step in workflow_def.steps:
            # Check dependencies
            if not self._check_step_dependencies(step, results):
                logger.warning(f"Skipping step {step.step_id} - dependencies not met")
                continue
            
            # Execute step with retry logic
            result = await self._execute_step_with_retry(step, context)
            results[step.step_id] = result
            workflow_state["step_results"][step.step_id] = result
            
            # Update context if successful and context propagation is enabled
            if result.success and workflow_def.context_propagation:
                if isinstance(result.result, dict):
                    context.update(result.result)
            
            # Handle step failure
            if not result.success:
                if step.on_failure:
                    try:
                        await step.on_failure(result, context)
                    except Exception as e:
                        logger.error(f"Step failure handler error: {str(e)}")
                
                # Stop execution if critical step fails
                if not workflow_def.rollback_on_failure:
                    break
            else:
                # Execute success handler
                if step.on_success:
                    try:
                        await step.on_success(result, context)
                    except Exception as e:
                        logger.error(f"Step success handler error: {str(e)}")
        
        return results

    async def _execute_parallel_workflow(self, 
                                       workflow_def: WorkflowDefinition,
                                       workflow_state: Dict[str, Any]) -> Dict[str, ToolResult]:
        """Execute workflow steps in parallel where possible"""
        results = {}
        context = workflow_state["context"]
        
        # Build dependency graph
        dependency_graph = self._build_dependency_graph(workflow_def.steps)
        
        # Execute steps in waves based on dependencies
        completed_steps = set()
        
        while len(completed_steps) < len(workflow_def.steps):
            # Find steps ready for execution
            ready_steps = [
                step for step in workflow_def.steps 
                if step.step_id not in completed_steps and 
                all(dep in completed_steps for dep in step.dependencies)
            ]
            
            if not ready_steps:
                logger.error("Circular dependency detected or no more executable steps")
                break
            
            # Execute ready steps in parallel
            tasks = [
                self._execute_step_with_retry(step, context.copy())
                for step in ready_steps
            ]
            
            step_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for step, result in zip(ready_steps, step_results):
                if isinstance(result, Exception):
                    result = ToolResult(
                        tool_name=step.tool_name,
                        success=False,
                        error=str(result)
                    )
                
                results[step.step_id] = result
                completed_steps.add(step.step_id)
                
                # Update shared context (with thread safety considerations)
                if result.success and workflow_def.context_propagation:
                    if isinstance(result.result, dict):
                        context.update(result.result)
        
        return results

    async def _execute_conditional_workflow(self, 
                                          workflow_def: WorkflowDefinition,
                                          workflow_state: Dict[str, Any]) -> Dict[str, ToolResult]:
        """Execute workflow with conditional step execution"""
        results = {}
        context = workflow_state["context"]
        
        for step in workflow_def.steps:
            # Check if step should be executed based on conditional logic
            if step.conditional_logic:
                try:
                    should_execute = await step.conditional_logic(context, results)
                    if not should_execute:
                        logger.info(f"Skipping step {step.step_id} - conditional logic returned False")
                        continue
                except Exception as e:
                    logger.error(f"Conditional logic error for step {step.step_id}: {str(e)}")
                    continue
            
            # Check dependencies
            if not self._check_step_dependencies(step, results):
                continue
            
            # Execute step
            result = await self._execute_step_with_retry(step, context)
            results[step.step_id] = result
            
            # Update context
            if result.success and workflow_def.context_propagation:
                if isinstance(result.result, dict):
                    context.update(result.result)
        
        return results

    async def _execute_step_with_retry(self, 
                                     step: WorkflowStep, 
                                     context: Dict[str, Any]) -> ToolResult:
        """Execute a single step with retry logic"""
        last_error = None
        
        for attempt in range(step.retry_attempts + 1):
            try:
                start_time = datetime.now()
                
                # Prepare parameters with context injection
                params = step.parameters.copy()
                if step.tool_name in ["communication", "scheduling"]:
                    params["context"] = context
                
                # Execute tool with timeout
                result = await asyncio.wait_for(
                    step.tool_function(**params),
                    timeout=step.timeout_seconds
                )
                
                execution_time = (datetime.now() - start_time).total_seconds() * 1000
                
                # Check if result indicates success
                if isinstance(result, dict) and result.get("success", True):
                    return ToolResult(
                        tool_name=step.tool_name,
                        success=True,
                        result=result,
                        execution_time_ms=execution_time,
                        retry_count=attempt
                    )
                else:
                    last_error = result.get("error", "Unknown error") if isinstance(result, dict) else "Invalid result format"
                    
            except asyncio.TimeoutError:
                last_error = f"Tool execution timeout after {step.timeout_seconds} seconds"
                logger.warning(f"Step {step.step_id} timeout on attempt {attempt + 1}")
                
            except Exception as e:
                last_error = str(e)
                logger.warning(f"Step {step.step_id} failed on attempt {attempt + 1}: {str(e)}")
            
            # Wait before retry (exponential backoff)
            if attempt < step.retry_attempts:
                wait_time = min(2 ** attempt, 10)  # Max 10 seconds
                await asyncio.sleep(wait_time)
        
        # All attempts failed
        return ToolResult(
            tool_name=step.tool_name,
            success=False,
            error=last_error,
            retry_count=step.retry_attempts
        )

    def _check_step_dependencies(self, 
                               step: WorkflowStep, 
                               completed_results: Dict[str, ToolResult]) -> bool:
        """Check if step dependencies are satisfied"""
        for dep_id in step.dependencies:
            if dep_id not in completed_results or not completed_results[dep_id].success:
                return False
        return True

    def _build_dependency_graph(self, steps: List[WorkflowStep]) -> Dict[str, List[str]]:
        """Build dependency graph for parallel execution planning"""
        graph = {}
        for step in steps:
            graph[step.step_id] = step.dependencies.copy()
        return graph

    async def _perform_rollback(self, workflow_state: Dict[str, Any]):
        """Perform rollback operations for failed workflow"""
        logger.info(f"Performing rollback for workflow: {workflow_state['workflow_id']}")
        
        # Implement rollback logic based on completed steps
        # This is a simplified version - in production, you'd need more sophisticated rollback
        rollback_operations = []
        
        for step_id, result in workflow_state["step_results"].items():
            if result.success and hasattr(result, 'rollback_data'):
                rollback_operations.append({
                    "step_id": step_id,
                    "rollback_data": result.rollback_data
                })
        
        # Execute rollback operations in reverse order
        for operation in reversed(rollback_operations):
            try:
                # Call appropriate rollback function based on tool type
                await self._execute_rollback_operation(operation)
            except Exception as e:
                logger.error(f"Rollback operation failed for step {operation['step_id']}: {str(e)}")
        
        workflow_state["rollback_performed"] = True

    async def _execute_rollback_operation(self, operation: Dict[str, Any]):
        """Execute individual rollback operation"""
        # Implement specific rollback logic based on tool type
        step_id = operation["step_id"]
        rollback_data = operation["rollback_data"]
        
        logger.info(f"Executing rollback for step: {step_id}")
        # Placeholder for actual rollback implementation

    def _update_execution_metrics(self, workflow_state: Dict[str, Any]):
        """Update performance metrics"""
        self.execution_metrics["total_workflows"] += 1
        
        if workflow_state["status"] == WorkflowStatus.COMPLETED:
            self.execution_metrics["successful_workflows"] += 1
        else:
            self.execution_metrics["failed_workflows"] += 1
        
        # Update average execution time
        total_time = workflow_state["total_execution_time"]
        current_avg = self.execution_metrics["average_execution_time"]
        total_workflows = self.execution_metrics["total_workflows"]
        
        self.execution_metrics["average_execution_time"] = (
            (current_avg * (total_workflows - 1) + total_time) / total_workflows
        )

    def _serialize_tool_result(self, result: ToolResult) -> Dict[str, Any]:
        """Serialize ToolResult for JSON response"""
        return {
            "tool_name": result.tool_name,
            "success": result.success,
            "result": result.result,
            "error": result.error,
            "execution_time_ms": result.execution_time_ms,
            "retry_count": result.retry_count,
            "timestamp": result.timestamp.isoformat() if result.timestamp else None
        }

    # Pre-defined workflow templates
    def _create_payment_workflow_template(self) -> WorkflowDefinition:
        """Create payment processing workflow template"""
        return WorkflowDefinition(
            workflow_id="payment_processing",
            name="Payment Processing Workflow",
            description="Complete payment processing with confirmation and notifications",
            steps=[
                WorkflowStep(
                    step_id="validate_payment",
                    tool_name="payment",
                    tool_function=self.payment_processor.validate_payment_details,
                    parameters={}
                ),
                WorkflowStep(
                    step_id="process_payment",
                    tool_name="payment",
                    tool_function=self.payment_processor.process_payment,
                    parameters={},
                    dependencies=["validate_payment"]
                ),
                WorkflowStep(
                    step_id="update_crm",
                    tool_name="crm",
                    tool_function=self.external_apis.update_customer_record,
                    parameters={},
                    dependencies=["process_payment"]
                ),
                WorkflowStep(
                    step_id="send_confirmation",
                    tool_name="communication",
                    tool_function=self.communication_manager.send_payment_confirmation,
                    parameters={},
                    dependencies=["process_payment"]
                )
            ],
            execution_mode=ToolExecutionMode.SEQUENTIAL,
            rollback_on_failure=True
        )

    def _create_support_ticket_template(self) -> WorkflowDefinition:
        """Create support ticket creation workflow template"""
        return WorkflowDefinition(
            workflow_id="support_ticket_creation",
            name="Support Ticket Creation Workflow",
            description="Create support ticket with notifications and scheduling",
            steps=[
                WorkflowStep(
                    step_id="create_ticket",
                    tool_name="ticketing",
                    tool_function=self.external_apis.create_support_ticket,
                    parameters={}
                ),
                WorkflowStep(
                    step_id="notify_customer",
                    tool_name="communication",
                    tool_function=self.communication_manager.send_ticket_notification,
                    parameters={},
                    dependencies=["create_ticket"]
                ),
                WorkflowStep(
                    step_id="schedule_followup",
                    tool_name="scheduling",
                    tool_function=self.scheduling_manager.schedule_followup,
                    parameters={},
                    dependencies=["create_ticket"]
                )
            ],
            execution_mode=ToolExecutionMode.PARALLEL  # Notification and scheduling can run in parallel
        )

    def _create_onboarding_template(self) -> WorkflowDefinition:
        """Create customer onboarding workflow template"""
        return WorkflowDefinition(
            workflow_id="customer_onboarding",
            name="Customer Onboarding Workflow", 
            description="Complete customer onboarding process",
            steps=[
                WorkflowStep(
                    step_id="create_account",
                    tool_name="crm",
                    tool_function=self.external_apis.create_customer_account,
                    parameters={}
                ),
                WorkflowStep(
                    step_id="send_welcome_email",
                    tool_name="communication",
                    tool_function=self.communication_manager.send_welcome_email,
                    parameters={},
                    dependencies=["create_account"]
                ),
                WorkflowStep(
                    step_id="schedule_onboarding_call",
                    tool_name="scheduling",
                    tool_function=self.scheduling_manager.schedule_onboarding_call,
                    parameters={},
                    dependencies=["create_account"]
                ),
                WorkflowStep(
                    step_id="setup_billing",
                    tool_name="payment",
                    tool_function=self.payment_processor.setup_customer_billing,
                    parameters={},
                    dependencies=["create_account"]
                )
            ],
            execution_mode=ToolExecutionMode.CONDITIONAL
        )

    def _create_dispute_resolution_template(self) -> WorkflowDefinition:
        """Create billing dispute resolution workflow template"""
        return WorkflowDefinition(
            workflow_id="billing_dispute_resolution",
            name="Billing Dispute Resolution Workflow",
            description="Handle billing disputes with investigation and resolution",
            steps=[
                WorkflowStep(
                    step_id="log_dispute",
                    tool_name="crm",
                    tool_function=self.external_apis.log_billing_dispute,
                    parameters={}
                ),
                WorkflowStep(
                    step_id="investigate_charges",
                    tool_name="payment",
                    tool_function=self.payment_processor.investigate_transaction,
                    parameters={},
                    dependencies=["log_dispute"]
                ),
                WorkflowStep(
                    step_id="notify_customer",
                    tool_name="communication",
                    tool_function=self.communication_manager.send_dispute_update,
                    parameters={},
                    dependencies=["log_dispute"]
                ),
                WorkflowStep(
                    step_id="process_resolution",
                    tool_name="payment",
                    tool_function=self.payment_processor.process_dispute_resolution,
                    parameters={},
                    dependencies=["investigate_charges"]
                )
            ],
            execution_mode=ToolExecutionMode.SEQUENTIAL
        )

    def _create_emergency_response_template(self) -> WorkflowDefinition:
        """Create emergency response workflow template"""
        return WorkflowDefinition(
            workflow_id="emergency_response",
            name="Emergency Response Workflow",
            description="Handle emergency situations with immediate escalation",
            steps=[
                WorkflowStep(
                    step_id="create_urgent_ticket",
                    tool_name="ticketing",
                    tool_function=self.external_apis.create_urgent_ticket,
                    parameters={"priority": "critical"}
                ),
                WorkflowStep(
                    step_id="notify_on_call",
                    tool_name="communication",
                    tool_function=self.communication_manager.notify_on_call_team,
                    parameters={},
                    dependencies=["create_urgent_ticket"]
                ),
                WorkflowStep(
                    step_id="schedule_immediate_callback",
                    tool_name="scheduling",
                    tool_function=self.scheduling_manager.schedule_emergency_callback,
                    parameters={},
                    dependencies=["create_urgent_ticket"]
                )
            ],
            execution_mode=ToolExecutionMode.PARALLEL,
            max_execution_time=60  # 1 minute for emergency response
        )

    # Public interface methods
    async def execute_template_workflow(self, 
                                      template_name: str,
                                      parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a pre-defined workflow template"""
        if template_name not in self.workflow_templates:
            raise ValueError(f"Unknown workflow template: {template_name}")
        
        template = self.workflow_templates[template_name]
        
        # Inject parameters into workflow steps
        for step in template.steps:
            step.parameters.update(parameters.get(step.step_id, {}))
        
        return await self.execute_workflow(template, parameters)

    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a running workflow"""
        if workflow_id in self.active_workflows:
            workflow_state = self.active_workflows[workflow_id]
            return {
                "workflow_id": workflow_id,
                "status": workflow_state["status"].value,
                "start_time": workflow_state["start_time"].isoformat(),
                "completed_steps": len(workflow_state["step_results"]),
                "total_steps": len(workflow_state["definition"].steps)
            }
        return None

    def get_execution_metrics(self) -> Dict[str, Any]:
        """Get orchestrator performance metrics"""
        return {
            **self.execution_metrics,
            "active_workflows": len(self.active_workflows),
            "template_count": len(self.workflow_templates),
            "success_rate": (
                self.execution_metrics["successful_workflows"] / 
                max(self.execution_metrics["total_workflows"], 1)
            ) * 100
        }

    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow"""
        if workflow_id in self.active_workflows:
            workflow_state = self.active_workflows[workflow_id]
            workflow_state["status"] = WorkflowStatus.CANCELLED
            
            # Perform cleanup if needed
            logger.info(f"Workflow cancelled: {workflow_id}")
            return True
        return False