import logging
from typing import Dict, Any, List, Optional, Tuple
import asyncio
from datetime import datetime
import json
import uuid

from agents.base_agent import BaseAgent
from agents.api_executor import api_executor
from storage import mongodb_connector, redis_connector
from models import (
    WorkflowTemplate, WorkflowStep, StepType, 
    WorkflowExecution, WorkflowStatus, WorkflowTrigger
)
from config.settings import config

logger = logging.getLogger(__name__)

class WorkflowAutomationAgent(BaseAgent):
    """
    The Process Manager - handles multi-step business workflows
    """
    
    def __init__(self):
        super().__init__(
            name="workflow_automation",
            description="Executes and manages multi-step automated workflows"
        )
        self.running_workflows: Dict[str, WorkflowExecution] = {}
        self.workflow_templates = self._load_default_templates()
    
    def _load_default_templates(self) -> Dict[str, WorkflowTemplate]:
        """Load pre-built workflow templates"""
        templates = {}
        
        # Order Creation Workflow
        order_workflow = WorkflowTemplate(
            id="",
            name="Order Creation Workflow",
            description="Complete order creation with quotes and approval",
            category="order_management",
            created_by=0,  # System
            steps=[
                WorkflowStep(
                    id="create_order",
                    name="Create Order",
                    type=StepType.API_CALL,
                    config={
                        "endpoint": "/api/orders",
                        "method": "POST",
                        "data_mapping": {
                            "description": "{{description}}",
                            "items": "{{items}}",
                            "project_id": "{{project_id}}"
                        }
                    },
                    next_steps=["get_quotes"]
                ),
                WorkflowStep(
                    id="get_quotes",
                    name="Request Quotes",
                    type=StepType.API_CALL,
                    config={
                        "endpoint": "/api/orders/{{order_id}}/quotes",
                        "method": "POST",
                        "data_mapping": {
                            "suppliers": "{{supplier_ids}}"
                        }
                    },
                    next_steps=["check_quotes"]
                ),
                WorkflowStep(
                    id="check_quotes",
                    name="Check Quote Responses",
                    type=StepType.WAIT,
                    config={
                        "wait_time": 300,  # 5 minutes
                        "check_endpoint": "/api/orders/{{order_id}}/quotes",
                        "success_condition": "quotes_received > 0"
                    },
                    next_steps=["select_quote"]
                ),
                WorkflowStep(
                    id="select_quote",
                    name="Select Best Quote",
                    type=StepType.CONDITION,
                    config={
                        "conditions": [
                            {
                                "if": "lowest_quote < budget",
                                "then": "approve_order"
                            },
                            {
                                "else": "manager_approval"
                            }
                        ]
                    },
                    next_steps=["approve_order", "manager_approval"]
                ),
                WorkflowStep(
                    id="approve_order",
                    name="Auto-Approve Order",
                    type=StepType.API_CALL,
                    config={
                        "endpoint": "/api/orders/{{order_id}}/approve",
                        "method": "POST",
                        "data_mapping": {
                            "quote_id": "{{selected_quote_id}}",
                            "notes": "Auto-approved within budget"
                        }
                    },
                    next_steps=["notify_complete"]
                ),
                WorkflowStep(
                    id="manager_approval",
                    name="Request Manager Approval",
                    type=StepType.NOTIFICATION,
                    config={
                        "type": "approval_request",
                        "to": "{{manager_id}}",
                        "message": "Order {{order_id}} requires approval - exceeds budget"
                    },
                    next_steps=["notify_complete"]
                ),
                WorkflowStep(
                    id="notify_complete",
                    name="Notify Completion",
                    type=StepType.NOTIFICATION,
                    config={
                        "type": "completion",
                        "to": "{{requester_id}}",
                        "message": "Order {{order_id}} workflow completed"
                    },
                    next_steps=[]
                )
            ]
        )
        templates["order_creation"] = order_workflow
        
        # Daily Reporting Workflow
        report_workflow = WorkflowTemplate(
            id="",
            name="Daily Progress Report",
            description="Generate and distribute daily progress reports",
            category="reporting",
            created_by=0,
            steps=[
                WorkflowStep(
                    id="collect_data",
                    name="Collect Project Data",
                    type=StepType.PARALLEL,
                    config={
                        "parallel_steps": [
                            {
                                "endpoint": "/api/projects/{{project_id}}/progress",
                                "store_as": "progress_data"
                            },
                            {
                                "endpoint": "/api/projects/{{project_id}}/issues",
                                "store_as": "issues_data"
                            },
                            {
                                "endpoint": "/api/projects/{{project_id}}/milestones",
                                "store_as": "milestone_data"
                            }
                        ]
                    },
                    next_steps=["generate_report"]
                ),
                WorkflowStep(
                    id="generate_report",
                    name="Generate Report",
                    type=StepType.TRANSFORM,
                    config={
                        "template": "daily_progress",
                        "data_sources": ["progress_data", "issues_data", "milestone_data"],
                        "output_format": "pdf"
                    },
                    next_steps=["distribute_report"]
                ),
                WorkflowStep(
                    id="distribute_report",
                    name="Distribute Report",
                    type=StepType.NOTIFICATION,
                    config={
                        "type": "report",
                        "recipients": "{{stakeholder_list}}",
                        "attachment": "{{generated_report}}"
                    },
                    next_steps=[]
                )
            ]
        )
        templates["daily_report"] = report_workflow
        
        return templates
    
    async def validate_request(self, request: Dict[str, Any]) -> bool:
        """Validate workflow request"""
        return "workflow_name" in request or "workflow_id" in request or "query" in request
    
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process workflow request - create, execute, or manage workflows
        """
        try:
            context = request.get("context", {})
            parameters = request.get("parameters", {})
            
            # Detect workflow intent
            if "workflow_id" in request:
                # Execute existing workflow
                return await self._execute_workflow(request["workflow_id"], request)
                
            elif "workflow_name" in request:
                # Execute named template
                template = self.workflow_templates.get(request["workflow_name"])
                if not template:
                    # Try to load from database
                    workflows = await mongodb_connector.get_workflows_by_category("all")
                    template = next(
                        (w for w in workflows if w["name"] == request["workflow_name"]),
                        None
                    )
                
                if template:
                    return await self._execute_workflow_template(template, request)
                else:
                    raise ValueError(f"Workflow '{request['workflow_name']}' not found")
                    
            else:
                # Natural language workflow request
                query = request.get("query", "")
                detected_workflow = await self._detect_workflow_intent(query)
                
                if detected_workflow:
                    return await self._execute_workflow_template(
                        detected_workflow["template"],
                        {**request, "variables": detected_workflow["variables"]}
                    )
                else:
                    return await self._create_custom_workflow(query, context)
                    
        except Exception as e:
            logger.error(f"Workflow automation error: {e}", exc_info=True)
            raise
    
    async def _detect_workflow_intent(self, query: str) -> Optional[Dict[str, Any]]:
        """Detect which workflow template to use based on query"""
        query_lower = query.lower()
        
        # Order workflow keywords
        if any(word in query_lower for word in ["order", "purchase", "quote", "supplier"]):
            if "create" in query_lower or "new" in query_lower:
                return {
                    "template": self.workflow_templates["order_creation"],
                    "variables": self._extract_order_variables(query)
                }
        
        # Report workflow keywords
        elif any(word in query_lower for word in ["report", "daily", "progress", "status"]):
            return {
                "template": self.workflow_templates["daily_report"],
                "variables": self._extract_report_variables(query)
            }
        
        # Approval workflow keywords
        elif "approve" in query_lower or "approval" in query_lower:
            return await self._build_approval_workflow(query)
        
        return None
    
    async def _execute_workflow_template(
        self,
        template: WorkflowTemplate,
        request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Execute a workflow template"""
        # Create execution record
        execution = WorkflowExecution(
            id="",
            workflow_id=template.id or str(uuid.uuid4()),
            workflow_name=template.name,
            user_id=request.get("context", {}).get("user_id", 0),
            status=WorkflowStatus.RUNNING,
            current_step=template.steps[0].id if template.steps else None,
            execution_data=request.get("variables", {})
        )
        
        # Save to database
        exec_id = await mongodb_connector.save_workflow(execution.dict())
        execution.id = exec_id
        
        # Cache in memory
        self.running_workflows[exec_id] = execution
        
        # Start execution
        asyncio.create_task(self._run_workflow(execution, template))
        
        return {
            "data": {
                "workflow_id": exec_id,
                "workflow_name": template.name,
                "status": "started",
                "message": f"Workflow '{template.name}' started successfully",
                "estimated_duration": self._estimate_duration(template)
            }
        }
    
    async def _run_workflow(
        self,
        execution: WorkflowExecution,
        template: WorkflowTemplate
    ):
        """Run workflow execution"""
        try:
            steps_by_id = {step.id: step for step in template.steps}
            
            while execution.current_step:
                current_step = steps_by_id.get(execution.current_step)
                if not current_step:
                    break
                
                logger.info(f"Executing step: {current_step.name}")
                
                # Execute step based on type
                if current_step.type == StepType.API_CALL:
                    result = await self._execute_api_step(current_step, execution)
                    
                elif current_step.type == StepType.CONDITION:
                    result = await self._execute_condition_step(current_step, execution)
                    
                elif current_step.type == StepType.PARALLEL:
                    result = await self._execute_parallel_step(current_step, execution)
                    
                elif current_step.type == StepType.WAIT:
                    result = await self._execute_wait_step(current_step, execution)
                    
                elif current_step.type == StepType.TRANSFORM:
                    result = await self._execute_transform_step(current_step, execution)
                    
                elif current_step.type == StepType.NOTIFICATION:
                    result = await self._execute_notification_step(current_step, execution)
                    
                else:
                    result = {"success": False, "error": f"Unknown step type: {current_step.type}"}
                
                # Store step result
                execution.step_results[current_step.id] = result
                
                # Handle failure
                if not result.get("success", False):
                    execution.status = WorkflowStatus.FAILED
                    execution.errors.append({
                        "step": current_step.id,
                        "error": result.get("error", "Unknown error"),
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    break
                
                # Determine next step
                next_step_id = await self._determine_next_step(
                    current_step,
                    result,
                    execution
                )
                
                execution.current_step = next_step_id
                
                # Save progress
                await mongodb_connector.save_workflow(execution.dict())
            
            # Workflow completed
            if not execution.errors:
                execution.status = WorkflowStatus.COMPLETED
                execution.completed_at = datetime.utcnow()
            
            # Final save
            await mongodb_connector.save_workflow(execution.dict())
            
            # Clean up
            if execution.id in self.running_workflows:
                del self.running_workflows[execution.id]
                
        except Exception as e:
            logger.error(f"Workflow execution error: {e}", exc_info=True)
            execution.status = WorkflowStatus.FAILED
            execution.errors.append({
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            })
            await mongodb_connector.save_workflow(execution.dict())
    
    async def _execute_api_step(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution
    ) -> Dict[str, Any]:
        """Execute API call step"""
        try:
            config = step.config
            
            # Resolve variables in endpoint
            endpoint = self._resolve_variables(
                config["endpoint"],
                execution.execution_data
            )
            
            # Resolve data mapping
            data = {}
            if "data_mapping" in config:
                for key, template in config["data_mapping"].items():
                    data[key] = self._resolve_variables(
                        template,
                        execution.execution_data
                    )
            
            # Execute API call
            result = await api_executor.process({
                "endpoint": endpoint,
                "method": config.get("method", "GET"),
                "data": data,
                "context": {"workflow_id": execution.id}
            })
            
            # Store response data
            if result["success"]:
                response_data = result["data"].get("response", {})
                
                # Store specific fields if configured
                if "store_fields" in config:
                    for field, var_name in config["store_fields"].items():
                        if field in response_data:
                            execution.execution_data[var_name] = response_data[field]
                else:
                    # Store entire response
                    execution.execution_data[f"{step.id}_response"] = response_data
            
            return result
            
        except Exception as e:
            logger.error(f"API step error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _execute_condition_step(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution
    ) -> Dict[str, Any]:
        """Execute conditional branching step"""
        try:
            conditions = step.config["conditions"]
            
            for condition in conditions:
                if "if" in condition:
                    # Evaluate condition
                    if self._evaluate_condition(
                        condition["if"],
                        execution.execution_data
                    ):
                        return {
                            "success": True,
                            "next_step": condition["then"]
                        }
                elif "else" in condition:
                    return {
                        "success": True,
                        "next_step": condition["else"]
                    }
            
            return {"success": False, "error": "No condition matched"}
            
        except Exception as e:
            logger.error(f"Condition step error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _execute_parallel_step(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution
    ) -> Dict[str, Any]:
        """Execute parallel API calls"""
        try:
            parallel_configs = step.config["parallel_steps"]
            
            # Create tasks for parallel execution
            tasks = []
            for config in parallel_configs:
                endpoint = self._resolve_variables(
                    config["endpoint"],
                    execution.execution_data
                )
                
                task = api_executor.process({
                    "endpoint": endpoint,
                    "method": config.get("method", "GET"),
                    "context": {"workflow_id": execution.id}
                })
                tasks.append(task)
            
            # Execute in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Store results
            all_success = True
            for i, (config, result) in enumerate(zip(parallel_configs, results)):
                if isinstance(result, Exception):
                    all_success = False
                    logger.error(f"Parallel step {i} failed: {result}")
                elif result.get("success"):
                    store_as = config.get("store_as", f"parallel_{i}")
                    execution.execution_data[store_as] = result["data"]
                else:
                    all_success = False
            
            return {"success": all_success}
            
        except Exception as e:
            logger.error(f"Parallel step error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _execute_wait_step(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution
    ) -> Dict[str, Any]:
        """Execute wait/polling step"""
        try:
            config = step.config
            wait_time = config.get("wait_time", 60)
            check_endpoint = config.get("check_endpoint")
            success_condition = config.get("success_condition")
            max_attempts = config.get("max_attempts", 10)
            
            for attempt in range(max_attempts):
                if attempt > 0:
                    await asyncio.sleep(wait_time)
                
                if check_endpoint:
                    # Check condition via API
                    endpoint = self._resolve_variables(
                        check_endpoint,
                        execution.execution_data
                    )
                    
                    result = await api_executor.process({
                        "endpoint": endpoint,
                        "method": "GET",
                        "context": {"workflow_id": execution.id}
                    })
                    
                    if result["success"]:
                        response = result["data"].get("response", {})
                        
                        # Check success condition
                        if self._evaluate_condition(success_condition, response):
                            return {"success": True, "data": response}
                else:
                    # Simple wait
                    await asyncio.sleep(wait_time)
                    return {"success": True}
            
            return {"success": False, "error": "Wait condition not met"}
            
        except Exception as e:
            logger.error(f"Wait step error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _execute_transform_step(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution
    ) -> Dict[str, Any]:
        """Execute data transformation step"""
        try:
            config = step.config
            template_name = config["template"]
            data_sources = config["data_sources"]
            
            # Gather data from sources
            combined_data = {}
            for source in data_sources:
                if source in execution.execution_data:
                    combined_data[source] = execution.execution_data[source]
            
            # Apply transformation based on template
            if template_name == "daily_progress":
                transformed = self._transform_daily_progress(combined_data)
            else:
                # Generic transformation
                transformed = combined_data
            
            # Store result
            execution.execution_data["transformed_data"] = transformed
            
            return {"success": True, "data": transformed}
            
        except Exception as e:
            logger.error(f"Transform step error: {e}")
            return {"success": False, "error": str(e)}
    
    async def _execute_notification_step(
        self,
        step: WorkflowStep,
        execution: WorkflowExecution
    ) -> Dict[str, Any]:
        """Execute notification step"""
        try:
            config = step.config
            
            # Resolve recipients
            recipients = self._resolve_variables(
                str(config.get("to", config.get("recipients", ""))),
                execution.execution_data
            )
            
            # Resolve message
            message = self._resolve_variables(
                config["message"],
                execution.execution_data
            )
            
            # Send notification (placeholder - would integrate with notification service)
            logger.info(f"Notification to {recipients}: {message}")
            
            # Store notification record
            execution.execution_data[f"{step.id}_notification"] = {
                "sent_at": datetime.utcnow().isoformat(),
                "recipients": recipients,
                "message": message
            }
            
            return {"success": True}
            
        except Exception as e:
            logger.error(f"Notification step error: {e}")
            return {"success": False, "error": str(e)}
    
    def _resolve_variables(self, template: str, data: Dict[str, Any]) -> str:
        """Resolve variables in template string"""
        import re
        
        def replacer(match):
            var_name = match.group(1)
            return str(data.get(var_name, match.group(0)))
        
        return re.sub(r'\{\{(\w+)\}\}', replacer, str(template))
    
    def _evaluate_condition(self, condition: str, data: Dict[str, Any]) -> bool:
        """Evaluate simple conditions"""
        # Simple implementation - would be more sophisticated in production
        try:
            # Replace variables
            resolved = self._resolve_variables(condition, data)
            
            # Simple comparisons
            if " < " in resolved:
                left, right = resolved.split(" < ")
                return float(left) < float(right)
            elif " > " in resolved:
                left, right = resolved.split(" > ")
                return float(left) > float(right)
            elif " == " in resolved:
                left, right = resolved.split(" == ")
                return left.strip() == right.strip()
            else:
                # Check if variable exists and is truthy
                return bool(data.get(condition))
                
        except Exception as e:
            logger.error(f"Condition evaluation error: {e}")
            return False
    
    async def _determine_next_step(
        self,
        current_step: WorkflowStep,
        result: Dict[str, Any],
        execution: WorkflowExecution
    ) -> Optional[str]:
        """Determine next step based on current step result"""
        # Check if result specifies next step (for conditions)
        if "next_step" in result:
            return result["next_step"]
        
        # Use first next step if available
        if current_step.next_steps:
            return current_step.next_steps[0]
        
        return None
    
    def _extract_order_variables(self, query: str) -> Dict[str, Any]:
        """Extract variables for order workflow from query"""
        # This would use NLP in production
        return {
            "description": "Materials order",
            "items": [],
            "project_id": None,
            "supplier_ids": []
        }
    
    def _extract_report_variables(self, query: str) -> Dict[str, Any]:
        """Extract variables for report workflow from query"""
        return {
            "project_id": None,
            "stakeholder_list": []
        }
    
    async def _build_approval_workflow(self, query: str) -> Dict[str, Any]:
        """Build dynamic approval workflow"""
        # This would create custom workflow based on query
        return None
    
    async def _create_custom_workflow(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create custom workflow from natural language"""
        # This would use AI to generate workflow steps
        return {
            "data": {
                "message": "Custom workflow creation not yet implemented",
                "suggestion": "Try using one of the pre-built workflows: order_creation, daily_report"
            }
        }
    
    def _estimate_duration(self, template: WorkflowTemplate) -> int:
        """Estimate workflow duration in seconds"""
        duration = 0
        
        for step in template.steps:
            if step.type == StepType.WAIT:
                duration += step.config.get("wait_time", 60)
            else:
                duration += 5  # Assume 5 seconds per step
        
        return duration
    
    def _transform_daily_progress(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform data for daily progress report"""
        return {
            "report_date": datetime.utcnow().isoformat(),
            "progress_summary": data.get("progress_data", {}),
            "active_issues": data.get("issues_data", {}),
            "milestone_status": data.get("milestone_data", {}),
            "generated_at": datetime.utcnow().isoformat()
        }

# Create singleton instance
workflow_automation = WorkflowAutomationAgent()