import logging
from typing import Dict, Any, List, Optional, Tuple
import re
from datetime import datetime, timedelta
from enum import Enum

from agents.base_agent import BaseAgent
from storage import mongodb_connector, redis_connector
from config.settings import config

logger = logging.getLogger(__name__)

class ValidationLevel(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class ComplianceType(str, Enum):
    BUSINESS_RULE = "business_rule"
    SAFETY = "safety"
    REGULATORY = "regulatory"
    DATA_QUALITY = "data_quality"
    PERMISSION = "permission"

class ComplianceValidationAgent(BaseAgent):
    """
    The Guardian - ensures all operations follow rules and maintain quality
    """
    
    def __init__(self):
        super().__init__(
            name="compliance",
            description="Validates data, enforces rules, and ensures compliance"
        )
        self.validation_rules = self._load_validation_rules()
        self.business_rules = self._load_business_rules()
        self.safety_regulations = self._load_safety_regulations()
        self.permission_matrix = self._load_permission_matrix()
    
    def _load_validation_rules(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load data validation rules"""
        return {
            "order": [
                {
                    "field": "total",
                    "type": "number",
                    "min": 0,
                    "max": 10000000,
                    "required": True,
                    "message": "Order total must be between 0 and 10,000,000"
                },
                {
                    "field": "items",
                    "type": "array",
                    "min_length": 1,
                    "required": True,
                    "message": "Order must contain at least one item"
                },
                {
                    "field": "project_id",
                    "type": "reference",
                    "entity": "project",
                    "required": True,
                    "message": "Valid project ID required"
                }
            ],
            "project": [
                {
                    "field": "start_date",
                    "type": "date",
                    "min": "today",
                    "required": True,
                    "message": "Project start date cannot be in the past"
                },
                {
                    "field": "end_date",
                    "type": "date",
                    "after_field": "start_date",
                    "required": True,
                    "message": "Project end date must be after start date"
                },
                {
                    "field": "budget",
                    "type": "number",
                    "min": 1000,
                    "required": True,
                    "message": "Project budget must be at least 1,000"
                }
            ],
            "user": [
                {
                    "field": "email",
                    "type": "email",
                    "required": True,
                    "unique": True,
                    "message": "Valid unique email address required"
                },
                {
                    "field": "role",
                    "type": "enum",
                    "values": ["admin", "manager", "supervisor", "worker"],
                    "required": True,
                    "message": "Valid user role required"
                }
            ]
        }
    
    def _load_business_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load business rules and thresholds"""
        return {
            "approval_limits": {
                "worker": 1000,
                "supervisor": 5000,
                "manager": 25000,
                "admin": 100000,
                "God": float('inf')  # No limit
            },
            "order_rules": {
                "max_items_per_order": 100,
                "require_quotes_above": 5000,
                "min_quotes_required": 3,
                "approval_escalation_threshold": 50000
            },
            "project_rules": {
                "max_duration_days": 730,  # 2 years
                "min_team_size": 2,
                "require_safety_officer_above_size": 10,
                "milestone_interval_days": 30
            },
            "workflow_rules": {
                "max_parallel_workflows": 10,
                "max_workflow_duration_hours": 72,
                "require_approval_for_automation": True
            }
        }
    
    def _load_safety_regulations(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load safety and regulatory compliance rules"""
        return {
            "ppe_requirements": [
                {
                    "location": "construction_site",
                    "required": ["hard_hat", "safety_vest", "steel_toe_boots"],
                    "enforcement": "mandatory"
                },
                {
                    "location": "heights",
                    "height_threshold": 6,  # feet
                    "required": ["harness", "lanyard"],
                    "enforcement": "mandatory"
                }
            ],
            "inspection_requirements": [
                {
                    "type": "scaffolding",
                    "frequency_days": 7,
                    "inspector_certification": "scaffold_competent",
                    "documentation": "required"
                },
                {
                    "type": "excavation",
                    "depth_threshold": 5,  # feet
                    "frequency": "daily",
                    "documentation": "required"
                }
            ],
            "permit_requirements": [
                {
                    "activity": "hot_work",
                    "permit_type": "hot_work_permit",
                    "validity_hours": 8,
                    "approver": "safety_officer"
                },
                {
                    "activity": "confined_space",
                    "permit_type": "confined_space_permit",
                    "validity_hours": 12,
                    "approver": "safety_officer"
                }
            ]
        }
    
    def _load_permission_matrix(self) -> Dict[str, Dict[str, List[str]]]:
        """Load role-based permission matrix"""
        return {
            "worker": {
                "allowed_actions": ["view", "create_timesheet", "report_issue"],
                "denied_actions": ["approve", "delete", "modify_budget"]
            },
            "supervisor": {
                "allowed_actions": ["view", "create", "update", "approve_limited"],
                "denied_actions": ["delete_project", "modify_permissions"]
            },
            "manager": {
                "allowed_actions": ["view", "create", "update", "approve", "delete_limited"],
                "denied_actions": ["modify_system_settings"]
            },
            "admin": {
                "allowed_actions": ["all_except_system"],
                "denied_actions": ["modify_core_system"]
            },
            "God": {
                "allowed_actions": ["all"],
                "denied_actions": []
            }
        }
    
    async def validate_request(self, request: Dict[str, Any]) -> bool:
        """Validate compliance request"""
        return True  # Compliance agent validates everything
    
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process validation and compliance checks
        """
        try:
            context = request.get("context", {})
            parameters = request.get("parameters", {})
            
            # Determine validation type
            if parameters.get("validate_only"):
                # Pre-execution validation
                return await self._validate_request(request)
            
            elif "data" in request:
                # Data validation
                return await self._validate_data(
                    request["data"],
                    request.get("entity_type"),
                    context
                )
            
            elif "action" in request:
                # Permission validation
                return await self._validate_permission(
                    request["action"],
                    context
                )
            
            elif "workflow" in request:
                # Workflow compliance
                return await self._validate_workflow(
                    request["workflow"],
                    context
                )
            
            else:
                # General compliance check
                return await self._perform_compliance_check(request)
                
        except Exception as e:
            logger.error(f"Compliance validation error: {e}", exc_info=True)
            raise
    
    async def _validate_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Validate request before execution"""
        violations = []
        warnings = []
        
        query = request.get("query", "")
        context = request.get("context", {})
        
        # Check for sensitive data exposure
        sensitive_patterns = [
            r'\b(?:password|pwd|pass)\b',
            r'\b(?:ssn|social.?security)\b',
            r'\b(?:credit.?card|cc.?num)\b',
            r'\b(?:api.?key|secret)\b'
        ]
        
        for pattern in sensitive_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                violations.append({
                    "type": ComplianceType.DATA_QUALITY,
                    "level": ValidationLevel.CRITICAL,
                    "message": "Query contains potentially sensitive information",
                    "field": "query"
                })
        
        # Check for SQL injection patterns
        sql_patterns = [
            r';\s*(?:drop|delete|truncate|update)\s+',
            r'(?:union|select).+(?:from|where)',
            r'(?:or|and)\s+["\']?\d+["\']?\s*=\s*["\']?\d+',
        ]
        
        for pattern in sql_patterns:
            if re.search(pattern, query, re.IGNORECASE):
                violations.append({
                    "type": ComplianceType.DATA_QUALITY,
                    "level": ValidationLevel.CRITICAL,
                    "message": "Query contains potentially malicious patterns",
                    "field": "query"
                })
        
        # Check rate limits
        user_id = context.get("user_id")
        if user_id:
            rate_limit_ok = await redis_connector.check_rate_limit(
                f"compliance:{user_id}",
                limit=1000,
                window=3600
            )
            
            if not rate_limit_ok:
                violations.append({
                    "type": ComplianceType.BUSINESS_RULE,
                    "level": ValidationLevel.HIGH,
                    "message": "Rate limit exceeded for compliance checks",
                    "field": "rate_limit"
                })
        
        # Check time-based restrictions
        current_hour = datetime.utcnow().hour
        if current_hour < 6 or current_hour > 22:  # Outside business hours
            warnings.append({
                "type": ComplianceType.BUSINESS_RULE,
                "level": ValidationLevel.LOW,
                "message": "Request made outside normal business hours",
                "field": "timestamp"
            })
        
        success = len(violations) == 0
        
        return {
            "data": {
                "valid": success,
                "violations": violations,
                "warnings": warnings,
                "checked_at": datetime.utcnow().isoformat()
            },
            "success": success,
            "metadata": {
                "validation_type": "pre_execution",
                "rules_checked": len(sensitive_patterns) + len(sql_patterns) + 2
            }
        }
    
    async def _validate_data(
        self,
        data: Dict[str, Any],
        entity_type: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate data against rules"""
        violations = []
        warnings = []
        
        # Get validation rules for entity type
        rules = self.validation_rules.get(entity_type, [])
        
        for rule in rules:
            field_name = rule["field"]
            field_value = data.get(field_name)
            
            # Check required fields
            if rule.get("required") and field_value is None:
                violations.append({
                    "type": ComplianceType.DATA_QUALITY,
                    "level": ValidationLevel.HIGH,
                    "message": rule.get("message", f"{field_name} is required"),
                    "field": field_name
                })
                continue
            
            if field_value is not None:
                # Type validation
                if not self._validate_field_type(field_value, rule["type"]):
                    violations.append({
                        "type": ComplianceType.DATA_QUALITY,
                        "level": ValidationLevel.MEDIUM,
                        "message": f"{field_name} has invalid type",
                        "field": field_name
                    })
                
                # Range validation
                if rule["type"] == "number":
                    if "min" in rule and field_value < rule["min"]:
                        violations.append({
                            "type": ComplianceType.DATA_QUALITY,
                            "level": ValidationLevel.MEDIUM,
                            "message": rule.get("message", f"{field_name} below minimum"),
                            "field": field_name
                        })
                    
                    if "max" in rule and field_value > rule["max"]:
                        violations.append({
                            "type": ComplianceType.DATA_QUALITY,
                            "level": ValidationLevel.MEDIUM,
                            "message": rule.get("message", f"{field_name} exceeds maximum"),
                            "field": field_name
                        })
                
                # Date validation
                elif rule["type"] == "date":
                    date_valid, date_message = self._validate_date_field(
                        field_value,
                        rule,
                        data
                    )
                    if not date_valid:
                        violations.append({
                            "type": ComplianceType.DATA_QUALITY,
                            "level": ValidationLevel.MEDIUM,
                            "message": date_message,
                            "field": field_name
                        })
                
                # Array validation
                elif rule["type"] == "array":
                    if "min_length" in rule and len(field_value) < rule["min_length"]:
                        violations.append({
                            "type": ComplianceType.DATA_QUALITY,
                            "level": ValidationLevel.MEDIUM,
                            "message": rule.get("message", f"{field_name} too short"),
                            "field": field_name
                        })
        
        # Business rule validation
        business_violations = await self._validate_business_rules(
            data,
            entity_type,
            context
        )
        violations.extend(business_violations)
        
        # Audit trail
        await self._create_audit_entry(
            "data_validation",
            entity_type,
            context,
            {
                "violations": len(violations),
                "warnings": len(warnings)
            }
        )
        
        success = len(violations) == 0
        
        return {
            "data": {
                "valid": success,
                "violations": violations,
                "warnings": warnings,
                "entity_type": entity_type,
                "validated_at": datetime.utcnow().isoformat()
            },
            "success": success
        }
    
    async def _validate_permission(
        self,
        action: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate user permissions for action"""
        user_role = context.get("user_role", "worker").lower()
        user_id = context.get("user_id")
        
        # Get permission matrix for role
        permissions = self.permission_matrix.get(user_role, {})
        allowed_actions = permissions.get("allowed_actions", [])
        denied_actions = permissions.get("denied_actions", [])
        
        # Check if action is explicitly denied
        if action in denied_actions:
            return {
                "success": False,
                "data": {
                    "allowed": False,
                    "reason": f"Action '{action}' is not permitted for role '{user_role}'",
                    "required_role": self._get_minimum_role_for_action(action)
                }
            }
        
        # Check if action is allowed
        if "all" in allowed_actions or action in allowed_actions:
            # Additional checks for specific actions
            if action.startswith("approve"):
                return await self._validate_approval_permission(
                    action,
                    context,
                    user_role
                )
            
            return {
                "success": True,
                "data": {
                    "allowed": True,
                    "role": user_role,
                    "action": action
                }
            }
        
        # Check pattern-based permissions
        for allowed_pattern in allowed_actions:
            if allowed_pattern.endswith("*") and action.startswith(allowed_pattern[:-1]):
                return {
                    "success": True,
                    "data": {
                        "allowed": True,
                        "role": user_role,
                        "action": action,
                        "matched_pattern": allowed_pattern
                    }
                }
        
        # Default deny
        return {
            "success": False,
            "data": {
                "allowed": False,
                "reason": f"Insufficient permissions for action '{action}'",
                "user_role": user_role
            }
        }
    
    async def _validate_workflow(
        self,
        workflow: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate workflow compliance"""
        violations = []
        warnings = []
        
        rules = self.business_rules["workflow_rules"]
        
        # Check workflow complexity
        step_count = len(workflow.get("steps", []))
        if step_count > 50:
            warnings.append({
                "type": ComplianceType.BUSINESS_RULE,
                "level": ValidationLevel.MEDIUM,
                "message": "Workflow has many steps, consider breaking into sub-workflows",
                "field": "steps"
            })
        
        # Check parallel execution limits
        parallel_steps = sum(
            1 for step in workflow.get("steps", [])
            if step.get("type") == "parallel"
        )
        
        if parallel_steps > rules["max_parallel_workflows"]:
            violations.append({
                "type": ComplianceType.BUSINESS_RULE,
                "level": ValidationLevel.HIGH,
                "message": f"Workflow exceeds parallel execution limit of {rules['max_parallel_workflows']}",
                "field": "parallel_steps"
            })
        
        # Check automation approval
        if rules["require_approval_for_automation"]:
            if not workflow.get("approved_by"):
                violations.append({
                    "type": ComplianceType.BUSINESS_RULE,
                    "level": ValidationLevel.HIGH,
                    "message": "Automated workflows require approval",
                    "field": "approval"
                })
        
        # Validate individual steps
        for i, step in enumerate(workflow.get("steps", [])):
            step_violations = self._validate_workflow_step(step, i)
            violations.extend(step_violations)
        
        success = len(violations) == 0
        
        return {
            "data": {
                "valid": success,
                "violations": violations,
                "warnings": warnings,
                "workflow_name": workflow.get("name", "Unknown"),
                "validated_at": datetime.utcnow().isoformat()
            },
            "success": success
        }
    
    async def _validate_business_rules(
        self,
        data: Dict[str, Any],
        entity_type: str,
        context: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Validate against business rules"""
        violations = []
        
        if entity_type == "order":
            rules = self.business_rules["order_rules"]
            
            # Check approval limits
            total = data.get("total", 0)
            user_role = context.get("user_role", "worker")
            approval_limit = self.business_rules["approval_limits"].get(user_role, 0)
            
            if total > approval_limit:
                violations.append({
                    "type": ComplianceType.BUSINESS_RULE,
                    "level": ValidationLevel.HIGH,
                    "message": f"Order total ${total} exceeds approval limit of ${approval_limit} for {user_role}",
                    "field": "total",
                    "requires_escalation": True
                })
            
            # Check quotes requirement
            if total > rules["require_quotes_above"]:
                quotes = data.get("quotes", [])
                if len(quotes) < rules["min_quotes_required"]:
                    violations.append({
                        "type": ComplianceType.BUSINESS_RULE,
                        "level": ValidationLevel.MEDIUM,
                        "message": f"Orders over ${rules['require_quotes_above']} require at least {rules['min_quotes_required']} quotes",
                        "field": "quotes"
                    })
            
            # Check item count
            items = data.get("items", [])
            if len(items) > rules["max_items_per_order"]:
                violations.append({
                    "type": ComplianceType.BUSINESS_RULE,
                    "level": ValidationLevel.LOW,
                    "message": f"Order has {len(items)} items, exceeds recommended maximum of {rules['max_items_per_order']}",
                    "field": "items"
                })
        
        elif entity_type == "project":
            rules = self.business_rules["project_rules"]
            
            # Check project duration
            start_date = data.get("start_date")
            end_date = data.get("end_date")
            
            if start_date and end_date:
                duration = (
                    datetime.fromisoformat(end_date) - 
                    datetime.fromisoformat(start_date)
                ).days
                
                if duration > rules["max_duration_days"]:
                    violations.append({
                        "type": ComplianceType.BUSINESS_RULE,
                        "level": ValidationLevel.MEDIUM,
                        "message": f"Project duration of {duration} days exceeds maximum of {rules['max_duration_days']}",
                        "field": "duration"
                    })
            
            # Check team size requirements
            team_size = len(data.get("team_members", []))
            if team_size < rules["min_team_size"]:
                violations.append({
                    "type": ComplianceType.BUSINESS_RULE,
                    "level": ValidationLevel.HIGH,
                    "message": f"Project requires minimum team size of {rules['min_team_size']}",
                    "field": "team_members"
                })
            
            # Check safety officer requirement
            if team_size > rules["require_safety_officer_above_size"]:
                has_safety_officer = any(
                    member.get("role") == "safety_officer"
                    for member in data.get("team_members", [])
                )
                
                if not has_safety_officer:
                    violations.append({
                        "type": ComplianceType.SAFETY,
                        "level": ValidationLevel.CRITICAL,
                        "message": f"Projects with {team_size} team members require a safety officer",
                        "field": "team_members"
                    })
        
        return violations
    
    async def _validate_approval_permission(
        self,
        action: str,
        context: Dict[str, Any],
        user_role: str
    ) -> Dict[str, Any]:
        """Validate approval permissions with limits"""
        # Extract approval amount if present
        amount_match = re.search(r'approve.*?(\d+(?:\.\d+)?)', action)
        
        if amount_match:
            amount = float(amount_match.group(1))
            approval_limit = self.business_rules["approval_limits"].get(user_role, 0)
            
            if amount > approval_limit:
                return {
                    "success": False,
                    "data": {
                        "allowed": False,
                        "reason": f"Approval amount ${amount} exceeds limit of ${approval_limit} for {user_role}",
                        "current_limit": approval_limit,
                        "required_amount": amount
                    }
                }
        
        return {
            "success": True,
            "data": {
                "allowed": True,
                "role": user_role,
                "action": action,
                "approval_limit": self.business_rules["approval_limits"].get(user_role, 0)
            }
        }
    
    def _validate_field_type(self, value: Any, expected_type: str) -> bool:
        """Validate field type"""
        type_validators = {
            "string": lambda v: isinstance(v, str),
            "number": lambda v: isinstance(v, (int, float)),
            "boolean": lambda v: isinstance(v, bool),
            "array": lambda v: isinstance(v, list),
            "object": lambda v: isinstance(v, dict),
            "email": lambda v: isinstance(v, str) and "@" in v and "." in v.split("@")[1],
            "date": lambda v: self._is_valid_date(v),
            "enum": lambda v: True,  # Handled separately
            "reference": lambda v: True  # Would check against database
        }
        
        validator = type_validators.get(expected_type, lambda v: True)
        return validator(value)
    
    def _is_valid_date(self, value: str) -> bool:
        """Check if string is valid ISO date"""
        try:
            datetime.fromisoformat(value.replace('Z', '+00:00'))
            return True
        except:
            return False
    
    def _validate_date_field(
        self,
        value: str,
        rule: Dict[str, Any],
        data: Dict[str, Any]
    ) -> Tuple[bool, str]:
        """Validate date field with constraints"""
        try:
            date_value = datetime.fromisoformat(value.replace('Z', '+00:00'))
            
            # Check minimum date
            if "min" in rule:
                if rule["min"] == "today":
                    min_date = datetime.utcnow()
                else:
                    min_date = datetime.fromisoformat(rule["min"])
                
                if date_value < min_date:
                    return False, rule.get("message", "Date is before minimum allowed")
            
            # Check maximum date
            if "max" in rule:
                if rule["max"] == "today":
                    max_date = datetime.utcnow()
                else:
                    max_date = datetime.fromisoformat(rule["max"])
                
                if date_value > max_date:
                    return False, rule.get("message", "Date is after maximum allowed")
            
            # Check relative to another field
            if "after_field" in rule:
                other_field = data.get(rule["after_field"])
                if other_field:
                    other_date = datetime.fromisoformat(other_field.replace('Z', '+00:00'))
                    if date_value <= other_date:
                        return False, rule.get("message", f"Date must be after {rule['after_field']}")
            
            return True, ""
            
        except Exception as e:
            return False, f"Invalid date format: {str(e)}"
    
    def _validate_workflow_step(
        self,
        step: Dict[str, Any],
        index: int
    ) -> List[Dict[str, Any]]:
        """Validate individual workflow step"""
        violations = []
        
        # Check required fields
        if not step.get("id"):
            violations.append({
                "type": ComplianceType.DATA_QUALITY,
                "level": ValidationLevel.HIGH,
                "message": f"Step {index} missing required 'id' field",
                "field": f"steps[{index}].id"
            })
        
        if not step.get("type"):
            violations.append({
                "type": ComplianceType.DATA_QUALITY,
                "level": ValidationLevel.HIGH,
                "message": f"Step {index} missing required 'type' field",
                "field": f"steps[{index}].type"
            })
        
        # Validate step configuration
        step_type = step.get("type")
        config = step.get("config", {})
        
        if step_type == "api_call" and not config.get("endpoint"):
            violations.append({
                "type": ComplianceType.DATA_QUALITY,
                "level": ValidationLevel.HIGH,
                "message": f"API call step {index} missing endpoint",
                "field": f"steps[{index}].config.endpoint"
            })
        
        elif step_type == "condition" and not config.get("conditions"):
            violations.append({
                "type": ComplianceType.DATA_QUALITY,
                "level": ValidationLevel.HIGH,
                "message": f"Condition step {index} missing conditions",
                "field": f"steps[{index}].config.conditions"
            })
        
        return violations
    
    async def _perform_compliance_check(
        self,
        request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Perform general compliance check"""
        context = request.get("context", {})
        check_type = request.get("check_type", "general")
        
        results = {
            "safety_compliance": await self._check_safety_compliance(context),
            "regulatory_compliance": await self._check_regulatory_compliance(context),
            "data_quality": await self._check_data_quality(context),
            "audit_status": await self._check_audit_status(context)
        }
        
        # Calculate overall compliance score
        total_checks = sum(r["checks_performed"] for r in results.values())
        total_passed = sum(r["checks_passed"] for r in results.values())
        
        compliance_score = (total_passed / total_checks * 100) if total_checks > 0 else 100
        
        return {
            "data": {
                "compliance_score": compliance_score,
                "results": results,
                "status": "compliant" if compliance_score >= 80 else "non_compliant",
                "checked_at": datetime.utcnow().isoformat()
            }
        }
    
    async def _check_safety_compliance(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check safety compliance"""
        violations = []
        checks_performed = 0
        checks_passed = 0
        
        # Check PPE compliance
        if context.get("location") == "construction_site":
            checks_performed += 1
            # Check actual PPE status from context
            if context.get("ppe_compliant", False):  # Default to False for safety
                checks_passed += 1
            else:
                violations.append({
                    "type": "ppe_violation",
                    "message": "Required PPE not worn",
                    "level": ValidationLevel.CRITICAL
                })
        
        # Check inspection status
        if context.get("project_id"):
            checks_performed += 1
            # Check actual inspection records from context
            last_inspection = context.get("last_inspection_days", 999)  # Default to overdue
            if last_inspection <= 7:
                checks_passed += 1
            else:
                violations.append({
                    "type": "inspection_overdue",
                    "message": f"Safety inspection overdue by {last_inspection - 7} days",
                    "level": ValidationLevel.HIGH
                })
        
        return {
            "checks_performed": checks_performed,
            "checks_passed": checks_passed,
            "violations": violations
        }
    
    async def _check_regulatory_compliance(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check regulatory compliance"""
        violations = []
        checks_performed = 0
        checks_passed = 0
        
        # Check permits
        if context.get("activity_type") in ["hot_work", "confined_space", "excavation"]:
            checks_performed += 1
            # This would check actual permit status
            if context.get("permit_valid", True):
                checks_passed += 1
            else:
                violations.append({
                    "type": "permit_violation",
                    "message": f"Valid permit required for {context['activity_type']}",
                    "level": ValidationLevel.CRITICAL
                })
        
        # Check certifications
        if context.get("requires_certification"):
            checks_performed += 1
            # Check user certifications
            if context.get("user_certified", True):
                checks_passed += 1
            else:
                violations.append({
                    "type": "certification_missing",
                    "message": "Required certification not found",
                    "level": ValidationLevel.HIGH
                })
        
        return {
            "checks_performed": checks_performed,
            "checks_passed": checks_passed,
            "violations": violations
        }
    
    async def _check_data_quality(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check data quality metrics"""
        from api.cms_client import cms_client
        
        violations = []
        checks_performed = 0
        checks_passed = 0
        
        try:
            # Check actual data quality from CMS
            if "project_id" in context:
                checks_performed += 1
                project_response = await cms_client.get(f"/api/projects/{context['project_id']}")
                
                # Check for required fields
                required_fields = ["name", "start_date", "end_date", "budget", "status"]
                missing_fields = []
                
                if project_response:
                    for field in required_fields:
                        if field not in project_response or not project_response[field]:
                            missing_fields.append(field)
                    
                    if missing_fields:
                        violations.append({
                            "type": "incomplete_data",
                            "message": f"Project missing required fields: {', '.join(missing_fields)}",
                            "level": ValidationLevel.MEDIUM
                        })
                    else:
                        checks_passed += 1
            
            # Check order data quality
            if "order_id" in context:
                checks_performed += 1
                order_response = await cms_client.get(f"/api/orders/{context['order_id']}")
                
                if order_response:
                    # Check for data completeness
                    if order_response.get("items") and len(order_response["items"]) > 0:
                        checks_passed += 1
                    else:
                        violations.append({
                            "type": "incomplete_data",
                            "message": "Order has no line items",
                            "level": ValidationLevel.HIGH
                        })
            
            # Check user data quality
            if "user_id" in context:
                checks_performed += 1
                user_response = await cms_client.get(f"/api/users/{context['user_id']}")
                
                if user_response:
                    # Check for complete profile
                    if user_response.get("email") and user_response.get("name"):
                        checks_passed += 1
                    else:
                        violations.append({
                            "type": "incomplete_profile",
                            "message": "User profile incomplete",
                            "level": ValidationLevel.LOW
                        })
            
            # If no specific context, do general quality check
            if checks_performed == 0:
                checks_performed = 1
                checks_passed = 1  # Assume good quality by default
                
        except Exception as e:
            logger.error(f"Failed to check data quality: {e}")
            checks_performed = 1
            violations.append({
                "type": "quality_check_failed",
                "message": "Unable to verify data quality",
                "level": ValidationLevel.LOW
            })
        
        return {
            "checks_performed": checks_performed,
            "checks_passed": checks_passed,
            "violations": violations
        }
    
    async def _check_audit_status(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Check audit trail completeness"""
        violations = []
        checks_performed = 0
        checks_passed = 0
        
        # Check if audit logging is enabled
        checks_performed += 1
        if context.get("audit_enabled", True):
            checks_passed += 1
        else:
            violations.append({
                "type": "audit_disabled",
                "message": "Audit logging is disabled",
                "level": ValidationLevel.HIGH
            })
        
        # Check audit retention period
        checks_performed += 1
        retention_days = context.get("audit_retention_days", 90)
        if retention_days >= 90:  # Minimum 90 days retention
            checks_passed += 1
        else:
            violations.append({
                "type": "insufficient_retention",
                "message": f"Audit retention period too short: {retention_days} days",
                "level": ValidationLevel.MEDIUM
            })
        
        # Check recent audit entries
        checks_performed += 1
        last_audit_hours = context.get("hours_since_last_audit", 0)
        if last_audit_hours <= 24:  # Should have recent audit entries
            checks_passed += 1
        else:
            violations.append({
                "type": "audit_gap",
                "message": f"No audit entries for {last_audit_hours} hours",
                "level": ValidationLevel.LOW
            })
        
        return {
            "checks_performed": checks_performed,
            "checks_passed": checks_passed,
            "violations": violations
        }
    
    async def _create_audit_entry(
        self,
        action: str,
        entity_type: str,
        context: Dict[str, Any],
        result: Dict[str, Any]
    ):
        """Create audit trail entry"""
        audit_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "entity_type": entity_type,
            "user_id": context.get("user_id"),
            "user_role": context.get("user_role"),
            "result": result,
            "success": result.get("violations", 0) == 0
        }
        
        # Store in MongoDB
        await mongodb_connector.find_one(
            "audit_log",
            audit_entry
        )
        
        # Cache recent audits in Redis
        await redis_connector.set_with_ttl(
            f"audit:{action}:{entity_type}:{datetime.utcnow().timestamp()}",
            audit_entry,
            86400  # 24 hours
        )
    
    def _get_minimum_role_for_action(self, action: str) -> str:
        """Get minimum role required for action"""
        # Check each role from lowest to highest
        role_hierarchy = ["worker", "supervisor", "manager", "admin", "God"]
        
        for role in role_hierarchy:
            permissions = self.permission_matrix.get(role, {})
            allowed = permissions.get("allowed_actions", [])
            
            if action in allowed or "all" in allowed:
                return role
        
        return "God"  # Default to highest role

# Create singleton instance
compliance_validation = ComplianceValidationAgent()