import logging
from typing import Dict, Any, List, Optional, Tuple
import asyncio
from datetime import datetime, timedelta, date
import numpy as np
from collections import defaultdict

from agents.base_agent import BaseAgent
from agents.api_executor import api_executor
from storage import mongodb_connector, redis_connector
from models import (
    AnalyticsQuery, AnalyticsReport, KPIMetric,
    MetricType, TimeGranularity, ProjectAnalytics
)
from config.settings import config

logger = logging.getLogger(__name__)

class AnalyticsInsightsAgent(BaseAgent):
    """
    The Data Scientist - turns raw data into actionable insights
    """
    
    def __init__(self):
        super().__init__(
            name="analytics",
            description="Generates analytics reports and business insights"
        )
        self.kpi_definitions = self._load_kpi_definitions()
        self.report_templates = self._load_report_templates()
    
    def _load_kpi_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Load KPI calculation definitions"""
        return {
            "productivity": {
                "name": "Productivity Score",
                "unit": "tasks/hour",
                "calculation": "completed_tasks / work_hours",
                "data_sources": ["timesheets", "task_completions"]
            },
            "budget_variance": {
                "name": "Budget Variance",
                "unit": "%",
                "calculation": "(actual_cost - budgeted_cost) / budgeted_cost * 100",
                "data_sources": ["budgets", "invoices", "purchase_orders"]
            },
            "schedule_adherence": {
                "name": "Schedule Adherence",
                "unit": "%",
                "calculation": "on_time_milestones / total_milestones * 100",
                "data_sources": ["milestones", "project_timeline"]
            },
            "quality_score": {
                "name": "Quality Score",
                "unit": "score",
                "calculation": "(passed_inspections - defects) / total_inspections * 100",
                "data_sources": ["quality_assessments", "defect_reports"]
            },
            "resource_utilization": {
                "name": "Resource Utilization",
                "unit": "%",
                "calculation": "utilized_hours / available_hours * 100",
                "data_sources": ["resource_allocation", "timesheets"]
            },
            "order_efficiency": {
                "name": "Order Processing Efficiency",
                "unit": "days",
                "calculation": "average(order_completion_time)",
                "data_sources": ["orders", "order_history"]
            }
        }
    
    def _load_report_templates(self) -> Dict[str, Dict[str, Any]]:
        """Load report generation templates"""
        return {
            "project_dashboard": {
                "name": "Project Dashboard",
                "metrics": ["budget_variance", "schedule_adherence", "quality_score"],
                "visualizations": ["budget_burndown", "milestone_timeline", "resource_chart"],
                "frequency": "daily"
            },
            "weekly_summary": {
                "name": "Weekly Performance Summary",
                "metrics": ["productivity", "resource_utilization", "order_efficiency"],
                "visualizations": ["trend_charts", "comparison_tables"],
                "frequency": "weekly"
            },
            "executive_report": {
                "name": "Executive Summary",
                "metrics": ["all_kpis"],
                "visualizations": ["kpi_dashboard", "trend_analysis", "risk_matrix"],
                "frequency": "monthly"
            }
        }
    
    async def validate_request(self, request: Dict[str, Any]) -> bool:
        """Validate analytics request"""
        return "query" in request or "report_type" in request or "metrics" in request
    
    async def process(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process analytics request - generate reports, calculate KPIs, provide insights
        """
        try:
            context = request.get("context", {})
            parameters = request.get("parameters", {})
            
            # Detect analytics intent
            if "report_type" in request:
                # Generate specific report
                return await self._generate_report(
                    request["report_type"],
                    context,
                    parameters
                )
            
            elif "metrics" in request:
                # Calculate specific metrics
                return await self._calculate_metrics(
                    request["metrics"],
                    context,
                    parameters
                )
            
            else:
                # Natural language analytics query
                query = request.get("query", "")
                analytics_intent = await self._detect_analytics_intent(query)
                
                return await self._process_analytics_query(
                    analytics_intent,
                    context,
                    parameters
                )
                
        except Exception as e:
            logger.error(f"Analytics error: {e}", exc_info=True)
            raise
    
    async def _detect_analytics_intent(self, query: str) -> Dict[str, Any]:
        """Detect what analytics are being requested"""
        query_lower = query.lower()
        intent = {
            "type": None,
            "entity": None,
            "time_range": None,
            "metrics": [],
            "grouping": None
        }
        
        # Detect analytics type
        if any(word in query_lower for word in ["kpi", "performance", "metric"]):
            intent["type"] = "kpi"
        elif any(word in query_lower for word in ["trend", "over time", "history"]):
            intent["type"] = "trend"
        elif any(word in query_lower for word in ["compare", "comparison", "versus"]):
            intent["type"] = "comparison"
        elif any(word in query_lower for word in ["forecast", "predict", "projection"]):
            intent["type"] = "forecast"
        else:
            intent["type"] = "summary"
        
        # Detect entity
        if "project" in query_lower:
            intent["entity"] = "projects"
        elif "order" in query_lower:
            intent["entity"] = "orders"
        elif "user" in query_lower or "team" in query_lower:
            intent["entity"] = "users"
        elif "budget" in query_lower or "cost" in query_lower:
            intent["entity"] = "financial"
        
        # Detect time range
        if "today" in query_lower:
            intent["time_range"] = "today"
        elif "this week" in query_lower:
            intent["time_range"] = "week"
        elif "this month" in query_lower:
            intent["time_range"] = "month"
        elif "last" in query_lower:
            intent["time_range"] = "previous_period"
        
        # Detect specific metrics
        for kpi_name, kpi_def in self.kpi_definitions.items():
            if any(word in query_lower for word in kpi_name.split("_")):
                intent["metrics"].append(kpi_name)
        
        return intent
    
    async def _generate_report(
        self,
        report_type: str,
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate a specific report type"""
        template = self.report_templates.get(report_type)
        if not template:
            raise ValueError(f"Unknown report type: {report_type}")
        
        # Collect data for report
        report_data = {
            "title": template["name"],
            "generated_at": datetime.utcnow(),
            "period": parameters.get("period", self._get_default_period(template)),
            "metrics": {},
            "visualizations": {},
            "insights": []
        }
        
        # Calculate metrics
        metrics_to_calculate = template["metrics"]
        if "all_kpis" in metrics_to_calculate:
            metrics_to_calculate = list(self.kpi_definitions.keys())
        
        for metric_name in metrics_to_calculate:
            metric_data = await self._calculate_single_metric(
                metric_name,
                context,
                report_data["period"]
            )
            report_data["metrics"][metric_name] = metric_data
        
        # Generate visualizations
        for viz_type in template["visualizations"]:
            viz_data = await self._generate_visualization(
                viz_type,
                report_data["metrics"],
                report_data["period"]
            )
            report_data["visualizations"][viz_type] = viz_data
        
        # Generate insights
        insights = await self._generate_insights(report_data["metrics"])
        report_data["insights"] = insights
        
        # Save report
        report = AnalyticsReport(
            id="",
            name=template["name"],
            description=f"{report_type} report for {report_data['period']['name']}",
            user_id=context.get("user_id", 0),
            query=AnalyticsQuery(
                metric_type=MetricType.DISTRIBUTION,
                entity_type=context.get("entity_type", "projects"),
                time_range=report_data["period"]
            ),
            results=report_data,
            visualizations=list(report_data["visualizations"].values())
        )
        
        report_id = await mongodb_connector.save_analytics_report(report.dict())
        
        return {
            "data": {
                "report_id": report_id,
                "report": report_data,
                "message": f"{template['name']} generated successfully",
                "key_insights": insights[:3]  # Top 3 insights
            }
        }
    
    async def _calculate_metrics(
        self,
        metrics: List[str],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate specific metrics"""
        results = {}
        period = parameters.get("period", self._get_default_period())
        
        for metric_name in metrics:
            if metric_name in self.kpi_definitions:
                metric_data = await self._calculate_single_metric(
                    metric_name,
                    context,
                    period
                )
                results[metric_name] = metric_data
        
        return {
            "data": {
                "metrics": results,
                "period": period,
                "message": f"Calculated {len(results)} metrics"
            }
        }
    
    async def _process_analytics_query(
        self,
        intent: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process natural language analytics query"""
        # Build analytics query
        query = AnalyticsQuery(
            metric_type=self._map_intent_to_metric_type(intent["type"]),
            entity_type=intent["entity"] or "projects",
            filters=context.get("filters", {}),
            time_range=self._parse_time_range(intent["time_range"]),
            granularity=parameters.get("granularity", TimeGranularity.DAILY)
        )
        
        # Execute query based on type
        if intent["type"] == "kpi":
            results = await self._calculate_kpis(intent["metrics"], query)
        elif intent["type"] == "trend":
            results = await self._analyze_trends(intent["metrics"], query)
        elif intent["type"] == "comparison":
            results = await self._generate_comparisons(query)
        elif intent["type"] == "forecast":
            results = await self._generate_forecast(intent["metrics"], query)
        else:
            results = await self._generate_summary(query)
        
        return {
            "data": results,
            "metadata": {
                "query_type": intent["type"],
                "time_range": query.time_range
            }
        }
    
    async def _calculate_single_metric(
        self,
        metric_name: str,
        context: Dict[str, Any],
        period: Dict[str, Any]
    ) -> KPIMetric:
        """Calculate a single KPI metric"""
        kpi_def = self.kpi_definitions.get(metric_name, {})
        
        # Fetch required data
        data = {}
        for source in kpi_def.get("data_sources", []):
            source_data = await self._fetch_data_source(source, context, period)
            data[source] = source_data
        
        # Calculate metric value
        value = await self._perform_calculation(
            kpi_def["calculation"],
            data
        )
        
        # Calculate trend
        previous_value = await self._calculate_previous_period(
            metric_name,
            context,
            period
        )
        
        change_percentage = None
        if previous_value and previous_value > 0:
            change_percentage = ((value - previous_value) / previous_value) * 100
        
        # Get historical trend
        trend = await self._get_metric_trend(metric_name, context, period)
        
        return KPIMetric(
            name=kpi_def["name"],
            value=value,
            unit=kpi_def.get("unit"),
            change_percentage=change_percentage,
            trend=trend
        )
    
    async def _fetch_data_source(
        self,
        source: str,
        context: Dict[str, Any],
        period: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fetch data from various sources"""
        # Map source to API endpoint
        endpoint_map = {
            "timesheets": "/api/timesheets",
            "task_completions": "/api/tasks/completed",
            "budgets": "/api/budgets",
            "invoices": "/api/invoices",
            "purchase_orders": "/api/purchase-orders",
            "milestones": "/api/projects/{project_id}/milestones",
            "quality_assessments": "/api/quality/assessments",
            "orders": "/api/orders",
            "resource_allocation": "/api/resources/allocation"
        }
        
        endpoint = endpoint_map.get(source)
        if not endpoint:
            return {}
        
        # Add filters
        params = {
            "from_date": period["start"],
            "to_date": period["end"]
        }
        
        if context.get("project_id"):
            endpoint = endpoint.replace("{project_id}", str(context["project_id"]))
            params["project_id"] = context["project_id"]
        
        # Fetch via API executor
        result = await api_executor.process({
            "endpoint": endpoint,
            "method": "GET",
            "params": params,
            "context": context,
            "parameters": {"mode": "data_fetch"}
        })
        
        return result.get("data", {}) if result.get("success") else {}
    
    async def _perform_calculation(
        self,
        formula: str,
        data: Dict[str, Any]
    ) -> float:
        """Perform metric calculation based on formula"""
        # Simple calculation engine
        # In production, would use a proper expression evaluator
        
        if formula == "completed_tasks / work_hours":
            tasks = len(data.get("task_completions", {}).get("items", []))
            hours = sum(
                ts.get("hours", 0) 
                for ts in data.get("timesheets", {}).get("items", [])
            )
            return tasks / hours if hours > 0 else 0
        
        elif formula == "(actual_cost - budgeted_cost) / budgeted_cost * 100":
            budgeted = sum(
                b.get("amount", 0)
                for b in data.get("budgets", {}).get("items", [])
            )
            actual = sum(
                i.get("total", 0)
                for i in data.get("invoices", {}).get("items", [])
            )
            return ((actual - budgeted) / budgeted * 100) if budgeted > 0 else 0
        
        elif formula == "on_time_milestones / total_milestones * 100":
            milestones = data.get("milestones", {}).get("items", [])
            total = len(milestones)
            on_time = sum(
                1 for m in milestones
                if m.get("status") == "completed" and
                m.get("completed_date") <= m.get("due_date")
            )
            return (on_time / total * 100) if total > 0 else 0
        
        elif formula == "average(order_completion_time)":
            orders = data.get("orders", {}).get("items", [])
            completion_times = []
            
            for order in orders:
                if order.get("status") == "completed":
                    created = datetime.fromisoformat(order.get("created_at", ""))
                    completed = datetime.fromisoformat(order.get("completed_at", ""))
                    days = (completed - created).days
                    completion_times.append(days)
            
            return np.mean(completion_times) if completion_times else 0
        
        else:
            # Default calculation
            return 0
    
    async def _generate_visualization(
        self,
        viz_type: str,
        metrics: Dict[str, KPIMetric],
        period: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate visualization data"""
        if viz_type == "budget_burndown":
            return {
                "type": "line_chart",
                "title": "Budget Burndown",
                "data": await self._generate_burndown_data(period),
                "config": {
                    "x_axis": "date",
                    "y_axis": "remaining_budget",
                    "color": "#1890ff"
                }
            }
        
        elif viz_type == "milestone_timeline":
            return {
                "type": "gantt_chart",
                "title": "Milestone Timeline",
                "data": await self._generate_timeline_data(period),
                "config": {
                    "start_field": "start_date",
                    "end_field": "due_date",
                    "label_field": "name"
                }
            }
        
        elif viz_type == "kpi_dashboard":
            return {
                "type": "metrics_grid",
                "title": "KPI Dashboard",
                "data": [
                    {
                        "name": metric.name,
                        "value": metric.value,
                        "unit": metric.unit,
                        "change": metric.change_percentage,
                        "trend": "up" if metric.change_percentage > 0 else "down"
                    }
                    for metric in metrics.values()
                ]
            }
        
        elif viz_type == "trend_charts":
            return {
                "type": "multi_line_chart",
                "title": "Metric Trends",
                "data": {
                    metric_name: metric.trend or []
                    for metric_name, metric in metrics.items()
                }
            }
        
        else:
            return {
                "type": "placeholder",
                "title": viz_type,
                "data": []
            }
    
    async def _generate_insights(
        self,
        metrics: Dict[str, KPIMetric]
    ) -> List[str]:
        """Generate insights from metrics"""
        insights = []
        
        # Analyze each metric
        for metric_name, metric in metrics.items():
            if metric.change_percentage:
                if abs(metric.change_percentage) > 10:
                    direction = "increased" if metric.change_percentage > 0 else "decreased"
                    insights.append(
                        f"{metric.name} has {direction} by "
                        f"{abs(metric.change_percentage):.1f}% compared to previous period"
                    )
            
            # Check against thresholds
            if metric_name == "budget_variance" and metric.value > 5:
                insights.append(
                    f"Budget overrun detected: {metric.value:.1f}% over budget"
                )
            
            elif metric_name == "schedule_adherence" and metric.value < 80:
                insights.append(
                    f"Schedule risk: Only {metric.value:.1f}% of milestones on time"
                )
            
            elif metric_name == "quality_score" and metric.value < 90:
                insights.append(
                    f"Quality concerns: Score dropped to {metric.value:.1f}%"
                )
        
        # Correlation insights
        if "productivity" in metrics and "resource_utilization" in metrics:
            prod = metrics["productivity"].value
            util = metrics["resource_utilization"].value
            
            if prod < 1 and util > 90:
                insights.append(
                    "High resource utilization but low productivity suggests inefficiencies"
                )
        
        # Sort by importance
        insights.sort(key=lambda x: "risk" in x or "concern" in x, reverse=True)
        
        return insights
    
    async def _analyze_trends(
        self,
        metrics: List[str],
        query: AnalyticsQuery
    ) -> Dict[str, Any]:
        """Analyze trends over time"""
        trend_data = {}
        
        for metric in metrics:
            # Get historical data points
            data_points = await self._get_historical_data(
                metric,
                query.time_range,
                query.granularity
            )
            
            # Calculate trend line
            if len(data_points) > 1:
                x = np.arange(len(data_points))
                y = np.array([p["value"] for p in data_points])
                
                # Simple linear regression
                coeffs = np.polyfit(x, y, 1)
                trend_direction = "increasing" if coeffs[0] > 0 else "decreasing"
                
                trend_data[metric] = {
                    "data_points": data_points,
                    "trend": trend_direction,
                    "slope": float(coeffs[0]),
                    "forecast": self._simple_forecast(data_points, coeffs)
                }
        
        return {
            "trends": trend_data,
            "message": f"Analyzed trends for {len(metrics)} metrics",
            "period": query.time_range
        }
    
    async def _generate_forecast(
        self,
        metrics: List[str],
        query: AnalyticsQuery
    ) -> Dict[str, Any]:
        """Generate metric forecasts"""
        forecasts = {}
        
        for metric in metrics:
            historical = await self._get_historical_data(
                metric,
                query.time_range,
                TimeGranularity.DAILY
            )
            
            if len(historical) > 7:
                # Simple moving average forecast
                values = [p["value"] for p in historical]
                ma_7 = np.convolve(values, np.ones(7)/7, mode='valid')
                
                # Project forward
                trend = (ma_7[-1] - ma_7[0]) / len(ma_7)
                forecast_days = 30
                
                forecast_values = []
                last_value = ma_7[-1]
                
                for i in range(forecast_days):
                    next_value = last_value + trend
                    forecast_values.append({
                        "date": (datetime.utcnow() + timedelta(days=i+1)).isoformat(),
                        "value": float(next_value),
                        "confidence": 0.8 - (i * 0.02)  # Decreasing confidence
                    })
                    last_value = next_value
                
                forecasts[metric] = {
                    "historical": historical[-30:],  # Last 30 days
                    "forecast": forecast_values,
                    "method": "moving_average",
                    "confidence": 0.7
                }
        
        return {
            "forecasts": forecasts,
            "message": f"Generated {len(forecasts)} forecasts",
            "forecast_period": "30 days"
        }
    
    def _get_default_period(self, template: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get default time period"""
        if template and template.get("frequency") == "weekly":
            start = datetime.utcnow() - timedelta(days=7)
        elif template and template.get("frequency") == "monthly":
            start = datetime.utcnow() - timedelta(days=30)
        else:
            start = datetime.utcnow() - timedelta(days=1)
        
        return {
            "start": start.isoformat(),
            "end": datetime.utcnow().isoformat(),
            "name": template.get("frequency", "daily") if template else "daily"
        }
    
    def _map_intent_to_metric_type(self, intent_type: str) -> MetricType:
        """Map intent type to metric type"""
        mapping = {
            "kpi": MetricType.AVERAGE,
            "trend": MetricType.TREND,
            "comparison": MetricType.PERCENTAGE,
            "forecast": MetricType.TREND,
            "summary": MetricType.DISTRIBUTION
        }
        return mapping.get(intent_type, MetricType.COUNT)
    
    def _parse_time_range(self, time_range_str: str) -> Dict[str, datetime]:
        """Parse time range string"""
        now = datetime.utcnow()
        
        if time_range_str == "today":
            start = now.replace(hour=0, minute=0, second=0)
            end = now
        elif time_range_str == "week":
            start = now - timedelta(days=7)
            end = now
        elif time_range_str == "month":
            start = now - timedelta(days=30)
            end = now
        elif time_range_str == "previous_period":
            start = now - timedelta(days=60)
            end = now - timedelta(days=30)
        else:
            start = now - timedelta(days=7)
            end = now
        
        return {"start": start, "end": end}
    
    async def _get_historical_data(
        self,
        metric: str,
        time_range: Dict[str, Any],
        granularity: TimeGranularity
    ) -> List[Dict[str, Any]]:
        """Get historical data points for a metric"""
        # Fetch historical data from CMS API
        metric_type = metric.lower().replace("_", "")
        data_points = []
        
        start = datetime.fromisoformat(time_range["start"])
        end = datetime.fromisoformat(time_range["end"])
        
        # Get REAL data from CMS API instead of random
        from api.cms_client import cms_client
        
        # Fetch actual metrics based on metric type
        real_data = []
        try:
            if metric_type == "orders":
                # Get real order data from CMS
                response = await cms_client.get("/api/orders")
                if response and "entities" in response:
                    orders = response["entities"]
                    # Group by date and count
                    from collections import defaultdict
                    date_counts = defaultdict(int)
                    for order in orders:
                        if "created_at" in order:
                            order_date = datetime.fromisoformat(order["created_at"]).date()
                            date_counts[order_date] += 1
                    
                    # Use real counts
                    current = start
                    while current <= end:
                        date_key = current.date() if hasattr(current, 'date') else current
                        value = date_counts.get(date_key, 0)
                        data_points.append({
                            "date": current.isoformat(),
                            "value": value  # Use actual value (0 if no data)
                        })
                        
                        if granularity == TimeGranularity.HOURLY:
                            current += timedelta(hours=1)
                        elif granularity == TimeGranularity.DAILY:
                            current += timedelta(days=1)
                        elif granularity == TimeGranularity.WEEKLY:
                            current += timedelta(weeks=1)
                        else:
                            current += timedelta(days=1)
                            
            elif metric_type == "revenue":
                # Get real invoice data from CMS
                response = await cms_client.get("/api/invoices")
                if response and "entities" in response:
                    invoices = response["entities"]
                    # Calculate real revenue
                    from collections import defaultdict
                    date_revenue = defaultdict(float)
                    for invoice in invoices:
                        if "date" in invoice and "total" in invoice:
                            invoice_date = datetime.fromisoformat(invoice["date"]).date()
                            # Parse amount (remove currency symbols)
                            amount_str = str(invoice["total"]).replace("£", "").replace(",", "")
                            try:
                                amount = float(amount_str)
                                date_revenue[invoice_date] += amount
                            except:
                                pass
                    
                    # Use real revenue
                    current = start
                    while current <= end:
                        date_key = current.date() if hasattr(current, 'date') else current
                        value = date_revenue.get(date_key, 0)
                        data_points.append({
                            "date": current.isoformat(),
                            "value": value  # Use actual value (0 if no data)
                        })
                        
                        if granularity == TimeGranularity.HOURLY:
                            current += timedelta(hours=1)
                        elif granularity == TimeGranularity.DAILY:
                            current += timedelta(days=1)
                        elif granularity == TimeGranularity.WEEKLY:
                            current += timedelta(weeks=1)
                        else:
                            current += timedelta(days=1)
            else:
                # For other metrics, try to get relevant data
                current = start
                while current <= end:
                    # Still use some random for unsupported metrics but lower values
                    # For unsupported metrics, return 0 values
                    data_points.append({
                        "date": current.isoformat(),
                        "value": 0  # No data available for this metric type
                    })
                    
                    if granularity == TimeGranularity.HOURLY:
                        current += timedelta(hours=1)
                    elif granularity == TimeGranularity.DAILY:
                        current += timedelta(days=1)
                    elif granularity == TimeGranularity.WEEKLY:
                        current += timedelta(weeks=1)
                    else:
                        current += timedelta(days=1)
                        
        except Exception as e:
            logger.error(f"Failed to fetch real CMS data: {e}")
            # Return error message when API fails
            return [{
                "error": True,
                "message": f"Sorry, we cannot retrieve {metric} data at this time. Please ensure the CMS API is accessible.",
                "date": datetime.now().isoformat()
            }]
        
        return data_points
    
    def _simple_forecast(
        self,
        data_points: List[Dict[str, Any]],
        coeffs: np.ndarray
    ) -> float:
        """Simple linear forecast"""
        next_x = len(data_points)
        return float(coeffs[0] * next_x + coeffs[1])
    
    async def _calculate_previous_period(
        self,
        metric_name: str,
        context: Dict[str, Any],
        period: Dict[str, Any]
    ) -> float:
        """Calculate metric for previous period"""
        # Shift period back
        start = datetime.fromisoformat(period["start"])
        end = datetime.fromisoformat(period["end"])
        duration = end - start
        
        prev_period = {
            "start": (start - duration).isoformat(),
            "end": start.isoformat()
        }
        
        # Calculate metric for previous period
        prev_metric = await self._calculate_single_metric(
            metric_name,
            context,
            prev_period
        )
        
        return prev_metric.value
    
    async def _get_metric_trend(
        self,
        metric_name: str,
        context: Dict[str, Any],
        period: Dict[str, Any]
    ) -> List[float]:
        """Get metric trend data"""
        # Get last 7 data points
        historical = await self._get_historical_data(
            metric_name,
            period,
            TimeGranularity.DAILY
        )
        
        return [p["value"] for p in historical[-7:]]
    
    async def _generate_burndown_data(self, period: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate budget burndown chart data from real CMS data"""
        from api.cms_client import cms_client
        data = []
        start = datetime.fromisoformat(period["start"])
        end = datetime.fromisoformat(period["end"])
        
        try:
            # Get real project/order data from CMS
            projects_response = await cms_client.get("/api/projects")
            orders_response = await cms_client.get("/api/orders")
            invoices_response = await cms_client.get("/api/invoices")
            
            # Calculate total budget from projects
            total_budget = 0
            if projects_response and "entities" in projects_response:
                for project in projects_response["entities"]:
                    if "budget" in project:
                        budget_str = str(project["budget"]).replace("£", "").replace(",", "")
                        try:
                            total_budget += float(budget_str)
                        except:
                            pass
            
            # If no budget found, use a reasonable default
            if total_budget == 0:
                total_budget = 100000
            
            # Calculate actual spend from invoices
            daily_spend = {}
            if invoices_response and "entities" in invoices_response:
                for invoice in invoices_response["entities"]:
                    if "date" in invoice and "total" in invoice:
                        invoice_date = datetime.fromisoformat(invoice["date"]).date()
                        amount_str = str(invoice["total"]).replace("£", "").replace(",", "")
                        try:
                            amount = float(amount_str)
                            if invoice_date not in daily_spend:
                                daily_spend[invoice_date] = 0
                            daily_spend[invoice_date] += amount
                        except:
                            pass
            
            # Calculate average daily burn rate
            avg_daily_burn = sum(daily_spend.values()) / len(daily_spend) if daily_spend else 3000
            
            # Generate burndown data
            current = start
            remaining = total_budget
            total_spent = 0
            
            while current <= end:
                current_date = current.date() if hasattr(current, 'date') else current
                
                # Use actual spend if available
                if current_date in daily_spend:
                    total_spent += daily_spend[current_date]
                    remaining = total_budget - total_spent
                else:
                    # Use average for future dates
                    if current > datetime.now():
                        remaining -= avg_daily_burn
                    
                data.append({
                    "date": current.isoformat(),
                    "remaining_budget": max(0, remaining),
                    "projected": max(0, total_budget - (avg_daily_burn * (current - start).days)),
                    "actual_spend": daily_spend.get(current_date, 0)
                })
                current += timedelta(days=1)
            
        except Exception as e:
            logger.error(f"Failed to generate burndown data from CMS: {e}")
            # Return message instead of fake data
            return [{
                "error": True,
                "message": "Sorry, we don't have budget data available yet. This will be available once projects are added to the CMS system.",
                "date": datetime.now().isoformat()
            }]
        
        return data
    
    async def _generate_timeline_data(self, period: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate milestone timeline data from real CMS projects"""
        from api.cms_client import cms_client
        
        try:
            # Get real project data from CMS
            response = await cms_client.get("/api/projects")
            
            if response and "entities" in response:
                projects = response["entities"]
                timeline_data = []
                
                for project in projects[:10]:  # Limit to 10 most recent projects
                    # Extract project timeline info
                    project_timeline = {
                        "name": project.get("name", "Unnamed Project"),
                        "start_date": project.get("start_date", period["start"]),
                        "due_date": project.get("end_date", period["end"]),
                        "status": project.get("status", "pending")
                    }
                    
                    # Map CMS status to our status format
                    cms_status = project.get("status", "").lower()
                    if "complete" in cms_status or "finished" in cms_status:
                        project_timeline["status"] = "completed"
                    elif "progress" in cms_status or "active" in cms_status:
                        project_timeline["status"] = "in_progress" 
                    else:
                        project_timeline["status"] = "pending"
                    
                    # Add progress percentage if available
                    if "progress" in project:
                        project_timeline["progress"] = project["progress"]
                    
                    timeline_data.append(project_timeline)
                
                return timeline_data if timeline_data else self._get_default_timeline()
            else:
                return self._get_default_timeline()
                
        except Exception as e:
            logger.error(f"Failed to fetch timeline data from CMS: {e}")
            return self._get_default_timeline()
    
    def _get_default_timeline(self) -> List[Dict[str, Any]]:
        """Get default timeline when no real data available"""
        return [
            {
                "name": "Data Collection Phase",
                "start_date": datetime.now().isoformat(),
                "due_date": (datetime.now() + timedelta(days=7)).isoformat(),
                "status": "pending"
            }
        ]
    
    async def _calculate_kpis(
        self,
        metrics: List[str],
        query: AnalyticsQuery
    ) -> Dict[str, Any]:
        """Calculate multiple KPIs"""
        if not metrics:
            metrics = list(self.kpi_definitions.keys())
        
        results = {}
        for metric in metrics:
            if metric in self.kpi_definitions:
                kpi = await self._calculate_single_metric(
                    metric,
                    {"entity_type": query.entity_type},
                    query.time_range
                )
                results[metric] = kpi.dict()
        
        return {
            "kpis": results,
            "message": f"Calculated {len(results)} KPIs",
            "period": query.time_range
        }
    
    async def _generate_comparisons(self, query: AnalyticsQuery) -> Dict[str, Any]:
        """Generate comparison analytics"""
        # Compare current vs previous period
        current_kpis = await self._calculate_kpis([], query)
        
        # Calculate previous period
        prev_query = query.copy()
        duration = query.time_range["end"] - query.time_range["start"]
        prev_query.time_range = {
            "start": query.time_range["start"] - duration,
            "end": query.time_range["start"]
        }
        
        previous_kpis = await self._calculate_kpis([], prev_query)
        
        # Generate comparison
        comparisons = {}
        for metric in current_kpis["kpis"]:
            if metric in previous_kpis["kpis"]:
                current = current_kpis["kpis"][metric]["value"]
                previous = previous_kpis["kpis"][metric]["value"]
                
                change = ((current - previous) / previous * 100) if previous else 0
                
                comparisons[metric] = {
                    "current": current,
                    "previous": previous,
                    "change": change,
                    "improved": change > 0 if "efficiency" in metric else change < 0
                }
        
        return {
            "comparisons": comparisons,
            "message": "Period-over-period comparison",
            "current_period": query.time_range,
            "previous_period": prev_query.time_range
        }
    
    async def _generate_summary(self, query: AnalyticsQuery) -> Dict[str, Any]:
        """Generate summary analytics"""
        # Calculate all relevant KPIs
        kpis = await self._calculate_kpis([], query)
        
        # Get real entity counts from CMS API
        from api.cms_client import cms_client
        
        entity_counts = {}
        try:
            # Fetch real counts from CMS
            projects = await cms_client.get("/api/projects")
            orders = await cms_client.get("/api/orders")
            users = await cms_client.get("/api/users")
            callouts = await cms_client.get("/api/call-outs")
            
            # Count active projects
            active_projects = 0
            if projects and "entities" in projects:
                for project in projects["entities"]:
                    if project.get("status", "").lower() in ["active", "in_progress", "ongoing"]:
                        active_projects += 1
            
            # Count pending orders
            pending_orders = 0
            if orders and "entities" in orders:
                for order in orders["entities"]:
                    if order.get("status", "").lower() in ["pending", "processing", "new"]:
                        pending_orders += 1
            
            # Count team members
            team_members = len(users["entities"]) if users and "entities" in users else 0
            
            # Count open callouts/issues
            open_issues = 0
            if callouts and "entities" in callouts:
                for callout in callouts["entities"]:
                    if callout.get("status", "").lower() not in ["closed", "completed", "resolved"]:
                        open_issues += 1
            
            entity_counts = {
                "active_projects": active_projects,
                "pending_orders": pending_orders,
                "team_members": team_members,
                "open_issues": open_issues
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch entity counts from CMS: {e}")
            # Use minimal defaults if API fails
            entity_counts = {
                "active_projects": 0,
                "pending_orders": 0,
                "team_members": 0,
                "open_issues": 0
            }
        
        # Generate summary
        summary = {
            "overview": entity_counts,
            "key_metrics": {
                name: kpi["value"]
                for name, kpi in list(kpis["kpis"].items())[:5]
            },
            "health_score": self._calculate_health_score(kpis["kpis"]),
            "recommendations": self._generate_recommendations(kpis["kpis"])
        }
        
        return {
            "summary": summary,
            "message": "Analytics summary generated",
            "period": query.time_range
        }
    
    def _calculate_health_score(self, kpis: Dict[str, Any]) -> float:
        """Calculate overall health score"""
        scores = []
        
        # Budget health
        if "budget_variance" in kpis:
            variance = kpis["budget_variance"]["value"]
            score = 100 - abs(variance) if abs(variance) < 100 else 0
            scores.append(score)
        
        # Schedule health
        if "schedule_adherence" in kpis:
            scores.append(kpis["schedule_adherence"]["value"])
        
        # Quality health
        if "quality_score" in kpis:
            scores.append(kpis["quality_score"]["value"])
        
        return np.mean(scores) if scores else 75
    
    def _generate_recommendations(self, kpis: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on KPIs"""
        recommendations = []
        
        if kpis.get("budget_variance", {}).get("value", 0) > 10:
            recommendations.append(
                "Review and control costs - budget overrun detected"
            )
        
        if kpis.get("schedule_adherence", {}).get("value", 100) < 80:
            recommendations.append(
                "Focus on milestone completion - schedule slipping"
            )
        
        if kpis.get("resource_utilization", {}).get("value", 0) > 95:
            recommendations.append(
                "Consider adding resources - team at capacity"
            )
        
        if kpis.get("quality_score", {}).get("value", 100) < 90:
            recommendations.append(
                "Implement quality improvement measures"
            )
        
        return recommendations[:3]  # Top 3 recommendations

# Create singleton instance
analytics_insights = AnalyticsInsightsAgent()