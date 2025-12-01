"""
Emergency API Tool

Implements emergency escalation functionality through an OpenAPI-compatible
endpoint. Handles crisis situations by notifying emergency contacts and
logging critical events.
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from enum import Enum

import aiohttp
import aiosqlite
from opentelemetry import trace

try:
    from config import config
except ImportError:
    from ..config import config

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertType(Enum):
    """Types of emergency alerts."""
    CRISIS_DETECTED = "crisis_detected"
    SAFETY_CHECK_FAILED = "safety_check_failed"
    ESCALATION_REQUESTED = "escalation_requested"
    WELFARE_CHECK = "welfare_check"
    EMERGENCY_CONTACT = "emergency_contact"


class AlertStatus(Enum):
    """Status of an alert."""
    PENDING = "pending"
    SENT = "sent"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"
    FAILED = "failed"


@dataclass
class EmergencyContact:
    """Emergency contact information."""
    id: str
    user_id: str
    name: str
    relationship: str
    phone: Optional[str] = None
    email: Optional[str] = None
    is_primary: bool = False
    notify_on_crisis: bool = True


@dataclass
class EmergencyAlert:
    """
    An emergency alert record.
    
    Attributes:
        id: Unique alert identifier
        user_id: User the alert is for
        session_id: Session during which alert was triggered
        alert_type: Type of emergency
        severity: Alert severity level
        status: Current status
        trigger_message: Message that triggered the alert
        risk_factors: Identified risk factors
        actions_taken: Actions performed
        notes: Additional notes
        created_at: When the alert was created
        resolved_at: When the alert was resolved
    """
    user_id: str
    alert_type: AlertType
    severity: AlertSeverity
    trigger_message: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    session_id: Optional[str] = None
    status: AlertStatus = AlertStatus.PENDING
    risk_factors: List[str] = field(default_factory=list)
    actions_taken: List[str] = field(default_factory=list)
    notes: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    resolved_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['alert_type'] = self.alert_type.value
        data['severity'] = self.severity.value
        data['status'] = self.status.value
        data['risk_factors'] = json.dumps(self.risk_factors)
        data['actions_taken'] = json.dumps(self.actions_taken)
        data['created_at'] = self.created_at.isoformat()
        data['resolved_at'] = self.resolved_at.isoformat() if self.resolved_at else None
        return data


class EmergencyAPI:
    """
    Emergency escalation API tool.
    
    Provides:
    - Emergency alert creation and tracking
    - Emergency contact management
    - External webhook notifications
    - Alert status management
    - Audit logging
    
    OpenAPI Integration:
    This tool can integrate with external emergency services
    through the OpenAPI specification in emergency_api.yaml
    """
    
    # Crisis resources for different regions
    CRISIS_RESOURCES = {
        "us": {
            "suicide_hotline": "988",
            "crisis_text": "Text HOME to 741741",
            "emergency": "911"
        },
        "uk": {
            "samaritans": "116 123",
            "crisis_text": "Text SHOUT to 85258",
            "emergency": "999"
        },
        "international": {
            "iasp": "https://www.iasp.info/resources/Crisis_Centres/"
        }
    }

    def __init__(
        self,
        db_path: Optional[Path] = None,
        webhook_url: Optional[str] = None
    ):
        """
        Initialize the Emergency API.
        
        Args:
            db_path: Path to SQLite database
            webhook_url: URL for emergency webhook notifications
        """
        self.db_path = db_path or config.memory_db_path
        self.webhook_url = webhook_url or config.emergency_webhook_url
        self._initialized = False
        self._db_lock = asyncio.Lock()
        
        # Metrics
        self._alerts_created = 0
        self._alerts_by_severity: Dict[str, int] = {}
        self._webhook_calls = 0
        
        logger.info("EmergencyAPI initialized")

    async def initialize(self) -> None:
        """Initialize the database schema."""
        if self._initialized:
            return
        
        async with self._db_lock:
            async with aiosqlite.connect(self.db_path) as db:
                # Emergency contacts table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS emergency_contacts (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        name TEXT NOT NULL,
                        relationship TEXT,
                        phone TEXT,
                        email TEXT,
                        is_primary INTEGER DEFAULT 0,
                        notify_on_crisis INTEGER DEFAULT 1
                    )
                """)
                
                # Emergency alerts table
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS emergency_alerts (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        session_id TEXT,
                        alert_type TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        status TEXT NOT NULL,
                        trigger_message TEXT,
                        risk_factors TEXT,
                        actions_taken TEXT,
                        notes TEXT,
                        created_at TEXT NOT NULL,
                        resolved_at TEXT
                    )
                """)
                
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_alerts_user 
                    ON emergency_alerts(user_id)
                """)
                
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_alerts_status 
                    ON emergency_alerts(status)
                """)
                
                await db.commit()
                
            self._initialized = True
            logger.info("Emergency API database initialized")

    async def create_alert(
        self,
        user_id: str,
        alert_type: str,
        severity: str,
        trigger_message: str,
        risk_factors: Optional[List[str]] = None,
        session_id: Optional[str] = None,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create an emergency alert.
        
        Args:
            user_id: User the alert is for
            alert_type: Type of emergency
            severity: Alert severity
            trigger_message: What triggered the alert
            risk_factors: Identified risk factors
            session_id: Current session ID
            notes: Additional notes
            
        Returns:
            Alert creation result
        """
        await self.initialize()
        
        with tracer.start_as_current_span("emergency_create_alert") as span:
            span.set_attribute("alert_type", alert_type)
            span.set_attribute("severity", severity)
            span.set_attribute("user_id", user_id)
            
            try:
                alert = EmergencyAlert(
                    user_id=user_id,
                    alert_type=AlertType(alert_type),
                    severity=AlertSeverity(severity),
                    trigger_message=trigger_message,
                    risk_factors=risk_factors or [],
                    session_id=session_id,
                    notes=notes
                )
                
                # Store alert
                async with self._db_lock:
                    async with aiosqlite.connect(self.db_path) as db:
                        data = alert.to_dict()
                        await db.execute("""
                            INSERT INTO emergency_alerts 
                            (id, user_id, session_id, alert_type, severity, status,
                             trigger_message, risk_factors, actions_taken, notes,
                             created_at, resolved_at)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            data['id'], data['user_id'], data['session_id'],
                            data['alert_type'], data['severity'], data['status'],
                            data['trigger_message'], data['risk_factors'],
                            data['actions_taken'], data['notes'],
                            data['created_at'], data['resolved_at']
                        ))
                        await db.commit()
                
                # Update metrics
                self._alerts_created += 1
                self._alerts_by_severity[severity] = \
                    self._alerts_by_severity.get(severity, 0) + 1
                
                logger.warning(
                    f"Emergency alert created: {alert.id} "
                    f"(type: {alert_type}, severity: {severity})"
                )
                
                # Trigger webhook if severity is high or critical
                webhook_result = None
                if severity in ["high", "critical"] and config.emergency_contact_enabled:
                    webhook_result = await self._send_webhook(alert)
                
                # Get crisis resources
                resources = self.get_crisis_resources()
                
                return {
                    "success": True,
                    "alert_id": alert.id,
                    "alert_type": alert_type,
                    "severity": severity,
                    "status": alert.status.value,
                    "message": "Emergency alert created and logged",
                    "webhook_sent": webhook_result is not None,
                    "crisis_resources": resources,
                    "immediate_actions": self._get_immediate_actions(severity)
                }
                
            except Exception as e:
                logger.error(f"Failed to create emergency alert: {e}")
                span.set_status(trace.Status(trace.StatusCode.ERROR, str(e)))
                return {
                    "success": False,
                    "error": str(e),
                    "crisis_resources": self.get_crisis_resources()
                }

    async def _send_webhook(self, alert: EmergencyAlert) -> Optional[Dict[str, Any]]:
        """
        Send emergency webhook notification.
        
        Args:
            alert: The emergency alert
            
        Returns:
            Webhook response or None if failed
        """
        if not self.webhook_url:
            logger.warning("No webhook URL configured")
            return None
        
        with tracer.start_as_current_span("emergency_webhook") as span:
            span.set_attribute("alert_id", alert.id)
            
            try:
                payload = {
                    "alert_id": alert.id,
                    "user_id": alert.user_id,
                    "alert_type": alert.alert_type.value,
                    "severity": alert.severity.value,
                    "trigger_message": alert.trigger_message[:200],  # Truncate for privacy
                    "risk_factors": alert.risk_factors,
                    "timestamp": alert.created_at.isoformat(),
                    "session_id": alert.session_id
                }
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        self.webhook_url,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        self._webhook_calls += 1
                        
                        if response.status == 200:
                            logger.info(f"Emergency webhook sent successfully for alert {alert.id}")
                            return {"status": "sent", "response_code": response.status}
                        else:
                            logger.error(f"Webhook failed with status {response.status}")
                            return {"status": "failed", "response_code": response.status}
                            
            except asyncio.TimeoutError:
                logger.error("Emergency webhook timed out")
                return {"status": "timeout"}
            except Exception as e:
                logger.error(f"Emergency webhook failed: {e}")
                return {"status": "error", "message": str(e)}

    def _get_immediate_actions(self, severity: str) -> List[str]:
        """Get immediate actions based on severity."""
        if severity == "critical":
            return [
                "Stay on the line with the user",
                "Provide crisis hotline numbers",
                "Encourage calling emergency services",
                "Assess immediate safety",
                "Document the interaction"
            ]
        elif severity == "high":
            return [
                "Validate the user's feelings",
                "Provide crisis resources",
                "Assess support system availability",
                "Create a safety plan",
                "Schedule follow-up"
            ]
        elif severity == "medium":
            return [
                "Continue supportive conversation",
                "Offer coping resources",
                "Monitor for escalation"
            ]
        else:
            return [
                "Continue normal support",
                "Log for pattern tracking"
            ]

    def get_crisis_resources(self, region: str = "us") -> Dict[str, Any]:
        """Get crisis resources for a region."""
        resources = self.CRISIS_RESOURCES.get(region, {})
        resources.update(self.CRISIS_RESOURCES.get("international", {}))
        
        return {
            "region": region,
            "resources": resources,
            "message": "If you're in immediate danger, please call emergency services."
        }

    async def update_alert_status(
        self,
        alert_id: str,
        status: str,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Update an alert's status.
        
        Args:
            alert_id: Alert to update
            status: New status
            notes: Additional notes
            
        Returns:
            Update result
        """
        await self.initialize()
        
        async with self._db_lock:
            async with aiosqlite.connect(self.db_path) as db:
                resolved_at = datetime.utcnow().isoformat() if status == "resolved" else None
                
                await db.execute("""
                    UPDATE emergency_alerts 
                    SET status = ?, notes = COALESCE(?, notes), resolved_at = ?
                    WHERE id = ?
                """, (status, notes, resolved_at, alert_id))
                
                if db.total_changes == 0:
                    return {"success": False, "error": "Alert not found"}
                
                await db.commit()
        
        logger.info(f"Alert {alert_id} status updated to {status}")
        
        return {
            "success": True,
            "alert_id": alert_id,
            "new_status": status,
            "resolved_at": resolved_at
        }

    async def add_emergency_contact(
        self,
        user_id: str,
        name: str,
        relationship: str,
        phone: Optional[str] = None,
        email: Optional[str] = None,
        is_primary: bool = False
    ) -> Dict[str, Any]:
        """
        Add an emergency contact for a user.
        
        Args:
            user_id: User to add contact for
            name: Contact name
            relationship: Relationship to user
            phone: Phone number
            email: Email address
            is_primary: Whether this is the primary contact
            
        Returns:
            Contact creation result
        """
        await self.initialize()
        
        contact = EmergencyContact(
            id=str(uuid.uuid4()),
            user_id=user_id,
            name=name,
            relationship=relationship,
            phone=phone,
            email=email,
            is_primary=is_primary
        )
        
        async with self._db_lock:
            async with aiosqlite.connect(self.db_path) as db:
                # If setting as primary, unset other primaries
                if is_primary:
                    await db.execute("""
                        UPDATE emergency_contacts SET is_primary = 0
                        WHERE user_id = ?
                    """, (user_id,))
                
                await db.execute("""
                    INSERT INTO emergency_contacts 
                    (id, user_id, name, relationship, phone, email, is_primary, notify_on_crisis)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    contact.id, user_id, name, relationship,
                    phone, email, 1 if is_primary else 0, 1
                ))
                await db.commit()
        
        logger.info(f"Emergency contact added for user {user_id}: {name}")
        
        return {
            "success": True,
            "contact_id": contact.id,
            "name": name,
            "relationship": relationship,
            "is_primary": is_primary
        }

    async def get_emergency_contacts(self, user_id: str) -> Dict[str, Any]:
        """Get all emergency contacts for a user."""
        await self.initialize()
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("""
                SELECT * FROM emergency_contacts WHERE user_id = ?
                ORDER BY is_primary DESC, name ASC
            """, (user_id,)) as cursor:
                rows = await cursor.fetchall()
                
                contacts = [
                    {
                        "id": row['id'],
                        "name": row['name'],
                        "relationship": row['relationship'],
                        "phone": row['phone'],
                        "email": row['email'],
                        "is_primary": bool(row['is_primary'])
                    }
                    for row in rows
                ]
                
                return {
                    "success": True,
                    "contacts": contacts,
                    "count": len(contacts)
                }

    async def get_alert_history(
        self,
        user_id: str,
        limit: int = 20
    ) -> Dict[str, Any]:
        """Get alert history for a user."""
        await self.initialize()
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("""
                SELECT * FROM emergency_alerts WHERE user_id = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (user_id, limit)) as cursor:
                rows = await cursor.fetchall()
                
                alerts = [
                    {
                        "id": row['id'],
                        "alert_type": row['alert_type'],
                        "severity": row['severity'],
                        "status": row['status'],
                        "created_at": row['created_at'],
                        "resolved_at": row['resolved_at']
                    }
                    for row in rows
                ]
                
                return {
                    "success": True,
                    "alerts": alerts,
                    "count": len(alerts)
                }

    def get_metrics(self) -> Dict[str, Any]:
        """Return tool metrics for observability."""
        return {
            "tool": "emergency_api",
            "alerts_created": self._alerts_created,
            "alerts_by_severity": self._alerts_by_severity,
            "webhook_calls": self._webhook_calls
        }

