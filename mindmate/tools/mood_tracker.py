"""
Mood Tracker Tool

A comprehensive mood tracking and analysis tool for mental health monitoring.
Supports mood logging, pattern detection, and trend analysis.
"""

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import aiosqlite
from opentelemetry import trace

try:
    from config import config
except ImportError:
    from ..config import config

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class MoodLevel(Enum):
    """Mood level categories."""
    VERY_LOW = 1
    LOW = 2
    BELOW_AVERAGE = 3
    NEUTRAL = 4
    ABOVE_AVERAGE = 5
    GOOD = 6
    VERY_GOOD = 7
    EXCELLENT = 8
    GREAT = 9
    AMAZING = 10


class EnergyLevel(Enum):
    """Energy level categories."""
    EXHAUSTED = 1
    VERY_LOW = 2
    LOW = 3
    MODERATE = 4
    NORMAL = 5
    GOOD = 6
    HIGH = 7
    VERY_HIGH = 8
    ENERGETIC = 9
    MAXIMUM = 10


@dataclass
class MoodEntry:
    """
    A mood tracking entry.
    
    Attributes:
        id: Unique entry identifier
        user_id: User who logged the mood
        mood_rating: Mood level (1-10)
        energy_level: Energy level (1-10)
        emotions: List of emotions felt
        context: What was happening
        activities: Activities engaged in
        sleep_hours: Hours of sleep
        notes: Additional notes
        timestamp: When the mood was logged
    """
    user_id: str
    mood_rating: int
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    energy_level: Optional[int] = None
    emotions: List[str] = field(default_factory=list)
    context: Optional[str] = None
    activities: List[str] = field(default_factory=list)
    sleep_hours: Optional[float] = None
    notes: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['emotions'] = json.dumps(self.emotions)
        data['activities'] = json.dumps(self.activities)
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MoodEntry':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        if isinstance(data['emotions'], str):
            data['emotions'] = json.loads(data['emotions'])
        if isinstance(data['activities'], str):
            data['activities'] = json.loads(data['activities'])
        return cls(**data)


@dataclass
class MoodAnalysis:
    """
    Analysis of mood patterns.
    
    Attributes:
        period_days: Days analyzed
        total_entries: Number of entries
        average_mood: Average mood rating
        mood_trend: Overall trend (improving/stable/declining)
        common_emotions: Most frequent emotions
        best_day_of_week: Day with highest average mood
        worst_day_of_week: Day with lowest average mood
        mood_stability: How stable mood is (high/moderate/low)
        correlations: Discovered correlations
    """
    period_days: int
    total_entries: int
    average_mood: float
    mood_trend: str
    common_emotions: List[Tuple[str, int]]
    best_day_of_week: Optional[str] = None
    worst_day_of_week: Optional[str] = None
    mood_stability: str = "moderate"
    correlations: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class MoodTracker:
    """
    Mood tracking and analysis tool.
    
    Provides:
    - Log mood entries with context
    - View mood history
    - Analyze patterns and trends
    - Identify correlations
    - Generate insights
    """
    
    # Common emotions for categorization
    EMOTION_CATEGORIES = {
        "positive": [
            "happy", "joyful", "content", "peaceful", "grateful",
            "excited", "hopeful", "loved", "proud", "calm"
        ],
        "negative": [
            "sad", "anxious", "angry", "frustrated", "lonely",
            "overwhelmed", "scared", "guilty", "ashamed", "hopeless"
        ],
        "neutral": [
            "neutral", "okay", "fine", "tired", "bored", "confused"
        ]
    }
    
    # Activity categories
    ACTIVITY_CATEGORIES = {
        "social": ["friends", "family", "social", "party", "date"],
        "physical": ["exercise", "walk", "gym", "sports", "yoga"],
        "relaxation": ["rest", "sleep", "nap", "meditation", "reading"],
        "work": ["work", "study", "project", "meeting"],
        "creative": ["art", "music", "writing", "cooking", "crafts"],
        "nature": ["outside", "nature", "park", "hiking", "garden"]
    }

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize the Mood Tracker.
        
        Args:
            db_path: Path to SQLite database
        """
        self.db_path = db_path or config.memory_db_path
        self._initialized = False
        self._db_lock = asyncio.Lock()
        
        # Metrics
        self._entries_logged = 0
        self._analyses_performed = 0

    async def initialize(self) -> None:
        """Initialize the database schema."""
        if self._initialized:
            return
        
        async with self._db_lock:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS mood_entries (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        mood_rating INTEGER NOT NULL,
                        energy_level INTEGER,
                        emotions TEXT,
                        context TEXT,
                        activities TEXT,
                        sleep_hours REAL,
                        notes TEXT,
                        timestamp TEXT NOT NULL
                    )
                """)
                
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_mood_user 
                    ON mood_entries(user_id)
                """)
                
                await db.execute("""
                    CREATE INDEX IF NOT EXISTS idx_mood_timestamp 
                    ON mood_entries(timestamp)
                """)
                
                await db.commit()
                
            self._initialized = True
            logger.info("Mood tracker database initialized")

    async def write(
        self,
        user_id: str,
        mood_rating: int,
        energy_level: Optional[int] = None,
        emotions: Optional[List[str]] = None,
        context: Optional[str] = None,
        activities: Optional[List[str]] = None,
        sleep_hours: Optional[float] = None,
        notes: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Log a mood entry.
        
        Args:
            user_id: User logging the mood
            mood_rating: Mood level (1-10)
            energy_level: Energy level (1-10)
            emotions: Emotions being felt
            context: What's happening
            activities: Activities engaged in
            sleep_hours: Hours of sleep
            notes: Additional notes
            
        Returns:
            Logged entry details
        """
        await self.initialize()
        
        with tracer.start_as_current_span("mood_tracker_write") as span:
            span.set_attribute("user_id", user_id)
            span.set_attribute("mood_rating", mood_rating)
            
            # Validate mood rating
            if not 1 <= mood_rating <= 10:
                return {"success": False, "error": "Mood rating must be between 1 and 10"}
            
            if energy_level and not 1 <= energy_level <= 10:
                return {"success": False, "error": "Energy level must be between 1 and 10"}
            
            entry = MoodEntry(
                user_id=user_id,
                mood_rating=mood_rating,
                energy_level=energy_level,
                emotions=emotions or [],
                context=context,
                activities=activities or [],
                sleep_hours=sleep_hours,
                notes=notes
            )
            
            async with self._db_lock:
                async with aiosqlite.connect(self.db_path) as db:
                    data = entry.to_dict()
                    await db.execute("""
                        INSERT INTO mood_entries 
                        (id, user_id, mood_rating, energy_level, emotions, 
                         context, activities, sleep_hours, notes, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        data['id'], data['user_id'], data['mood_rating'],
                        data['energy_level'], data['emotions'], data['context'],
                        data['activities'], data['sleep_hours'], data['notes'],
                        data['timestamp']
                    ))
                    await db.commit()
            
            self._entries_logged += 1
            logger.info(f"Mood entry logged: {entry.id} (rating: {mood_rating})")
            
            # Generate immediate feedback
            feedback = self._generate_feedback(mood_rating, emotions)
            
            return {
                "success": True,
                "entry_id": entry.id,
                "mood_rating": mood_rating,
                "mood_level": MoodLevel(mood_rating).name.replace("_", " ").title(),
                "feedback": feedback,
                "logged_at": entry.timestamp.isoformat()
            }

    def _generate_feedback(
        self,
        mood_rating: int,
        emotions: Optional[List[str]] = None
    ) -> str:
        """Generate immediate feedback based on mood."""
        if mood_rating >= 7:
            return "I'm glad you're feeling good! It's valuable to notice what contributes to these positive moments."
        elif mood_rating >= 4:
            return "Thank you for checking in. These middle-ground feelings are completely valid."
        else:
            return "I hear that things are tough right now. Thank you for sharing. Remember, difficult moments pass."

    async def read(
        self,
        user_id: str,
        days: int = 7,
        limit: int = 50
    ) -> Dict[str, Any]:
        """
        Read mood history for a user.
        
        Args:
            user_id: User to read for
            days: Number of days to retrieve
            limit: Maximum entries
            
        Returns:
            Mood history
        """
        await self.initialize()
        
        start_date = datetime.utcnow() - timedelta(days=days)
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("""
                SELECT * FROM mood_entries 
                WHERE user_id = ? AND timestamp >= ?
                ORDER BY timestamp DESC
                LIMIT ?
            """, (user_id, start_date.isoformat(), limit)) as cursor:
                rows = await cursor.fetchall()
                
                entries = []
                for row in rows:
                    entry_data = dict(row)
                    entries.append({
                        "id": entry_data['id'],
                        "mood_rating": entry_data['mood_rating'],
                        "mood_level": MoodLevel(entry_data['mood_rating']).name.replace("_", " ").title(),
                        "energy_level": entry_data['energy_level'],
                        "emotions": json.loads(entry_data['emotions'] or '[]'),
                        "context": entry_data['context'],
                        "timestamp": entry_data['timestamp']
                    })
                
                return {
                    "success": True,
                    "period_days": days,
                    "entry_count": len(entries),
                    "entries": entries
                }

    async def analyze(
        self,
        user_id: str,
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze mood patterns over time.
        
        Args:
            user_id: User to analyze
            days: Number of days to analyze
            
        Returns:
            Comprehensive mood analysis
        """
        await self.initialize()
        
        with tracer.start_as_current_span("mood_tracker_analyze") as span:
            span.set_attribute("user_id", user_id)
            span.set_attribute("days", days)
            
            start_date = datetime.utcnow() - timedelta(days=days)
            
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute("""
                    SELECT * FROM mood_entries 
                    WHERE user_id = ? AND timestamp >= ?
                    ORDER BY timestamp ASC
                """, (user_id, start_date.isoformat())) as cursor:
                    rows = await cursor.fetchall()
                    
                    if not rows:
                        return {
                            "success": True,
                            "period_days": days,
                            "total_entries": 0,
                            "message": "No mood entries found in this period. Start tracking to see patterns!"
                        }
                    
                    entries = [MoodEntry.from_dict(dict(row)) for row in rows]
                    
                    # Calculate analysis
                    analysis = self._perform_analysis(entries, days)
                    self._analyses_performed += 1
                    
                    return {
                        "success": True,
                        "period_days": analysis.period_days,
                        "total_entries": analysis.total_entries,
                        "average_mood": round(analysis.average_mood, 2),
                        "mood_level": MoodLevel(round(analysis.average_mood)).name.replace("_", " ").title(),
                        "trend": analysis.mood_trend,
                        "stability": analysis.mood_stability,
                        "common_emotions": analysis.common_emotions,
                        "best_day": analysis.best_day_of_week,
                        "worst_day": analysis.worst_day_of_week,
                        "correlations": analysis.correlations,
                        "recommendations": analysis.recommendations
                    }

    def _perform_analysis(
        self,
        entries: List[MoodEntry],
        days: int
    ) -> MoodAnalysis:
        """Perform comprehensive mood analysis."""
        mood_ratings = [e.mood_rating for e in entries]
        avg_mood = sum(mood_ratings) / len(mood_ratings)
        
        # Calculate trend
        if len(mood_ratings) >= 4:
            first_half = mood_ratings[:len(mood_ratings)//2]
            second_half = mood_ratings[len(mood_ratings)//2:]
            avg_first = sum(first_half) / len(first_half)
            avg_second = sum(second_half) / len(second_half)
            
            if avg_second > avg_first + 0.5:
                trend = "improving"
            elif avg_second < avg_first - 0.5:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "insufficient data"
        
        # Calculate stability
        if len(mood_ratings) >= 3:
            variance = sum((r - avg_mood) ** 2 for r in mood_ratings) / len(mood_ratings)
            std_dev = variance ** 0.5
            
            if std_dev < 1.5:
                stability = "high"
            elif std_dev < 2.5:
                stability = "moderate"
            else:
                stability = "variable"
        else:
            stability = "insufficient data"
        
        # Count emotions
        emotion_counts: Dict[str, int] = {}
        for entry in entries:
            for emotion in entry.emotions:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        common_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Day of week analysis
        day_moods: Dict[str, List[int]] = {}
        for entry in entries:
            day = entry.timestamp.strftime("%A")
            if day not in day_moods:
                day_moods[day] = []
            day_moods[day].append(entry.mood_rating)
        
        day_averages = {day: sum(moods)/len(moods) for day, moods in day_moods.items()}
        best_day = max(day_averages, key=day_averages.get) if day_averages else None
        worst_day = min(day_averages, key=day_averages.get) if day_averages else None
        
        # Find correlations
        correlations = self._find_correlations(entries)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            avg_mood, trend, stability, common_emotions
        )
        
        return MoodAnalysis(
            period_days=days,
            total_entries=len(entries),
            average_mood=avg_mood,
            mood_trend=trend,
            common_emotions=common_emotions,
            best_day_of_week=best_day,
            worst_day_of_week=worst_day,
            mood_stability=stability,
            correlations=correlations,
            recommendations=recommendations
        )

    def _find_correlations(self, entries: List[MoodEntry]) -> List[str]:
        """Find correlations between activities/context and mood."""
        correlations = []
        
        # Activity correlations
        activity_moods: Dict[str, List[int]] = {}
        for entry in entries:
            for activity in entry.activities:
                for category, keywords in self.ACTIVITY_CATEGORIES.items():
                    if any(kw in activity.lower() for kw in keywords):
                        if category not in activity_moods:
                            activity_moods[category] = []
                        activity_moods[category].append(entry.mood_rating)
        
        overall_avg = sum(e.mood_rating for e in entries) / len(entries)
        
        for activity, moods in activity_moods.items():
            if len(moods) >= 3:
                avg = sum(moods) / len(moods)
                if avg > overall_avg + 1:
                    correlations.append(f"{activity.title()} activities correlate with better mood (+{avg-overall_avg:.1f})")
        
        # Sleep correlation
        sleep_entries = [(e.sleep_hours, e.mood_rating) for e in entries if e.sleep_hours]
        if len(sleep_entries) >= 5:
            high_sleep = [m for s, m in sleep_entries if s and s >= 7]
            low_sleep = [m for s, m in sleep_entries if s and s < 6]
            
            if high_sleep and low_sleep:
                high_avg = sum(high_sleep) / len(high_sleep)
                low_avg = sum(low_sleep) / len(low_sleep)
                if high_avg > low_avg + 1:
                    correlations.append(f"Good sleep (7+ hours) correlates with better mood (+{high_avg-low_avg:.1f})")
        
        return correlations

    def _generate_recommendations(
        self,
        avg_mood: float,
        trend: str,
        stability: str,
        common_emotions: List[Tuple[str, int]]
    ) -> List[str]:
        """Generate personalized recommendations based on analysis."""
        recommendations = []
        
        if avg_mood < 4:
            recommendations.append(
                "Your mood has been low. Consider reaching out to a mental health professional for support."
            )
        
        if trend == "declining":
            recommendations.append(
                "I notice a downward trend. Tracking specific triggers might help identify patterns."
            )
        
        if stability == "variable":
            recommendations.append(
                "Your mood shows high variability. Regular routines and sleep patterns may help stabilize."
            )
        
        # Emotion-based recommendations
        negative_emotions = [e for e, c in common_emotions 
                          if e in self.EMOTION_CATEGORIES["negative"]]
        
        if "anxious" in negative_emotions:
            recommendations.append(
                "Anxiety appears frequently. Breathing exercises and grounding techniques may help."
            )
        
        if "lonely" in negative_emotions:
            recommendations.append(
                "Loneliness is common in your entries. Consider reaching out to supportive people."
            )
        
        if avg_mood >= 6 and trend in ["stable", "improving"]:
            recommendations.append(
                "You're doing well! Keep noting what contributes to your positive moments."
            )
        
        return recommendations

    async def get_latest(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get the most recent mood entry."""
        await self.initialize()
        
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("""
                SELECT * FROM mood_entries 
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT 1
            """, (user_id,)) as cursor:
                row = await cursor.fetchone()
                
                if not row:
                    return None
                
                entry_data = dict(row)
                return {
                    "mood_rating": entry_data['mood_rating'],
                    "mood_level": MoodLevel(entry_data['mood_rating']).name.replace("_", " ").title(),
                    "energy_level": entry_data['energy_level'],
                    "emotions": json.loads(entry_data['emotions'] or '[]'),
                    "timestamp": entry_data['timestamp'],
                    "hours_ago": (datetime.utcnow() - datetime.fromisoformat(entry_data['timestamp'])).seconds // 3600
                }

    def get_metrics(self) -> Dict[str, Any]:
        """Return tool metrics for observability."""
        return {
            "tool": "mood_tracker",
            "entries_logged": self._entries_logged,
            "analyses_performed": self._analyses_performed
        }

