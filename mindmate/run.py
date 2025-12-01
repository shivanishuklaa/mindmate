#!/usr/bin/env python3
"""
MindMate Runner Script

Simple script to run MindMate in different modes:
- server: Run the FastAPI server
- cli: Interactive command-line interface
- eval: Run evaluation suite
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


def run_server(host: str = "0.0.0.0", port: int = 8080, reload: bool = False):
    """Run the FastAPI server."""
    import uvicorn
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


async def run_cli():
    """Run interactive CLI mode."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.markdown import Markdown
    
    from config import config
    from memory import SessionService, LongTermMemory
    from workflows import MainRouter
    
    console = Console()
    
    console.print(Panel.fit(
        "[bold green]🧠 MindMate[/bold green]\n"
        "[dim]Mental Health Multi-Agent Support System[/dim]\n\n"
        "Type your message and press Enter.\n"
        "Commands: /quit, /mood, /journal, /help",
        title="Welcome",
        border_style="green"
    ))
    
    # Initialize services
    console.print("[dim]Initializing agents...[/dim]")
    
    session_service = SessionService()
    long_term_memory = LongTermMemory()
    await long_term_memory.initialize()
    
    router = MainRouter(
        session_service=session_service,
        long_term_memory=long_term_memory
    )
    
    # Create session
    session = await session_service.create_session(user_id="cli_user")
    console.print(f"[dim]Session started: {session.id[:8]}...[/dim]\n")
    
    console.print("[bold yellow]⚠️ Disclaimer:[/bold yellow] I'm an AI assistant providing emotional support, "
                  "not a licensed mental health professional. If you're in crisis, please contact "
                  "emergency services or a crisis helpline (988 in the US).\n")
    
    while True:
        try:
            user_input = console.input("[bold cyan]You:[/bold cyan] ")
            
            if not user_input.strip():
                continue
            
            # Handle commands
            if user_input.lower() == "/quit":
                console.print("\n[dim]Ending session. Take care! 💚[/dim]")
                await session_service.end_session(session.id)
                break
            
            if user_input.lower() == "/help":
                console.print(Panel(
                    "**Commands:**\n"
                    "- `/quit` - End session and exit\n"
                    "- `/mood` - Log your current mood\n"
                    "- `/journal` - Write a journal entry\n"
                    "- `/help` - Show this help\n\n"
                    "Just type normally to chat with MindMate.",
                    title="Help"
                ))
                continue
            
            if user_input.lower() == "/mood":
                mood = console.input("[dim]Rate your mood (1-10):[/dim] ")
                try:
                    mood_rating = int(mood)
                    result = await router.mood_tracker.write(
                        user_id="cli_user",
                        mood_rating=mood_rating
                    )
                    console.print(f"[green]✓ Mood logged: {result.get('mood_level', mood_rating)}[/green]\n")
                except ValueError:
                    console.print("[red]Please enter a number 1-10[/red]\n")
                continue
            
            if user_input.lower() == "/journal":
                console.print("[dim]Write your journal entry (press Enter twice to finish):[/dim]")
                lines = []
                while True:
                    line = console.input()
                    if line == "":
                        break
                    lines.append(line)
                
                if lines:
                    content = "\n".join(lines)
                    await router.journal_tool.execute(
                        action="write",
                        user_id="cli_user",
                        content=content,
                        session_id=session.id
                    )
                    console.print("[green]✓ Journal entry saved[/green]\n")
                continue
            
            # Process through router
            with console.status("[dim]Thinking...[/dim]"):
                result = await router.process_message(
                    user_input=user_input,
                    session_id=session.id
                )
            
            # Display response
            agents = ", ".join(result.agents_used) if result.agents_used else "unknown"
            console.print(f"\n[bold green]MindMate[/bold green] [dim]({agents})[/dim]:")
            console.print(Markdown(result.response))
            console.print()
            
        except KeyboardInterrupt:
            console.print("\n\n[dim]Session interrupted. Take care! 💚[/dim]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


async def run_eval(output_dir: str = "./eval_results"):
    """Run the evaluation suite."""
    from rich.console import Console
    from pathlib import Path
    
    from evaluation.crisis_eval import run_all_evaluations
    
    console = Console()
    output_path = Path(output_dir)
    
    console.print("[bold]Running MindMate Evaluation Suite[/bold]\n")
    
    results = await run_all_evaluations(output_path)
    
    console.print("\n[bold green]✓ Evaluation Complete[/bold green]")
    console.print(f"Results saved to: {output_path}")
    
    # Print summary
    if "empathy" in results:
        emp = results["empathy"]["aggregate_metrics"]
        console.print(f"\n[bold]Empathy Evaluation:[/bold]")
        console.print(f"  Overall Score: {emp['average_overall']:.1%}")
        console.print(f"  Pass Rate: {emp['pass_rate']:.1%}")
    
    if "crisis" in results:
        crisis = results["crisis"]["critical_metrics"]
        console.print(f"\n[bold]Crisis Detection:[/bold]")
        console.print(f"  Recall: {crisis['recall']:.1%}")
        console.print(f"  False Negative Rate: {crisis['false_negative_rate']:.1%}")


def main():
    parser = argparse.ArgumentParser(
        description="MindMate - Mental Health Multi-Agent System"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Server command
    server_parser = subparsers.add_parser("server", help="Run the API server")
    server_parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    server_parser.add_argument("--port", type=int, default=8080, help="Port to listen on")
    server_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    # CLI command
    subparsers.add_parser("cli", help="Interactive CLI mode")
    
    # Eval command
    eval_parser = subparsers.add_parser("eval", help="Run evaluation suite")
    eval_parser.add_argument("--output", default="./eval_results", help="Output directory")
    
    args = parser.parse_args()
    
    if args.command == "server":
        run_server(host=args.host, port=args.port, reload=args.reload)
    elif args.command == "cli":
        asyncio.run(run_cli())
    elif args.command == "eval":
        asyncio.run(run_eval(output_dir=args.output))
    else:
        # Default: run server
        run_server()


if __name__ == "__main__":
    main()

